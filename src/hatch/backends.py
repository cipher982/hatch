"""Backend configurations for different AI agent CLIs.

Each backend knows how to configure environment variables and build commands
for its respective CLI tool.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import Any

from hatch.context import ExecutionContext
from hatch.context import detect_context


def _build_simple_claude_headless_cmd(
    *,
    output_format: str,
    include_partial_messages: bool,
    effort: str = "low",
) -> list[str]:
    """Build a minimal Claude headless command that avoids user profile overhead."""
    cmd = [
        "claude",
    ]

    # Claude CLI requires --verbose when using stream-json under --print.
    if output_format == "stream-json":
        cmd.append("--verbose")

    cmd.extend([
        "--print",
        "-",  # Read prompt from stdin
        "--output-format",
        output_format,
        "--dangerously-skip-permissions",
        "--setting-sources",
        "local",
        "--no-session-persistence",
        "--tools",
        "",
        "--effort",
        effort,
    ])

    if include_partial_messages:
        cmd.append("--include-partial-messages")

    return cmd


class Backend(str, Enum):
    """Supported agent backends."""

    ZAI = "zai"  # Claude Code CLI with z.ai/GLM-5.1
    BEDROCK = "bedrock"  # Claude Code CLI with AWS Bedrock
    CODEX = "codex"  # OpenAI Codex CLI
    GEMINI = "gemini"  # Google Gemini CLI
    OPENCODE = "opencode"  # OpenCode runtime with provider-native models


@dataclass
class BackendConfig:
    """Configuration produced by a backend."""

    cmd: list[str]
    env: dict[str, str]
    env_unset: list[str] = field(default_factory=list)
    stdin_data: bytes | None = None  # Prompt via stdin to avoid ARG_MAX limits

    def build_env(self) -> dict[str, str]:
        """Build final environment dict."""
        # Start with current environment
        result = dict(os.environ)

        # Remove vars that need to be unset
        for key in self.env_unset:
            result.pop(key, None)

        # Apply backend-specific vars
        result.update(self.env)

        return result


def configure_zai(
    prompt: str,
    ctx: ExecutionContext | None = None,
    *,
    api_key: str | None = None,
    base_url: str = "https://api.z.ai/api/anthropic",
    model: str = "glm-5.1",
    resume: str | None = None,
    output_format: str = "text",
    include_partial_messages: bool = False,
    **_: Any,
) -> BackendConfig:
    """Configure z.ai backend (Claude Code CLI with GLM-5.1).

    Key insight: z.ai uses ANTHROPIC_AUTH_TOKEN (not ANTHROPIC_API_KEY),
    and requires CLAUDE_CODE_USE_BEDROCK to be unset.

    Prompt passed via stdin to avoid ARG_MAX limits on large prompts.
    """
    ctx = ctx or detect_context()

    key = api_key or os.environ.get("ZAI_API_KEY")
    if not key:
        raise ValueError("ZAI_API_KEY not set and no api_key provided")

    env = {
        "ANTHROPIC_BASE_URL": base_url,
        "ANTHROPIC_AUTH_TOKEN": key,  # NOT ANTHROPIC_API_KEY
        "ANTHROPIC_MODEL": model,
    }

    # Set HOME for containers with read-only filesystems
    if ctx.in_container and not ctx.home_writable:
        env["HOME"] = "/tmp"

    cmd = _build_simple_claude_headless_cmd(
        output_format=output_format,
        include_partial_messages=include_partial_messages,
    )

    # Add resume flag for session continuity
    if resume:
        cmd.extend(["--resume", resume])

    return BackendConfig(
        cmd=cmd,
        env=env,
        env_unset=["CLAUDE_CODE_USE_BEDROCK", "ANTHROPIC_API_KEY"],
        stdin_data=prompt.encode("utf-8"),
    )


def configure_bedrock(
    prompt: str,
    ctx: ExecutionContext | None = None,
    *,
    model: str = "us.anthropic.claude-sonnet-4-6",
    aws_profile: str = "zh-qa-engineer",
    aws_region: str = "us-east-1",
    resume: str | None = None,
    output_format: str = "text",
    include_partial_messages: bool = False,
    **_: Any,
) -> BackendConfig:
    """Configure Bedrock backend (Claude Code CLI with AWS Bedrock).

    Prompt passed via stdin to avoid ARG_MAX limits on large prompts.
    """
    ctx = ctx or detect_context()

    env = {
        "CLAUDE_CODE_USE_BEDROCK": "1",
        "AWS_PROFILE": aws_profile,
        "AWS_REGION": aws_region,
        "ANTHROPIC_MODEL": model,
    }

    # Set HOME for containers with read-only filesystems
    if ctx.in_container and not ctx.home_writable:
        env["HOME"] = "/tmp"

    cmd = _build_simple_claude_headless_cmd(
        output_format=output_format,
        include_partial_messages=include_partial_messages,
    )

    # Add resume flag for session continuity
    if resume:
        cmd.extend(["--resume", resume])

    return BackendConfig(
        cmd=cmd,
        env=env,
        env_unset=[
            "ANTHROPIC_AUTH_TOKEN",
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_BASE_URL",
        ],
        stdin_data=prompt.encode("utf-8"),
    )


def configure_codex(
    prompt: str,
    ctx: ExecutionContext | None = None,
    *,
    api_key: str | None = None,
    model: str | None = None,
    reasoning_effort: str | None = None,
    skip_git_repo_check: bool = False,
    full_auto: bool = True,
    **_: Any,
) -> BackendConfig:
    """Configure Codex backend (OpenAI Codex CLI).

    Uses `codex exec` subcommand for non-interactive mode.
    Prompt passed via stdin (using `-` as prompt arg) to avoid ARG_MAX limits.
    """
    ctx = ctx or detect_context()

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY not set and no api_key provided")

    env = {
        "OPENAI_API_KEY": key,
    }

    # Set HOME for containers (Codex writes to ~/.codex)
    if ctx.in_container and not ctx.home_writable:
        env["HOME"] = ctx.effective_home

    # Codex exec subcommand for non-interactive mode
    # `-` means read prompt from stdin
    cmd = [
        "codex",
        "exec",
        "-",  # Read prompt from stdin
    ]

    # Full auto mode for automatic execution without prompts
    if full_auto:
        cmd.append("--full-auto")

    # Model override if specified
    if model:
        cmd.extend(["-m", model])

    # Reasoning effort override — passed as -c config override
    if reasoning_effort:
        cmd.extend(["-c", f"model_reasoning_effort={reasoning_effort}"])

    if skip_git_repo_check:
        cmd.append("--skip-git-repo-check")

    return BackendConfig(cmd=cmd, env=env, stdin_data=prompt.encode("utf-8"))


def configure_gemini(
    prompt: str,
    ctx: ExecutionContext | None = None,
    *,
    model: str = "gemini-3-pro-preview",
    **_: Any,
) -> BackendConfig:
    """Configure Gemini backend (Google Gemini CLI).

    Uses OAuth - no API key needed.
    Prompt passed via stdin to avoid ARG_MAX limits on large prompts.
    """
    ctx = ctx or detect_context()

    env: dict[str, str] = {}

    # Set HOME for containers (Gemini writes to ~/.config)
    if ctx.in_container and not ctx.home_writable:
        env["HOME"] = ctx.effective_home

    # Gemini CLI reads from stdin when given -p -
    # --yolo bypasses confirmation prompts for tool use
    cmd = [
        "gemini",
        "--model",
        model,
        "--yolo",
        "-p",
        "-",  # Read prompt from stdin
    ]

    return BackendConfig(cmd=cmd, env=env, stdin_data=prompt.encode("utf-8"))


def _map_opencode_variant(model: str, reasoning_effort: str | None) -> str | None:
    """Map hatch reasoning-effort levels onto OpenCode model variants."""
    if not reasoning_effort:
        return None

    if model.startswith("openai/"):
        # GPT-5 family supports minimal/low/medium/high. Codex xhigh maps to high.
        if reasoning_effort == "xhigh":
            return "high"
        return reasoning_effort

    return None


def configure_opencode(
    prompt: str,
    ctx: ExecutionContext | None = None,
    *,
    model: str | None = None,
    api_key: str | None = None,
    reasoning_effort: str | None = None,
    agent: str | None = None,
    aws_profile: str | None = None,
    aws_region: str | None = None,
    pure: bool = True,
    **_: Any,
) -> BackendConfig:
    """Configure OpenCode for provider-native non-interactive execution."""
    ctx = ctx or detect_context()

    if not model:
        raise ValueError("OpenCode backend requires an explicit model")

    env: dict[str, str] = {}

    # Set HOME for containers (OpenCode writes under user config/state dirs).
    if ctx.in_container and not ctx.home_writable:
        env["HOME"] = ctx.effective_home

    if model.startswith("openai/") and api_key:
        env["OPENAI_API_KEY"] = api_key

    if model.startswith("amazon-bedrock/"):
        env["AWS_PROFILE"] = aws_profile or os.environ.get("AWS_PROFILE", "zh-qa-engineer")
        env["AWS_REGION"] = aws_region or os.environ.get("AWS_REGION", "us-east-1")

    cmd = ["opencode", "run"]

    if pure:
        cmd.append("--pure")

    cmd.extend(["--format", "json", "-m", model])

    variant = _map_opencode_variant(model, reasoning_effort)
    if variant:
        cmd.extend(["--variant", variant])

    if agent:
        cmd.extend(["--agent", agent])

    # OpenCode's non-interactive CLI accepts the prompt as argv, not stdin.
    cmd.append(prompt)

    return BackendConfig(cmd=cmd, env=env, stdin_data=None)


# Backend to configure function mapping
BACKEND_CONFIGURATORS = {
    Backend.ZAI: configure_zai,
    Backend.BEDROCK: configure_bedrock,
    Backend.CODEX: configure_codex,
    Backend.GEMINI: configure_gemini,
    Backend.OPENCODE: configure_opencode,
}


def get_config(
    backend: Backend,
    prompt: str,
    ctx: ExecutionContext | None = None,
    **kwargs: Any,
) -> BackendConfig:
    """Get configuration for a backend."""
    configurator = BACKEND_CONFIGURATORS[backend]
    return configurator(prompt, ctx, **kwargs)
