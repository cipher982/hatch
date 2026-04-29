"""Surfaced model aliases for Hatch's OpenCode-backed commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


CodexModelAlias = Literal["nano", "mini", "max"]
ClaudeModelAlias = Literal["haiku", "sonnet", "opus"]
OpenRouterModelAlias = Literal["deepseek-v4-pro"]


@dataclass(frozen=True)
class SurfacedProvider:
    """Public provider surface that resolves to an OpenCode model."""

    backend: str
    label: str
    tool_name: str
    surface_name: str
    models: dict[str, str]


OPENROUTER_DEEPSEEK_V4_PRO = "openrouter/deepseek/deepseek-v4-pro"

SURFACED_PROVIDERS: dict[str, SurfacedProvider] = {
    "codex": SurfacedProvider(
        backend="opencode",
        label="Codex",
        tool_name="hatch_codex",
        surface_name="hatch codex",
        models={
            "nano": "openai/gpt-5.4-nano",
            "mini": "openai/gpt-5.4-mini",
            "max": "openai/gpt-5.5",
        },
    ),
    "claude": SurfacedProvider(
        backend="opencode",
        label="Claude",
        tool_name="hatch_claude",
        surface_name="hatch claude",
        models={
            "haiku": "amazon-bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "sonnet": "amazon-bedrock/us.anthropic.claude-sonnet-4-6",
            "opus": "amazon-bedrock/us.anthropic.claude-opus-4-7",
        },
    ),
    "openrouter": SurfacedProvider(
        backend="opencode",
        label="OpenRouter",
        tool_name="hatch_openrouter",
        surface_name="hatch openrouter",
        models={
            "deepseek-v4-pro": OPENROUTER_DEEPSEEK_V4_PRO,
        },
    ),
}

TOOL_TO_PROVIDER = {
    spec.tool_name: provider for provider, spec in SURFACED_PROVIDERS.items()
}

SURFACE_LABELS = {
    "hatch_default": "Agent",
    "hatch_gemini": "Gemini",
    **{spec.tool_name: spec.label for spec in SURFACED_PROVIDERS.values()},
}

SURFACE_NAMES = {
    "hatch_default": "hatch",
    "hatch_gemini": "hatch -b gemini",
    **{spec.tool_name: spec.surface_name for spec in SURFACED_PROVIDERS.values()},
}


def model_choices(provider: str) -> str:
    """Return a stable comma-separated alias list for errors/help."""
    spec = SURFACED_PROVIDERS[provider]
    return ", ".join(spec.models)


def tool_model_choices(tool_name: str) -> str:
    """Return model aliases for a surfaced MCP tool."""
    return model_choices(TOOL_TO_PROVIDER[tool_name])


def resolve_provider_model(provider: str, model_alias: str) -> str | None:
    """Resolve a CLI provider alias to an OpenCode model ID."""
    return SURFACED_PROVIDERS[provider].models.get(model_alias)


def resolve_tool_model(tool_name: str, model_alias: str) -> str | None:
    """Resolve an MCP tool model alias to an OpenCode model ID."""
    provider = TOOL_TO_PROVIDER.get(tool_name)
    if provider is None:
        return None
    return resolve_provider_model(provider, model_alias)


def opencode_progress_label(model_name: str) -> str:
    """Map OpenCode model IDs to user-facing progress labels."""
    if model_name.startswith("openai/"):
        return "Codex"
    if model_name.startswith("amazon-bedrock/") or model_name.startswith("anthropic/"):
        return "Claude"
    if model_name.startswith("google/") or model_name.startswith("gemini/"):
        return "Gemini"
    if model_name.startswith("openrouter/"):
        return "OpenRouter"
    return "Agent"
