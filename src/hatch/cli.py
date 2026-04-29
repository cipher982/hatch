"""Command-line interface for hatch."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Sequence

from hatch import __version__
from hatch.backends import Backend
from hatch.backends import get_config
from hatch.context import detect_context
from hatch.credentials import credential_backend_for
from hatch.credentials import hydrate_backend_kwargs
from hatch.expert import DEFAULT_EXPERT_MODEL
from hatch.expert import run_expert_sync
from hatch.models import SURFACED_PROVIDERS
from hatch.models import model_choices
from hatch.models import opencode_progress_label
from hatch.models import resolve_provider_model
from hatch.runner import AgentResult
from hatch.runner import run_claude_stream_sync
from hatch.runner import run_opencode_stream_sync
from hatch.runner import run_sync

# Exit codes
EXIT_SUCCESS = 0
EXIT_AGENT_ERROR = 1
EXIT_TIMEOUT = 2
EXIT_NOT_FOUND = 3
EXIT_CONFIG_ERROR = 4
RAW_BACKEND_NAMES = ("bedrock", "codex", "gemini")
CLAUDE_BACKENDS = {Backend.BEDROCK}
OPENCODE_BACKENDS = {Backend.OPENCODE}
EXPLICIT_PROVIDER_MSG = (
    "No default model is configured. Choose one of: "
    "hatch codex <nano|mini|max>, "
    "hatch claude <haiku|sonnet|opus>, "
    "hatch openrouter deepseek-v4-pro"
)
ZAI_DISABLED_MSG = (
    "z.ai/GLM-5.1 is disabled because the coding plan has no active resource package; "
    "choose codex, claude, or openrouter instead"
)

FLAGS_WITH_VALUE = {
    "-b",
    "--backend",
    "-t",
    "--timeout",
    "-C",
    "--cwd",
    "--model",
    "--reasoning-effort",
    "--output-format",
    "--api-key",
    "-r",
    "--resume",
}
HELP_FLAGS = {"-h", "--help", "--advanced-help"}


def _has_explicit_flag(argv: Sequence[str], *names: str) -> bool:
    """Return True when argv already sets one of the named flags."""
    long_names = [name for name in names if name.startswith("--")]
    for token in argv:
        if token in names:
            return True
        if any(token.startswith(f"{name}=") for name in long_names):
            return True
    return False


def _find_alias_candidate_index(argv: Sequence[str]) -> int | None:
    """Find the first positional token that isn't an option value."""
    expecting_value = False
    for index, token in enumerate(argv):
        if expecting_value:
            expecting_value = False
            continue
        if token == "--":
            return None
        if token in FLAGS_WITH_VALUE:
            expecting_value = True
            continue
        if token.startswith("--"):
            if "=" in token:
                continue
            if token in FLAGS_WITH_VALUE:
                expecting_value = True
                continue
            continue
        if token.startswith("-") and token != "-":
            continue
        return index
    return None


def infer_machine_defaults(argv: Sequence[str], stdout_is_tty: bool) -> tuple[bool, bool]:
    """Infer json/automation defaults for real non-interactive CLI calls."""
    json_output = _has_explicit_flag(argv, "--json")
    automation = _has_explicit_flag(argv, "--automation")

    if stdout_is_tty:
        return json_output, automation

    return True, True


def _print_mcp_help() -> None:
    print(
        "usage: hatch mcp [doctor ...]\n\n"
        "  hatch mcp                 Run the hatch MCP server over stdio\n"
        "  hatch mcp doctor tools    Verify the server exposes the expected tools\n"
        "  hatch mcp doctor smoke --cwd /abs/path\n",
        file=sys.stderr,
    )


def create_expert_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hatch expert",
        description="Ask one slow synchronous expert question using the Responses API",
    )
    parser.add_argument("prompt", nargs="*", metavar="PROMPT")
    parser.add_argument(
        "--reasoning-effort",
        choices=["medium", "high", "xhigh"],
        default="medium",
        help="Reasoning effort: medium is fastest/cheapest valid, xhigh is slowest/deepest",
    )
    parser.add_argument(
        "--web-search",
        action="store_true",
        help="Allow the model to use OpenAI web search during the single call",
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=int,
        default=900,
        metavar="SECONDS",
        help="Timeout in seconds (default: 900)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("HATCH_EXPERT_MODEL", DEFAULT_EXPERT_MODEL),
        help=f"Responses API model (default: {DEFAULT_EXPERT_MODEL})",
    )
    parser.add_argument(
        "--api-key",
        metavar="KEY",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output JSON result instead of plain text",
    )
    return parser


def _dispatch_expert(raw_argv: Sequence[str]) -> int:
    parser = create_expert_parser()
    args = parser.parse_args(raw_argv[1:])
    if args.timeout <= 0:
        msg = "timeout must be > 0"
        if args.json_output:
            print(json.dumps({
                "ok": False,
                "status": "config_error",
                "output": "",
                "duration_ms": 0,
                "error": msg,
            }))
        else:
            print(f"Error: {msg}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    prompt = get_prompt(args)
    print(
        f"[hatch] expert call started: model={args.model} "
        f"reasoning={args.reasoning_effort} web_search={str(args.web_search).lower()}",
        file=sys.stderr,
        flush=True,
    )
    result = run_expert_sync(
        prompt=prompt,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        web_search=args.web_search,
        timeout_s=args.timeout,
        api_key=args.api_key,
    )
    if args.json_output:
        print(json.dumps(result.to_dict()))
    else:
        if result.ok:
            print(result.output.rstrip())
        else:
            print(f"Error: {result.error}", file=sys.stderr)
    if result.ok:
        return EXIT_SUCCESS
    if result.status == "timeout":
        return EXIT_TIMEOUT
    return EXIT_AGENT_ERROR


def _dispatch_special_command(raw_argv: Sequence[str]) -> int | None:
    if not raw_argv:
        return None

    if raw_argv[0] == "expert":
        return _dispatch_expert(raw_argv)

    if raw_argv[0] != "mcp":
        return None

    if len(raw_argv) == 1:
        from hatch.mcp.server import main as mcp_main

        mcp_main()
        return EXIT_SUCCESS

    subcommand = raw_argv[1]
    if subcommand in {"-h", "--help"}:
        _print_mcp_help()
        return EXIT_SUCCESS

    if subcommand == "doctor":
        from hatch.mcp.doctor import main as mcp_doctor_main

        return int(mcp_doctor_main(raw_argv[2:]))

    print(f"Error: unknown hatch mcp subcommand '{subcommand}'", file=sys.stderr)
    _print_mcp_help()
    return EXIT_CONFIG_ERROR


def normalize_argv(argv: Sequence[str] | None) -> list[str]:
    """Map surfaced provider/model aliases onto the existing backend flags."""
    normalized = list(argv or [])

    if _has_explicit_flag(normalized, "-b", "--backend"):
        return normalized

    has_explicit_model = _has_explicit_flag(normalized, "--model")

    first_positional = _find_alias_candidate_index(normalized)
    if first_positional is None:
        if has_explicit_model:
            return ["--backend", "opencode", *normalized]
        return normalized

    provider = normalized[first_positional]
    provider_spec = SURFACED_PROVIDERS.get(provider)
    if not provider_spec:
        if has_explicit_model:
            return ["--backend", "opencode", *normalized]
        return normalized

    before = normalized[:first_positional]
    after = normalized[first_positional + 1:]

    if not after:
        choices = model_choices(provider)
        raise ValueError(f"{provider} requires an explicit model: {choices}")

    if after[0].startswith("-"):
        if after[0] in HELP_FLAGS:
            return normalized
        if has_explicit_model:
            return [*before, "--backend", provider_spec.backend, *after]
        choices = model_choices(provider)
        raise ValueError(f"{provider} requires an explicit model: {choices}")

    model_alias = after[0]
    resolved_model = resolve_provider_model(provider, model_alias)
    if not resolved_model:
        choices = model_choices(provider)
        raise ValueError(f"invalid {provider} model '{model_alias}'. Choose one of: {choices}")
    after = after[1:]

    rewritten = [*before, "--backend", provider_spec.backend]
    if not has_explicit_model:
        rewritten.extend(["--model", resolved_model])
    rewritten.extend(after)
    return rewritten


def create_parser(*, show_advanced: bool = False) -> argparse.ArgumentParser:
    """Create the argument parser."""
    advanced_help = None if show_advanced else argparse.SUPPRESS
    parser = argparse.ArgumentParser(
        prog="hatch",
        usage=(
            'hatch claude <haiku|sonnet|opus> [OPTIONS] "prompt"\n'
            '       hatch codex <nano|mini|max> [OPTIONS] "prompt"\n'
            '       hatch openrouter <deepseek-v4-pro> [OPTIONS] "prompt"\n'
            '       hatch expert [OPTIONS] "prompt"'
        ),
        description="One headless CLI for Claude, Codex, Gemini, OpenRouter, and expert calls",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Start Here:
  hatch codex mini "Review this branch"
  hatch claude sonnet "Review this diff"
  hatch openrouter deepseek-v4-pro "Review this branch"
  hatch expert "Is this refactor direction sound?"

Surfaces:
  codex tiers     nano, mini, max
  claude tiers    haiku, sonnet, opus
  openrouter      deepseek-v4-pro
  expert          one synchronous GPT pro Responses API call

Advanced:
  hatch codex max --reasoning-effort low "Write unit tests"
  hatch expert --reasoning-effort xhigh --web-search "Evaluate this design"
  hatch -b gemini "Summarize this image"
  hatch mcp              # run the MCP server
  hatch codex mini --json "Analyze this" | jq .output
  hatch --advanced-help   # show raw/backend-specific flags

Environment Variables:
  OPENAI_API_KEY      API key for codex backend
  OPENROUTER_API_KEY  API key for OpenRouter models
  AWS_PROFILE         AWS profile for bedrock backend
  AWS_REGION          AWS region for bedrock backend
""",
    )

    parser.add_argument(
        "prompt",
        nargs="*",
        metavar="PROMPT",
        help="Prompt text (reads from stdin if '-' or omitted)",
    )

    parser.add_argument(
        "-b",
        "--backend",
        default=None,
        help="Advanced backend escape hatch",
    )

    parser.add_argument(
        "-t",
        "--timeout",
        type=int,
        default=900,
        metavar="SECONDS",
        help="Timeout in seconds (default: 900)",
    )

    parser.add_argument(
        "-C",
        "--cwd",
        metavar="DIR",
        help="Working directory for the agent (default: current directory)",
    )

    parser.add_argument(
        "--model",
        metavar="MODEL",
        help=advanced_help,
    )

    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high", "xhigh"],
        help="Optional reasoning effort for Codex/OpenAI models",
    )

    parser.add_argument(
        "--skip-git-repo-check",
        action="store_true",
        help=advanced_help,
    )

    parser.add_argument(
        "--output-format",
        choices=["text", "json", "stream-json"],
        default="text",
        help=advanced_help,
    )

    parser.add_argument(
        "--include-partial-messages",
        action="store_true",
        help=advanced_help,
    )

    parser.add_argument(
        "--api-key",
        metavar="KEY",
        help=advanced_help,
    )

    parser.add_argument(
        "-r",
        "--resume",
        metavar="SESSION_ID",
        help=advanced_help,
    )

    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output JSON result instead of plain text",
    )

    parser.add_argument(
        "--automation",
        action="store_true",
        help=advanced_help,
    )

    parser.add_argument(
        "--advanced-help",
        action="store_true",
        help="Show raw/backend-specific flags",
    )

    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    return parser


def get_prompt(args: argparse.Namespace) -> str:
    """Get prompt from args or stdin."""
    if not args.prompt or args.prompt == ["-"]:
        # Read from stdin
        if sys.stdin.isatty():
            print("Reading prompt from stdin (Ctrl+D to end):", file=sys.stderr)
        prompt = sys.stdin.read()
        if not prompt.strip():
            print("Error: Empty prompt", file=sys.stderr)
            sys.exit(EXIT_CONFIG_ERROR)
        return prompt
    return " ".join(args.prompt)


def result_to_exit_code(result: AgentResult) -> int:
    """Convert AgentResult to exit code."""
    if result.ok:
        return EXIT_SUCCESS
    if result.exit_code == -1:  # timeout
        return EXIT_TIMEOUT
    if result.exit_code == -2:  # CLI not found
        return EXIT_NOT_FOUND
    return EXIT_AGENT_ERROR


def main(argv: Sequence[str] | None = None) -> int:
    """Main entry point.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:])

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    special_exit_code = _dispatch_special_command(raw_argv)
    if special_exit_code is not None:
        return special_exit_code
    parser = create_parser(show_advanced=_has_explicit_flag(raw_argv, "--advanced-help"))
    try:
        normalized_argv = normalize_argv(raw_argv)
    except ValueError as e:
        json_output = (
            infer_machine_defaults(raw_argv, stdout_is_tty=sys.stdout.isatty())[0]
            if argv is None
            else _has_explicit_flag(raw_argv, "--json")
        )
        if json_output:
            print(json.dumps({
                "ok": False,
                "status": "config_error",
                "output": "",
                "exit_code": EXIT_CONFIG_ERROR,
                "duration_ms": 0,
                "error": str(e),
                "stderr": None,
            }))
        else:
            print(f"Error: {e}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    args = parser.parse_args(normalized_argv)
    if args.advanced_help:
        parser.print_help()
        return EXIT_SUCCESS

    if argv is None:
        args.json_output, args.automation = infer_machine_defaults(
            raw_argv,
            stdout_is_tty=sys.stdout.isatty(),
        )

    # Get prompt
    prompt = get_prompt(args)

    # Validate timeout
    if args.timeout <= 0:
        msg = "timeout must be > 0"
        if args.json_output:
            error_result = {
                "ok": False,
                "status": "config_error",
                "output": "",
                "exit_code": EXIT_CONFIG_ERROR,
                "duration_ms": 0,
                "error": msg,
                "stderr": None,
            }
            print(json.dumps(error_result))
        else:
            print(f"Error: {msg}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    # Validate cwd
    if args.cwd:
        cwd_path = Path(args.cwd)
        if not cwd_path.exists():
            msg = f"cwd does not exist: {args.cwd}"
        elif not cwd_path.is_dir():
            msg = f"cwd is not a directory: {args.cwd}"
        else:
            msg = ""
        if msg:
            if args.json_output:
                error_result = {
                    "ok": False,
                    "status": "config_error",
                    "output": "",
                    "exit_code": EXIT_CONFIG_ERROR,
                    "duration_ms": 0,
                    "error": msg,
                    "stderr": None,
                }
                print(json.dumps(error_result))
            else:
                print(f"Error: {msg}", file=sys.stderr)
            return EXIT_CONFIG_ERROR

    if args.backend is None:
        if args.json_output:
            print(json.dumps({
                "ok": False,
                "status": "config_error",
                "output": "",
                "exit_code": EXIT_CONFIG_ERROR,
                "duration_ms": 0,
                "error": EXPLICIT_PROVIDER_MSG,
                "stderr": None,
            }))
        else:
            print(f"Error: {EXPLICIT_PROVIDER_MSG}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    # Parse backend
    try:
        backend = Backend(args.backend)
    except ValueError:
        msg = (
            f"invalid backend '{args.backend}'. "
            f"Choose one of: {', '.join(RAW_BACKEND_NAMES)}"
        )
        if args.json_output:
            print(json.dumps({
                "ok": False,
                "status": "config_error",
                "output": "",
                "exit_code": EXIT_CONFIG_ERROR,
                "duration_ms": 0,
                "error": msg,
                "stderr": None,
            }))
        else:
            print(f"Error: {msg}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    if backend == Backend.ZAI:
        if args.json_output:
            print(json.dumps({
                "ok": False,
                "status": "config_error",
                "output": "",
                "exit_code": EXIT_CONFIG_ERROR,
                "duration_ms": 0,
                "error": ZAI_DISABLED_MSG,
                "stderr": None,
            }))
        else:
            print(f"Error: {ZAI_DISABLED_MSG}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    if backend in OPENCODE_BACKENDS and args.skip_git_repo_check:
        msg = "--skip-git-repo-check is not supported for surfaced providers"
        if args.json_output:
            print(json.dumps({
                "ok": False,
                "status": "config_error",
                "output": "",
                "exit_code": EXIT_CONFIG_ERROR,
                "duration_ms": 0,
                "error": msg,
                "stderr": None,
            }))
        else:
            print(f"Error: {msg}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    model_name = str(args.model or "")
    if backend in OPENCODE_BACKENDS and (
        model_name.startswith("zai/")
        or model_name.startswith("z.ai/")
        or model_name.startswith("zai-coding-plan/")
    ):
        msg = ZAI_DISABLED_MSG
        if args.json_output:
            print(json.dumps({
                "ok": False,
                "status": "config_error",
                "output": "",
                "exit_code": EXIT_CONFIG_ERROR,
                "duration_ms": 0,
                "error": msg,
                "stderr": None,
            }))
        else:
            print(f"Error: {msg}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    if args.reasoning_effort:
        reasoning_supported = (
            backend == Backend.CODEX
            or (backend == Backend.OPENCODE and model_name.startswith("openai/"))
        )
        if not reasoning_supported:
            msg = "--reasoning-effort only works with Codex models"
            if args.json_output:
                print(json.dumps({
                    "ok": False,
                    "status": "config_error",
                    "output": "",
                    "exit_code": EXIT_CONFIG_ERROR,
                    "duration_ms": 0,
                    "error": msg,
                    "stderr": None,
                }))
            else:
                print(f"Error: {msg}", file=sys.stderr)
            return EXIT_CONFIG_ERROR

    # Build backend kwargs
    backend_kwargs: dict = {}
    if args.model:
        backend_kwargs["model"] = args.model
    if args.api_key:
        backend_kwargs["api_key"] = args.api_key
    if args.reasoning_effort:
        backend_kwargs["reasoning_effort"] = args.reasoning_effort
    if args.skip_git_repo_check:
        backend_kwargs["skip_git_repo_check"] = True
    if args.resume:
        backend_kwargs["resume"] = args.resume
    use_internal_claude_stream = backend in CLAUDE_BACKENDS and args.output_format == "text"
    use_internal_opencode_stream = backend in OPENCODE_BACKENDS
    if use_internal_claude_stream:
        backend_kwargs["output_format"] = "stream-json"
        backend_kwargs["include_partial_messages"] = True
    elif args.output_format:
        backend_kwargs["output_format"] = args.output_format
    if args.include_partial_messages:
        backend_kwargs["include_partial_messages"] = True

    # Resolve credentials before building backend config.
    try:
        ctx = detect_context()
        credential_backend = credential_backend_for(backend, backend_kwargs)
        if credential_backend is None:
            resolved_backend_kwargs = dict(backend_kwargs)
        else:
            resolved_backend_kwargs = hydrate_backend_kwargs(credential_backend, backend_kwargs)
        config = get_config(backend, prompt, ctx, **resolved_backend_kwargs)
    except ValueError as e:
        if args.json_output:
            error_result = {
                "ok": False,
                "status": "config_error",
                "output": "",
                "exit_code": EXIT_CONFIG_ERROR,
                "duration_ms": 0,
                "error": str(e),
                "stderr": None,
            }
            print(json.dumps(error_result))
        else:
            print(f"Error: {e}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    # Build environment
    env = config.build_env()
    if getattr(args, "automation", False):
        env["LONGHOUSE_IS_SIDECHAIN"] = "1"
    cwd = args.cwd

    # Run the agent
    start = time.monotonic()
    opencode_error: str | None = None

    try:
        if use_internal_claude_stream:
            stream_result = run_claude_stream_sync(
                config.cmd,
                config.stdin_data,
                env,
                cwd,
                args.timeout,
                progress_handler=lambda message: print(message, file=sys.stderr, flush=True),
            )
            stdout = stream_result.final_output or ""
            raw_stdout = stream_result.stdout
            stderr = stream_result.stderr
            return_code = stream_result.return_code
            timed_out = stream_result.timed_out
        elif use_internal_opencode_stream:
            stream_result = run_opencode_stream_sync(
                config.cmd,
                config.stdin_data,
                env,
                cwd,
                args.timeout,
                progress_label=opencode_progress_label(model_name),
                progress_handler=lambda message: print(message, file=sys.stderr, flush=True),
            )
            stdout = stream_result.final_output or ""
            raw_stdout = stream_result.stdout
            stderr = stream_result.stderr
            return_code = stream_result.return_code
            timed_out = stream_result.timed_out
            opencode_error = stream_result.error_message
        else:
            raw_stdout = ""
            stdout, stderr, return_code, timed_out = run_sync(
                config.cmd,
                config.stdin_data,
                env,
                cwd,
                args.timeout,
            )
    except FileNotFoundError as e:
        duration_ms = int((time.monotonic() - start) * 1000)
        result = AgentResult(
            ok=False,
            output="",
            exit_code=-2,
            duration_ms=duration_ms,
            error=f"CLI not found: {e}",
        )
        if args.json_output:
            print(json.dumps(result.to_dict()))
        else:
            print(f"Error: {result.error}", file=sys.stderr)
        return EXIT_NOT_FOUND
    except Exception as e:
        duration_ms = int((time.monotonic() - start) * 1000)
        result = AgentResult(
            ok=False,
            output="",
            exit_code=-3,
            duration_ms=duration_ms,
            error=str(e),
        )
        if args.json_output:
            print(json.dumps(result.to_dict()))
        else:
            print(f"Error: {result.error}", file=sys.stderr)
        return EXIT_AGENT_ERROR

    duration_ms = int((time.monotonic() - start) * 1000)

    # Build result
    if timed_out:
        result = AgentResult(
            ok=False,
            output="",
            exit_code=-1,
            duration_ms=duration_ms,
            error=f"Agent timed out after {args.timeout}s",
        )
    elif return_code != 0:
        result = AgentResult(
            ok=False,
            output=raw_stdout or stdout,
            exit_code=return_code,
            duration_ms=duration_ms,
            error=stderr or f"Exit code {return_code}",
            stderr=stderr,
        )
    elif use_internal_opencode_stream and opencode_error:
        result = AgentResult(
            ok=False,
            output=raw_stdout,
            exit_code=0,
            duration_ms=duration_ms,
            error=opencode_error,
            stderr=stderr,
        )
    elif (use_internal_claude_stream or use_internal_opencode_stream) and not stdout.strip() and raw_stdout.strip():
        result = AgentResult(
            ok=False,
            output=raw_stdout,
            exit_code=0,
            duration_ms=duration_ms,
            error="Agent stream completed without a final result",
            stderr=stderr,
        )
    elif not stdout.strip():
        result = AgentResult(
            ok=False,
            output="",
            exit_code=0,
            duration_ms=duration_ms,
            error="Empty output from agent",
            stderr=stderr,
        )
    else:
        result = AgentResult(
            ok=True,
            output=stdout,
            exit_code=0,
            duration_ms=duration_ms,
            stderr=stderr,
        )

    # Output
    if args.json_output:
        print(json.dumps(result.to_dict()))
    else:
        if result.ok:
            # Strip trailing whitespace for cleaner output
            print(result.output.rstrip())
        else:
            print(f"Error: {result.error}", file=sys.stderr)
            if result.output:
                print(result.output, file=sys.stderr)

    return result_to_exit_code(result)


if __name__ == "__main__":
    sys.exit(main())
