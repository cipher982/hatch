"""Surfaced model aliases for Hatch commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


CodexModelAlias = Literal["sol", "terra", "luna", "nano", "mini", "max"]
ClaudeModelAlias = Literal["haiku", "sonnet", "opus", "fable"]
CursorModelAlias = Literal["grok"]
OpenRouterModelAlias = Literal["deepseek-v4-pro"]


@dataclass(frozen=True)
class SurfacedProvider:
    """Public provider surface that resolves to a backend model name."""

    backend: str
    label: str
    surface_name: str
    models: dict[str, str]


OPENROUTER_DEEPSEEK_V4_PRO = "openrouter/deepseek/deepseek-v4-pro"
CURSOR_GROK = "cursor-grok-4.5-high"

SURFACED_PROVIDERS: dict[str, SurfacedProvider] = {
    "codex": SurfacedProvider(
        backend="opencode",
        label="Codex",
        surface_name="hatch codex",
        models={
            "sol": "openai/gpt-5.6-sol",
            "terra": "openai/gpt-5.6-terra",
            "luna": "openai/gpt-5.6-luna",
            # Compatibility aliases for prompts written before GPT-5.6.
            "nano": "openai/gpt-5.4-nano",
            "mini": "openai/gpt-5.4-mini",
            "max": "openai/gpt-5.5",
        },
    ),
    "claude": SurfacedProvider(
        backend="claude",
        label="Claude",
        surface_name="hatch claude",
        models={
            "haiku": "haiku",
            "sonnet": "sonnet",
            "opus": "opus",
            "fable": "fable",
        },
    ),
    "cursor": SurfacedProvider(
        backend="cursor",
        label="Cursor",
        surface_name="hatch cursor",
        models={
            # Grok 4.5 High via Cursor Agent CLI.
            "grok": CURSOR_GROK,
        },
    ),
    "openrouter": SurfacedProvider(
        backend="opencode",
        label="OpenRouter",
        surface_name="hatch openrouter",
        models={
            "deepseek-v4-pro": OPENROUTER_DEEPSEEK_V4_PRO,
        },
    ),
}

def model_choices(provider: str) -> str:
    """Return a stable comma-separated alias list for errors/help."""
    spec = SURFACED_PROVIDERS[provider]
    return ", ".join(spec.models)


def resolve_provider_model(provider: str, model_alias: str) -> str | None:
    """Resolve a CLI provider alias to an OpenCode model ID."""
    return SURFACED_PROVIDERS[provider].models.get(model_alias)


def opencode_progress_label(model_name: str) -> str:
    """Map OpenCode model IDs to user-facing progress labels."""
    if model_name.startswith("openai/"):
        return "Codex"
    if model_name.startswith("openrouter/") and ("/anthropic/" in model_name or "/~anthropic/" in model_name):
        return "Claude"
    if model_name.startswith("google/") or model_name.startswith("gemini/"):
        return "Gemini"
    if model_name.startswith("openrouter/"):
        return "OpenRouter"
    return "Agent"
