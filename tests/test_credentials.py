"""Tests for backend credential hydration."""

from __future__ import annotations

import os
from unittest import mock

import pytest

from hatch.backends import Backend
from hatch.credentials import OPENROUTER_CREDENTIAL
from hatch.credentials import credential_backend_for
from hatch.credentials import ensure_opencode_credentials
from hatch.credentials import hydrate_backend_kwargs
from hatch.credentials import resolve_env_secret


class TestHydrateBackendKwargs:
    """Tests for canonical credential loading."""

    def test_returns_input_for_backend_without_secret(self):
        """Gemini does not require credential hydration."""
        assert hydrate_backend_kwargs(Backend.GEMINI, {"model": "x"}) == {"model": "x"}

    def test_preserves_explicit_api_key(self):
        """Explicit CLI override wins."""
        kwargs = hydrate_backend_kwargs(Backend.CODEX, {"api_key": "explicit"})
        assert kwargs["api_key"] == "explicit"

    def test_uses_helper_before_ambient_openai_env(self):
        """Codex uses the stable helper before ambient shell env."""
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}, clear=False):
            with mock.patch("hatch.credentials._load_secret_from_helper", return_value="helper-key"):
                kwargs = hydrate_backend_kwargs(Backend.CODEX, {})
        assert kwargs["api_key"] == "helper-key"

    def test_uses_env_when_helper_missing(self):
        """Ambient env remains a fallback when the helper is unavailable."""
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}, clear=False):
            with mock.patch("hatch.credentials._load_secret_from_helper", return_value=None):
                kwargs = hydrate_backend_kwargs(Backend.CODEX, {})
        assert kwargs["api_key"] == "env-key"

    def test_uses_marked_roulette_key(self):
        """The shell roulette wrapper has an explicit one-process handoff."""
        with mock.patch.dict(
            os.environ,
            {
                "HATCH_CREDENTIAL_ROULETTE": "1",
                "HATCH_ROULETTE_OPENAI_API_KEY": "roulette-key",
                "OPENAI_API_KEY": "ambient-key",
            },
            clear=False,
        ):
            kwargs = hydrate_backend_kwargs(Backend.CODEX, {})
        assert kwargs["api_key"] == "roulette-key"

    def test_uses_helper_when_env_missing(self):
        """Canonical helper is used when env is absent."""
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            with mock.patch("hatch.credentials._load_secret_from_helper", return_value="helper-key") as loader:
                kwargs = hydrate_backend_kwargs(Backend.CODEX, {})

        loader.assert_called_once()
        assert kwargs["api_key"] == "helper-key"

    def test_raises_clear_error_when_secret_unavailable(self):
        """Failure message points at the canonical helper path."""
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            with mock.patch("hatch.credentials._load_secret_from_helper", return_value=None):
                with pytest.raises(ValueError, match="from Infisical"):
                    hydrate_backend_kwargs(Backend.CODEX, {})

    def test_openrouter_uses_openrouter_env(self):
        """OpenRouter models use the OpenRouter credential policy."""
        backend = credential_backend_for(
            Backend.OPENCODE,
            {"model": "openrouter/deepseek/deepseek-v4-pro"},
        )
        assert backend == OPENROUTER_CREDENTIAL

        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-env"}, clear=False):
            kwargs = hydrate_backend_kwargs(backend, {})
        assert kwargs["api_key"] == "sk-or-env"

    def test_cursor_skips_infisical_hydration(self):
        """Cursor uses local login; no Infisical credential policy."""
        assert credential_backend_for(Backend.CURSOR, {"model": "grok-4.5-fast-xhigh"}) is None
        assert credential_backend_for(Backend.CLAUDE, {"model": "sonnet"}) is None

    def test_resolve_env_secret_ignores_whitespace_env(self):
        with mock.patch("hatch.credentials._load_secret_from_helper", return_value="sk-or-helper"):
            value = resolve_env_secret(
                OPENROUTER_CREDENTIAL,
                {"OPENROUTER_API_KEY": "   "},
            )
        assert value == "sk-or-helper"

    def test_ensure_opencode_credentials_raises_for_missing_openrouter(self):
        with mock.patch("hatch.credentials.resolve_env_secret", return_value=None):
            with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
                ensure_opencode_credentials(
                    "openrouter/deepseek/deepseek-v4-pro",
                    {},
                )
