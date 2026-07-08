"""Credential resolution for agent backends.

Centralizes how hatch finds provider credentials so backend config builders
stay focused on command/env shape instead of secret fetching policy.

The actual Infisical cache lives in David's shared local helper at
``~/git/me/scripts/infisical_cache.py``. Hatch deliberately delegates there so
agent runtimes share one cache, TTL policy, and machine-token path.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from hatch.backends import Backend

_PERSONAL_PROJECT = "personal-shell"
OPENROUTER_CREDENTIAL = "openrouter"
_ROULETTE_MARKER = "HATCH_CREDENTIAL_ROULETTE"
_SECRET_HELPER_DISABLE = "HATCH_DISABLE_SECRET_HELPER"
_DEFAULT_ME_SCRIPTS_DIR = Path.home() / "git" / "me" / "scripts"
_ME_SCRIPTS_DIR_ENV = "DROSE_ME_SCRIPTS_DIR"


@dataclass(frozen=True)
class SecretSpec:
    """Secret lookup contract for a backend."""

    env_var: str
    project: str = _PERSONAL_PROJECT


SECRET_SPECS: dict[Backend | str, SecretSpec] = {
    Backend.CODEX: SecretSpec(env_var="OPENAI_API_KEY"),
    OPENROUTER_CREDENTIAL: SecretSpec(env_var="OPENROUTER_API_KEY"),
}

class CredentialCache:
    """Compatibility facade over the canonical local Infisical cache."""

    @classmethod
    def get(cls, key: str, *, project: str = _PERSONAL_PROJECT) -> str | None:
        return _load_secret_from_canonical_cache(key, project=project)


def _non_empty_env_value(env: dict[str, str] | os._Environ, name: str) -> str | None:
    """Return an env var only when it contains non-whitespace."""
    value = env.get(name)
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _canonical_scripts_dir() -> Path:
    configured = os.environ.get(_ME_SCRIPTS_DIR_ENV, "").strip()
    return Path(configured).expanduser() if configured else _DEFAULT_ME_SCRIPTS_DIR


def _load_secret_from_canonical_cache(key: str, *, project: str) -> str | None:
    if os.environ.get(_SECRET_HELPER_DISABLE, "").strip() == "1":
        return None

    scripts_dir = _canonical_scripts_dir()
    if not scripts_dir.exists():
        return None
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    try:
        from infisical_cache import SecretCacheError
        from infisical_cache import get_secret
    except Exception:
        return None

    try:
        return get_secret(key, project=project, prefer_env=False)
    except SecretCacheError:
        return None


def _load_secret_from_helper(spec: SecretSpec) -> str | None:
    """Fetch a secret from the cached Infisical credential store."""
    return CredentialCache.get(spec.env_var, project=spec.project)


def resolve_env_secret(
    backend: Backend | str,
    env: dict[str, str],
) -> str | None:
    """Resolve a provider secret for an OpenCode subprocess env."""
    spec = SECRET_SPECS.get(backend)
    if not spec:
        return None

    existing = _non_empty_env_value(env, spec.env_var)
    if existing:
        return existing

    helper_value = _load_secret_from_helper(spec)
    if helper_value:
        return helper_value

    return _non_empty_env_value(os.environ, spec.env_var)


def credential_status() -> dict[str, bool]:
    """Report whether canonical provider secrets are currently resolvable."""
    return {
        spec.env_var: bool(resolve_env_secret(backend, {}))
        for backend, spec in SECRET_SPECS.items()
    }


def ensure_opencode_credentials(model: str, env: dict[str, str]) -> None:
    """Fail fast when an OpenCode model needs credentials that are unavailable."""
    if model.startswith("openrouter/"):
        backend: Backend | str = OPENROUTER_CREDENTIAL
    elif model.startswith("openai/"):
        backend = Backend.CODEX
    else:
        return

    spec = SECRET_SPECS[backend]
    if resolve_env_secret(backend, env):
        return

    raise ValueError(
        f"{spec.env_var} not set and hatch could not load it from Infisical"
    )


def hydrate_backend_kwargs(
    backend: Backend | str,
    backend_kwargs: dict,
) -> dict:
    """Return backend kwargs with canonical credentials populated when needed."""
    resolved = dict(backend_kwargs)

    if resolved.get("api_key"):
        return resolved

    spec = SECRET_SPECS.get(backend)
    if not spec:
        return resolved

    if os.environ.get(_ROULETTE_MARKER, "").strip() == "1":
        roulette_value = os.environ.get(f"HATCH_ROULETTE_{spec.env_var}", "").strip()
        if roulette_value:
            resolved["api_key"] = roulette_value
            return resolved

    if backend != Backend.CODEX:
        env_value = os.environ.get(spec.env_var, "").strip()
        if env_value:
            resolved["api_key"] = env_value
            return resolved

    helper_value = _load_secret_from_helper(spec)
    if helper_value:
        resolved["api_key"] = helper_value
        return resolved

    env_value = os.environ.get(spec.env_var, "").strip()
    if env_value:
        resolved["api_key"] = env_value
        return resolved

    raise ValueError(
        f"{spec.env_var} not set and hatch could not load it from Infisical"
    )


def credential_backend_for(
    backend: Backend,
    backend_kwargs: dict,
) -> Backend | str | None:
    """Choose which credential policy applies for a backend/model combination."""
    # Cursor uses local Cursor login (or optional CURSOR_API_KEY already in env).
    # Do not force Infisical hydration — fail closed to the CLI's own auth.
    if backend in {Backend.CLAUDE, Backend.CURSOR, Backend.GEMINI, Backend.BEDROCK}:
        return None

    if backend != Backend.OPENCODE:
        return backend

    model = str(backend_kwargs.get("model") or "")
    if model.startswith("openai/"):
        return Backend.CODEX
    if model.startswith("openrouter/"):
        return OPENROUTER_CREDENTIAL
    return None
