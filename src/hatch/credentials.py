"""Credential resolution for agent backends.

Centralizes how hatch finds provider credentials so backend config builders
stay focused on command/env shape instead of secret fetching policy.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass

from hatch.backends import Backend

_INFISICAL_HELPER = os.path.expanduser("~/git/me/scripts/infisical-get.py")
_PERSONAL_PROJECT = "personal-shell"
OPENROUTER_CREDENTIAL = "openrouter"
_ROULETTE_MARKER = "HATCH_CREDENTIAL_ROULETTE"


@dataclass(frozen=True)
class SecretSpec:
    """Secret lookup contract for a backend."""

    env_var: str
    project: str = _PERSONAL_PROJECT


SECRET_SPECS: dict[Backend | str, SecretSpec] = {
    Backend.CODEX: SecretSpec(env_var="OPENAI_API_KEY"),
    OPENROUTER_CREDENTIAL: SecretSpec(env_var="OPENROUTER_API_KEY"),
}

_HELPER_PATH_PREFIXES = (
    "/opt/homebrew/bin",
    "/usr/local/bin",
    "/usr/bin",
    "/bin",
)


def _non_empty_env_value(env: dict[str, str] | os._Environ, name: str) -> str | None:
    """Return an env var only when it contains non-whitespace."""
    value = env.get(name)
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _helper_subprocess_env() -> dict[str, str]:
    """Ensure Infisical CLI discovery works under stripped MCP launch envs."""
    env = dict(os.environ)
    path_parts = [part for part in env.get("PATH", "").split(os.pathsep) if part]
    for prefix in _HELPER_PATH_PREFIXES:
        if prefix not in path_parts:
            path_parts.append(prefix)
    env["PATH"] = os.pathsep.join(path_parts)
    return env


def _load_secret_from_helper(spec: SecretSpec) -> str | None:
    """Fetch a secret via the canonical local helper."""
    if os.environ.get("HATCH_DISABLE_SECRET_HELPER", "").strip() == "1":
        return None
    if not os.path.exists(_INFISICAL_HELPER):
        return None

    try:
        result = subprocess.run(
            [
                sys.executable,
                _INFISICAL_HELPER,
                spec.env_var,
                "--project",
                spec.project,
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            env=_helper_subprocess_env(),
        )
    except Exception:
        return None

    if result.returncode != 0:
        return None

    value = result.stdout.strip()
    return value or None


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
        f"{spec.env_var} not set and hatch could not load it via "
        f"{_INFISICAL_HELPER} --project {spec.project}"
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
        f"{spec.env_var} not set and hatch could not load it via "
        f"{_INFISICAL_HELPER} --project {spec.project}"
    )


def credential_backend_for(
    backend: Backend,
    backend_kwargs: dict,
) -> Backend | str | None:
    """Choose which credential policy applies for a backend/model combination."""
    if backend != Backend.OPENCODE:
        return backend

    model = str(backend_kwargs.get("model") or "")
    if model.startswith("openai/"):
        return Backend.CODEX
    if model.startswith("openrouter/"):
        return OPENROUTER_CREDENTIAL
    return None
