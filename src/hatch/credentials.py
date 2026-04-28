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


@dataclass(frozen=True)
class SecretSpec:
    """Secret lookup contract for a backend."""

    env_var: str
    project: str = _PERSONAL_PROJECT


SECRET_SPECS: dict[Backend | str, SecretSpec] = {
    Backend.ZAI: SecretSpec(env_var="ZAI_API_KEY"),
    Backend.CODEX: SecretSpec(env_var="OPENAI_API_KEY"),
    OPENROUTER_CREDENTIAL: SecretSpec(env_var="OPENROUTER_API_KEY"),
}


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
            timeout=15,
            check=False,
        )
    except Exception:
        return None

    if result.returncode != 0:
        return None

    value = result.stdout.strip()
    return value or None


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

    env_value = os.environ.get(spec.env_var, "").strip()
    if env_value:
        resolved["api_key"] = env_value
        return resolved

    helper_value = _load_secret_from_helper(spec)
    if helper_value:
        resolved["api_key"] = helper_value
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
    if model.startswith("z.ai/") or model.startswith("zai/"):
        return Backend.ZAI

    return None
