"""Credential resolution for agent backends.

Centralizes how hatch finds provider credentials so backend config builders
stay focused on command/env shape instead of secret fetching policy.

Credentials are cached locally — one `infisical secrets` call fetches all
keys for the personal-shell project, and individual keys are served from a
persisted cache that falls back to stale data when Infisical is unreachable.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from hatch.backends import Backend

_PERSONAL_PROJECT = "personal-shell"
OPENROUTER_CREDENTIAL = "openrouter"
_ROULETTE_MARKER = "HATCH_CREDENTIAL_ROULETTE"

_INFISICAL_CLI = "infisical"
_INFISICAL_PROJECT_ID = "a3f40ca4-1a1f-4499-be6b-8a4e96b3a3cf"
_INFISICAL_DOMAIN = "https://secrets.drose.io"
_INFISICAL_ENV = "dev"
_INFISICAL_PATH = "/"

_CACHE_DIR = Path.home() / ".cache" / "hatch"
_CACHE_FILE = _CACHE_DIR / "credentials.json"
_FRESH_TTL = 3600
_STALE_TTL = 86400


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


class CredentialCache:
    """Cached Infisical secret store with local file persistence.

    One ``infisical secrets`` call fetches all secrets for the personal-shell
    project. Individual key lookups are served from the in-memory dict.
    On transient Infisical failures, stale cache (up to 24h) is used as
    fallback instead of failing hard.
    """

    _secrets: dict[str, str] | None = None
    _fetched_at: float = 0

    @classmethod
    def get(cls, key: str) -> str | None:
        if os.environ.get("HATCH_DISABLE_SECRET_HELPER", "").strip() == "1":
            return None

        now = time.time()

        if cls._secrets is not None and (now - cls._fetched_at) < _FRESH_TTL:
            return cls._secrets.get(key)

        cached = cls._load_file()
        if cached and cached[1] > cls._fetched_at:
            cls._secrets, cls._fetched_at = cached

        if cls._secrets is not None and (now - cls._fetched_at) < _FRESH_TTL:
            return cls._secrets.get(key)

        try:
            cls._refresh()
            cls._write_file()
        except Exception:
            pass

        if cls._secrets is not None and (now - cls._fetched_at) < _STALE_TTL:
            return cls._secrets.get(key)

        return None

    @classmethod
    def _refresh(cls) -> None:
        token = cls._resolve_token()
        result = subprocess.run(
            [
                _INFISICAL_CLI, "secrets",
                "--projectId", _INFISICAL_PROJECT_ID,
                "--token", token,
                "--env", _INFISICAL_ENV,
                "--path", _INFISICAL_PATH,
                "--output", "json",
                "--silent",
                "--domain", _INFISICAL_DOMAIN,
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
            env=_helper_subprocess_env(),
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()[:200]
            raise RuntimeError(f"infisical exited {result.returncode}: {detail}")
        items = json.loads(result.stdout)
        cls._secrets = {
            i["secretKey"]: str(i["secretValue"])
            for i in items
            if isinstance(i, dict) and i.get("secretKey") and i.get("secretValue") is not None
        }
        cls._fetched_at = time.time()

    @classmethod
    def _load_file(cls) -> tuple[dict[str, str], float] | None:
        try:
            data = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
            secrets = data.get("secrets", {})
            fetched_at = data.get("fetched_at", 0)
            if isinstance(secrets, dict) and isinstance(fetched_at, (int, float)):
                return secrets, float(fetched_at)
        except Exception:
            pass
        return None

    @classmethod
    def _write_file(cls) -> None:
        if cls._secrets is None:
            return
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({
            "secrets": cls._secrets,
            "fetched_at": cls._fetched_at,
        }, indent=2)
        _CACHE_FILE.write_text(payload, encoding="utf-8")

    @staticmethod
    def _resolve_token() -> str:
        env_token = os.environ.get("INFISICAL_TOKEN")
        if env_token:
            return env_token
        token_file = Path.home() / ".config" / "infisical" / "access-token"
        if token_file.is_file():
            token = token_file.read_text(encoding="utf-8").strip()
            if token:
                return token
        raise RuntimeError("No Infisical access token available")


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
    """Fetch a secret from the cached Infisical credential store."""
    return CredentialCache.get(spec.env_var)


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
    if backend != Backend.OPENCODE:
        return backend

    model = str(backend_kwargs.get("model") or "")
    if model.startswith("openai/"):
        return Backend.CODEX
    if model.startswith("openrouter/"):
        return OPENROUTER_CREDENTIAL
    return None
