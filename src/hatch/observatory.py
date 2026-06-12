"""Optional Agent Observatory runtime integration."""

from __future__ import annotations

import os
from pathlib import Path


OBSERVATORY_CA_ENV_VARS = ("NODE_EXTRA_CA_CERTS", "CODEX_CA_CERTIFICATE")


def observatory_ca_path(env: dict[str, str] | None = None) -> Path | None:
    """Return the installed Agent Observatory CA path when it is available."""
    env = env or os.environ
    for key in OBSERVATORY_CA_ENV_VARS:
        value = env.get(key)
        if value:
            path = Path(value).expanduser()
            if path.is_file():
                return path

    home = Path(env.get("HOME") or Path.home()).expanduser()
    path = home / ".local" / "state" / "agent-observatory" / "ca" / "observatory-ca.pem"
    return path if path.is_file() else None


def apply_observatory_trust_env(env: dict[str, str]) -> dict[str, str]:
    """Inject Observatory trust vars for child runtimes without replacing roots."""
    ca_path = observatory_ca_path(env)
    if ca_path is None:
        return env

    ca = str(ca_path)
    for key in OBSERVATORY_CA_ENV_VARS:
        value = env.get(key)
        if value and Path(value).expanduser().is_file():
            continue
        env[key] = ca
    return env


def observatory_trust_signature(env: dict[str, str]) -> tuple[tuple[str, str], ...]:
    """Stable signature of the Observatory trust values a child process should use."""
    ca_path = observatory_ca_path(env)
    if ca_path is None:
        return ()
    ca = str(ca_path)
    return tuple((key, env.get(key) or ca) for key in OBSERVATORY_CA_ENV_VARS)
