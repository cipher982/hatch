"""Longhouse origin metadata for Hatch-launched automation runs."""

from __future__ import annotations

import json
import contextlib
import os
from pathlib import Path
import uuid
from typing import Sequence

LONGHOUSE_HATCH_ORIGIN_KIND = "hatch_automation"
LONGHOUSE_PARENT_SESSION_ENVS = (
    "LONGHOUSE_MANAGED_SESSION_ID",
    "LONGHOUSE_SESSION_ID",
    "LONGHOUSE_CHANNEL_SESSION_ID",
)
OPENCODE_SESSION_METADATA_ROOT_ENV = "LONGHOUSE_OPENCODE_SESSION_METADATA_ROOT"


def _first_nonempty_env(env: dict[str, str], names: Sequence[str]) -> str | None:
    for name in names:
        value = str(env.get(name) or "").strip()
        if value:
            return value
    return None


def mark_longhouse_automation_env(env: dict[str, str]) -> None:
    """Mark a Hatch-launched provider run as Longhouse-hidden automation."""

    env["LONGHOUSE_IS_SIDECHAIN"] = "1"
    env["LONGHOUSE_ORIGIN_KIND"] = LONGHOUSE_HATCH_ORIGIN_KIND
    env["LONGHOUSE_HATCH_RUN_ID"] = f"hatch-{uuid.uuid4()}"
    env.setdefault(OPENCODE_SESSION_METADATA_ROOT_ENV, str(opencode_session_metadata_root(env)))

    parent_session_id = _first_nonempty_env(env, LONGHOUSE_PARENT_SESSION_ENVS)
    if parent_session_id:
        env["LONGHOUSE_PARENT_SESSION_ID"] = parent_session_id

    parent_thread_id = str(env.get("LONGHOUSE_THREAD_ID") or "").strip()
    if parent_thread_id:
        env["LONGHOUSE_PARENT_THREAD_ID"] = parent_thread_id

    parent_provider_session_id = str(env.get("LONGHOUSE_PROVIDER_SESSION_ID") or "").strip()
    if parent_provider_session_id:
        env["LONGHOUSE_PARENT_PROVIDER_SESSION_ID"] = parent_provider_session_id


def longhouse_home(env: dict[str, str] | None = None) -> Path:
    source = os.environ if env is None else env
    value = str(source.get("LONGHOUSE_HOME") or "").strip()
    if value:
        return Path(value).expanduser()
    return Path.home() / ".longhouse"


def opencode_session_metadata_root(env: dict[str, str] | None = None) -> Path:
    source = os.environ if env is None else env
    value = str(source.get(OPENCODE_SESSION_METADATA_ROOT_ENV) or "").strip()
    if value:
        return Path(value).expanduser()
    return longhouse_home(source).joinpath("provider-session-metadata", "opencode")


def write_opencode_origin_sidecar(provider_session_id: str, env: dict[str, str]) -> Path | None:
    """Persist Hatch origin for daemon-side OpenCode database shipping."""

    provider_session_id = str(provider_session_id or "").strip()
    if not provider_session_id or env.get("LONGHOUSE_ORIGIN_KIND") != LONGHOUSE_HATCH_ORIGIN_KIND:
        return None

    root = opencode_session_metadata_root(env)
    payload = {
        "artifact_kind": "longhouse_provider_session_metadata",
        "provider": "opencode",
        "provider_session_id": provider_session_id,
        "origin_kind": LONGHOUSE_HATCH_ORIGIN_KIND,
        "hatch_run_id": env.get("LONGHOUSE_HATCH_RUN_ID"),
        "parent_longhouse_session_id": env.get("LONGHOUSE_PARENT_SESSION_ID"),
        "parent_thread_id": env.get("LONGHOUSE_PARENT_THREAD_ID"),
        "parent_provider_session_id": env.get("LONGHOUSE_PARENT_PROVIDER_SESSION_ID"),
    }
    payload = {key: value for key, value in payload.items() if value not in (None, "")}
    path = root / f"{provider_session_id}.json"
    temp_path = root / f".{provider_session_id}.{uuid.uuid4().hex}.tmp"
    try:
        root.mkdir(parents=True, exist_ok=True)
        temp_path.write_text(json.dumps(payload, sort_keys=True) + "\n")
        temp_path.replace(path)
    except OSError:
        with contextlib.suppress(OSError):
            temp_path.unlink()
        return None
    return path


def opencode_session_id_from_event_line(line: str) -> str | None:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None
    if payload.get("type") != "step_start":
        return None
    session_id = str(payload.get("sessionID") or "").strip()
    return session_id or None


def maybe_write_opencode_origin_sidecar_from_line(
    line: str,
    env: dict[str, str],
    written_session_ids: set[str],
) -> Path | None:
    session_id = opencode_session_id_from_event_line(line)
    if not session_id or session_id in written_session_ids:
        return None
    path = write_opencode_origin_sidecar(session_id, env)
    if path is not None:
        written_session_ids.add(session_id)
    return path
