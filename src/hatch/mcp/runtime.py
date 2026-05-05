"""Persistent OpenCode-backed runtime for hatch MCP."""

from __future__ import annotations

import atexit
import json
import os
import queue
import shutil
import socket
import subprocess
import threading
import time
import urllib.parse
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Callable

from hatch.backends import Backend
from hatch.credentials import SECRET_SPECS
from hatch.credentials import OPENROUTER_CREDENTIAL
from hatch.credentials import _load_secret_from_helper
from hatch.models import SURFACE_LABELS
from hatch.models import SURFACE_NAMES
from hatch.models import TOOL_TO_PROVIDER
from hatch.models import resolve_tool_model
from hatch.models import tool_model_choices
from hatch.opencode_stream import OpenCodeStreamAccumulator


OPENCODE_BIN = os.environ.get("HATCH_MCP_OPENCODE_BIN", "opencode")
ATTACH_URL_ENV = "HATCH_MCP_OPENCODE_ATTACH_URL"
OPENCODE_PORT_ENV = "HATCH_MCP_OPENCODE_PORT"
OPENCODE_ROOT_ENV = "HATCH_MCP_OPENCODE_ROOT"
OPENCODE_CONFIG_ENV = "HATCH_MCP_OPENCODE_CONFIG"
DEFAULT_GEMINI_MODEL = os.environ.get(
    "HATCH_MCP_GEMINI_MODEL",
    "google/gemini-3-flash-preview",
)
SERVER_START_TIMEOUT_S = 15.0
DEFAULT_OPENCODE_ROOT = Path.home() / ".local" / "share" / "hatch" / "mcp-runtime"
DEFAULT_OPENCODE_CONFIG = Path(__file__).with_name("opencode.json")


@dataclass
class OpenCodeRunResult:
    """Captured result from an OpenCode JSON event stream subprocess."""

    stdout: str
    stderr: str
    return_code: int
    timed_out: bool
    final_output: str | None
    error_message: str | None
    attach_url: str
    model: str
    completed: bool = False
    event_types: tuple[str, ...] = ()
    session_id: str | None = None
    artifact_path: str | None = None


@dataclass
class RecoveredOpenCodeOutput:
    output: str
    completed: bool


class HatchMcpRuntimeError(RuntimeError):
    """Structured runtime error for hatch MCP."""


def _runtime_root() -> Path:
    configured = os.environ.get(OPENCODE_ROOT_ENV, "").strip()
    return Path(configured).expanduser() if configured else DEFAULT_OPENCODE_ROOT


def _runtime_config_path() -> Path:
    configured = os.environ.get(OPENCODE_CONFIG_ENV, "").strip()
    return Path(configured).expanduser() if configured else DEFAULT_OPENCODE_CONFIG


def _runtime_paths() -> dict[str, Path]:
    runtime_root = _runtime_root()
    return {
        "XDG_CONFIG_HOME": runtime_root / "config",
        "XDG_DATA_HOME": runtime_root / "data",
        "XDG_STATE_HOME": runtime_root / "state",
        "XDG_CACHE_HOME": runtime_root / "cache",
    }


def _ensure_runtime_paths() -> dict[str, Path]:
    runtime_paths = _runtime_paths()
    for path in runtime_paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return runtime_paths


def _run_artifacts_dir() -> Path:
    path = _runtime_paths()["XDG_CACHE_HOME"] / "hatch" / "mcp-runs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _maybe_load_secret(backend: Backend | str) -> str | None:
    spec = SECRET_SPECS.get(backend)
    if not spec:
        return None
    return _load_secret_from_helper(spec)


def _build_server_env() -> dict[str, str]:
    env = dict(os.environ)
    runtime_paths = _runtime_paths()

    if not env.get("OPENAI_API_KEY"):
        secret = _maybe_load_secret(Backend.CODEX)
        if secret:
            env["OPENAI_API_KEY"] = secret

    if not env.get("OPENROUTER_API_KEY"):
        secret = _maybe_load_secret(OPENROUTER_CREDENTIAL)
        if secret:
            env["OPENROUTER_API_KEY"] = secret

    env.setdefault("AWS_PROFILE", os.environ.get("AWS_PROFILE", "zh-qa-engineer"))
    env.setdefault("AWS_REGION", os.environ.get("AWS_REGION", "us-east-1"))
    for key, value in runtime_paths.items():
        env[key] = str(value)
    env["OPENCODE_CONFIG"] = str(_runtime_config_path())
    env.pop("OPENCODE_CONFIG_CONTENT", None)
    return env


def _build_run_env(model: str) -> dict[str, str]:
    env = _build_server_env()

    if not model.startswith("openai/"):
        env.pop("OPENAI_API_KEY", None)
    if not model.startswith("openrouter/"):
        env.pop("OPENROUTER_API_KEY", None)

    return env


def _opencode_event_summary(stdout: str) -> tuple[tuple[str, ...], str | None]:
    event_types: list[str] = []
    session_id: str | None = None

    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        payload_type = payload.get("type")
        if isinstance(payload_type, str):
            event_types.append(payload_type)
        if session_id is None and isinstance(payload.get("sessionID"), str):
            session_id = payload["sessionID"]

    return tuple(event_types), session_id


def _recover_attached_session_output(attach_url: str, session_id: str) -> RecoveredOpenCodeOutput | None:
    """Fetch final text from the attached OpenCode server when run stdout is incomplete."""
    encoded_session_id = urllib.parse.quote(session_id, safe="")
    url = f"{attach_url.rstrip('/')}/session/{encoded_session_id}/message"

    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return None

    if not isinstance(payload, list):
        return None

    for message in reversed(payload):
        if not isinstance(message, dict):
            continue
        info = message.get("info") or {}
        if not isinstance(info, dict) or info.get("role") != "assistant":
            continue

        chunks: list[str] = []
        completed = bool((info.get("time") or {}).get("completed")) if isinstance(info.get("time"), dict) else False
        parts = message.get("parts") or []
        if not isinstance(parts, list):
            continue

        for part in parts:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text" and isinstance(part.get("text"), str):
                chunks.append(part["text"])
            if part.get("type") == "step-finish" and part.get("reason") == "stop":
                completed = True

        text = "".join(chunks).strip()
        if text:
            return RecoveredOpenCodeOutput(output=text, completed=completed)

    return None


def _write_run_artifact(
    *,
    stdout: str,
    stderr: str,
    return_code: int,
    timed_out: bool,
    model: str,
    attach_url: str,
    cmd: list[str],
    event_types: tuple[str, ...],
    session_id: str | None,
) -> str:
    artifact_id = f"{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
    path = _run_artifacts_dir() / f"{artifact_id}.json"
    payload = {
        "return_code": return_code,
        "timed_out": timed_out,
        "model": model,
        "attach_url": attach_url,
        "cmd": cmd[:-1],
        "event_types": list(event_types),
        "session_id": session_id,
        "stdout": stdout,
        "stderr": stderr,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(path)


def _missing_final_error(result: OpenCodeRunResult) -> tuple[str, str]:
    if result.error_message:
        return "opencode_error", result.error_message

    if not result.stdout.strip():
        return "opencode_protocol_error", "OpenCode exited 0 without JSON output"

    event_types = result.event_types
    if event_types == ("step_start",):
        return (
            "opencode_protocol_error",
            "OpenCode exited 0 after step_start without text or step_finish",
        )

    if "text" not in event_types:
        return (
            "opencode_protocol_error",
            f"OpenCode exited 0 without text output (events: {', '.join(event_types) or 'none'})",
        )

    return "opencode_protocol_error", "OpenCode exited 0 without final text"


def _resolve_model(tool_name: str, model: str | None) -> str:
    if tool_name == "hatch_gemini":
        return DEFAULT_GEMINI_MODEL

    if tool_name not in TOOL_TO_PROVIDER:
        raise HatchMcpRuntimeError(f"Unknown hatch MCP tool: {tool_name}")
    if not model:
        raise HatchMcpRuntimeError(f"{tool_name} requires an explicit surfaced model")

    resolved = resolve_tool_model(tool_name, model)
    if not resolved:
        choices = tool_model_choices(tool_name)
        raise HatchMcpRuntimeError(f"Invalid model '{model}'. Choose one of: {choices}")
    return resolved


def _map_variant(model: str, reasoning_effort: str | None) -> str | None:
    if not reasoning_effort:
        return None
    if model.startswith("openai/"):
        if reasoning_effort == "xhigh":
            return "high"
        return reasoning_effort
    return None


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _healthcheck(url: str, timeout_s: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/global/health", timeout=timeout_s) as response:
            payload = json.load(response)
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return False
    return bool(payload.get("healthy"))


class OpenCodeServerManager:
    """Own a single persistent local OpenCode server per hatch MCP process."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[str] | None = None
        self._url: str | None = None
        self._managed = False
        atexit.register(self.shutdown)

    @property
    def current_url(self) -> str | None:
        return self._url

    def ensure_server(self) -> str:
        configured_url = os.environ.get(ATTACH_URL_ENV, "").strip()
        if configured_url:
            if self._managed and self._proc and self._proc.poll() is None:
                self.shutdown()
            if not _healthcheck(configured_url):
                raise HatchMcpRuntimeError(
                    f"Configured attach url is not healthy: {configured_url}"
                )
            self._url = configured_url
            self._managed = False
            return configured_url

        with self._lock:
            if self._proc and self._proc.poll() is None and self._url and _healthcheck(self._url):
                return self._url

            self._start_locked()
            assert self._url is not None
            return self._url

    def _start_locked(self) -> None:
        port = int(os.environ.get(OPENCODE_PORT_ENV) or _find_free_port())
        url = f"http://127.0.0.1:{port}"
        _ensure_runtime_paths()

        cmd = [
            OPENCODE_BIN,
            "serve",
            "--hostname",
            "127.0.0.1",
            "--port",
            str(port),
        ]
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=_build_server_env(),
            text=True,
        )

        deadline = time.monotonic() + SERVER_START_TIMEOUT_S
        while time.monotonic() < deadline:
            if _healthcheck(url):
                self._proc = proc
                self._url = url
                self._managed = True
                return
            if proc.poll() is not None:
                raise HatchMcpRuntimeError(f"OpenCode server exited during startup ({proc.returncode})")
            time.sleep(0.2)

        proc.terminate()
        raise HatchMcpRuntimeError("Timed out waiting for OpenCode server to become healthy")

    def shutdown(self) -> None:
        proc = self._proc
        if proc is None or proc.poll() is not None or not self._managed:
            return
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            proc.kill()


SERVER_MANAGER = OpenCodeServerManager()


def build_run_command(
    *,
    tool_name: str,
    prompt: str,
    attach_url: str,
    model: str | None = None,
    cwd: str | None = None,
    reasoning_effort: str | None = None,
) -> list[str]:
    if not prompt.strip():
        raise HatchMcpRuntimeError("prompt must not be empty")

    if cwd:
        cwd_path = Path(cwd)
        if not cwd_path.is_absolute():
            raise HatchMcpRuntimeError("cwd must be an absolute path")
        if not cwd_path.exists():
            raise HatchMcpRuntimeError(f"cwd does not exist: {cwd}")
        if not cwd_path.is_dir():
            raise HatchMcpRuntimeError(f"cwd is not a directory: {cwd}")

    cmd = [
        OPENCODE_BIN,
        "run",
        "--attach",
        attach_url,
        "--format",
        "json",
        "--pure",
    ]

    if cwd:
        cmd.extend(["--dir", cwd])

    resolved_model = _resolve_model(tool_name, model)
    cmd.extend(["-m", resolved_model])

    variant = _map_variant(resolved_model, reasoning_effort)
    if variant:
        cmd.extend(["--variant", variant])

    cmd.append(prompt)
    return cmd


def _stream_reader(
    stream: Any,
    source: str,
    output_queue: queue.Queue[tuple[str, str | None]],
) -> None:
    try:
        for line in iter(stream.readline, ""):
            output_queue.put((source, line))
    finally:
        output_queue.put((source, None))


def run_attached_command(
    cmd: list[str],
    *,
    model: str,
    attach_url: str,
    timeout_s: int,
    progress_label: str,
    progress_handler: Callable[[str], None] | None = None,
    heartbeat_s: int = 30,
) -> OpenCodeRunResult:
    _ensure_runtime_paths()
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_build_run_env(model),
        text=True,
        bufsize=1,
    )

    if proc.stdout is None or proc.stderr is None:
        raise HatchMcpRuntimeError("OpenCode process streams were not created")

    output_queue: queue.Queue[tuple[str, str | None]] = queue.Queue()
    readers = [
        threading.Thread(target=_stream_reader, args=(proc.stdout, "stdout", output_queue), daemon=True),
        threading.Thread(target=_stream_reader, args=(proc.stderr, "stderr", output_queue), daemon=True),
    ]
    for reader in readers:
        reader.start()

    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    accumulator = OpenCodeStreamAccumulator(progress_label=progress_label)
    start = time.monotonic()
    last_progress_at = start
    stdout_open = True
    stderr_open = True
    timed_out = False

    while stdout_open or stderr_open:
        elapsed = time.monotonic() - start
        remaining = timeout_s - elapsed
        if remaining <= 0:
            timed_out = True
            proc.kill()
            proc.wait()
            break

        try:
            source, line = output_queue.get(timeout=min(0.1, remaining))
        except queue.Empty:
            if progress_handler and proc.poll() is None and (time.monotonic() - last_progress_at) >= heartbeat_s:
                progress_handler(f"[hatch] still running ({int(time.monotonic() - start)}s)")
                last_progress_at = time.monotonic()
            continue

        if line is None:
            if source == "stdout":
                stdout_open = False
            else:
                stderr_open = False
            continue

        if source == "stdout":
            stdout_chunks.append(line)
            if progress_handler:
                for message in accumulator.handle_line(line):
                    progress_handler(message)
                    last_progress_at = time.monotonic()
            else:
                accumulator.handle_line(line)
        else:
            stderr_chunks.append(line)

    for reader in readers:
        reader.join(timeout=1)

    if not timed_out and proc.returncode is None:
        proc.wait()

    while True:
        try:
            source, line = output_queue.get_nowait()
        except queue.Empty:
            break
        if line is None:
            continue
        if source == "stdout":
            stdout_chunks.append(line)
            accumulator.handle_line(line)
        else:
            stderr_chunks.append(line)

    stdout = "".join(stdout_chunks)
    stderr = "".join(stderr_chunks)
    event_types, session_id = _opencode_event_summary(stdout)
    final_output = accumulator.final_output
    completed = accumulator.completed

    if not final_output and session_id:
        recovered = _recover_attached_session_output(attach_url, session_id)
        if recovered:
            final_output = recovered.output
            completed = recovered.completed

    return OpenCodeRunResult(
        stdout=stdout,
        stderr=stderr,
        return_code=-1 if timed_out else proc.returncode,
        timed_out=timed_out,
        final_output=final_output,
        error_message=accumulator.error_message,
        attach_url=attach_url,
        model=model,
        completed=completed,
        event_types=event_types,
        session_id=session_id,
    )


def doctor() -> dict[str, Any]:
    opencode_path = shutil.which(OPENCODE_BIN)
    attach_url = os.environ.get(ATTACH_URL_ENV, "").strip() or SERVER_MANAGER.current_url
    return {
        "ok": bool(opencode_path),
        "opencode_path": opencode_path,
        "attach_url": attach_url,
        "attach_url_healthy": bool(attach_url and _healthcheck(attach_url)),
        "gemini_model": DEFAULT_GEMINI_MODEL,
        "opencode_root": str(_runtime_root()),
        "opencode_config": str(_runtime_config_path()),
        "run_artifacts_dir": str(_run_artifacts_dir()),
    }


def run_surface(
    *,
    tool_name: str,
    prompt: str,
    model: str | None = None,
    cwd: str | None = None,
    timeout_s: int = 900,
    reasoning_effort: str | None = None,
    progress_handler: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    attach_url = SERVER_MANAGER.ensure_server()
    resolved_model = _resolve_model(tool_name, model)
    cmd = build_run_command(
        tool_name=tool_name,
        prompt=prompt,
        attach_url=attach_url,
        model=model,
        cwd=cwd,
        reasoning_effort=reasoning_effort,
    )

    start = time.perf_counter()
    result = run_attached_command(
        cmd,
        model=resolved_model,
        attach_url=attach_url,
        timeout_s=timeout_s,
        progress_label=SURFACE_LABELS[tool_name],
        progress_handler=progress_handler,
    )
    duration_ms = int((time.perf_counter() - start) * 1000)
    artifact_path: str | None = None

    def ensure_artifact() -> str:
        nonlocal artifact_path
        if artifact_path is None:
            artifact_path = _write_run_artifact(
                stdout=result.stdout,
                stderr=result.stderr,
                return_code=result.return_code,
                timed_out=result.timed_out,
                model=resolved_model,
                attach_url=attach_url,
                cmd=cmd,
                event_types=result.event_types,
                session_id=result.session_id,
            )
        return artifact_path

    if result.timed_out:
        artifact_path = ensure_artifact()
        return {
            "ok": False,
            "status": "timeout",
            "output": "",
            "exit_code": -1,
            "duration_ms": duration_ms,
            "error": f"{SURFACE_NAMES[tool_name]} timed out after {timeout_s}s",
            "stderr": result.stderr or None,
            "surface": SURFACE_NAMES[tool_name],
            "model": model,
            "resolved_model": resolved_model,
            "cwd": cwd,
            "attach_url": attach_url,
            "cmd": cmd[:-1],
            "artifact_path": artifact_path,
        }

    if result.return_code != 0:
        artifact_path = ensure_artifact()
        return {
            "ok": False,
            "status": "error",
            "output": result.final_output or "",
            "exit_code": result.return_code,
            "duration_ms": duration_ms,
            "error": result.error_message or result.stderr or f"Exit code {result.return_code}",
            "stderr": result.stderr or None,
            "surface": SURFACE_NAMES[tool_name],
            "model": model,
            "resolved_model": resolved_model,
            "cwd": cwd,
            "attach_url": attach_url,
            "cmd": cmd[:-1],
            "artifact_path": artifact_path,
        }

    if not (result.final_output or "").strip():
        status, error = _missing_final_error(result)
        artifact_path = ensure_artifact()
        return {
            "ok": False,
            "status": status,
            "output": "",
            "exit_code": result.return_code,
            "duration_ms": duration_ms,
            "error": error,
            "stderr": result.stderr or None,
            "surface": SURFACE_NAMES[tool_name],
            "model": model,
            "resolved_model": resolved_model,
            "cwd": cwd,
            "attach_url": attach_url,
            "cmd": cmd[:-1],
            "raw_stdout": result.stdout[:4000],
            "event_types": list(result.event_types),
            "session_id": result.session_id,
            "artifact_path": artifact_path,
        }

    return {
        "ok": True,
        "status": "ok",
        "output": result.final_output,
        "exit_code": result.return_code,
        "duration_ms": duration_ms,
        "error": None,
        "stderr": result.stderr or None,
        "surface": SURFACE_NAMES[tool_name],
        "model": model,
        "resolved_model": resolved_model,
        "cwd": cwd,
        "attach_url": attach_url,
        "cmd": cmd[:-1],
    }
