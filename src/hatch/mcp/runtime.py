"""Persistent OpenCode-backed runtime for hatch MCP."""

from __future__ import annotations

import atexit
import contextlib
import json
import os
import queue
import shutil
import signal
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

from hatch.aws_preflight import DEFAULT_BEDROCK_AWS_PROFILE
from hatch.aws_preflight import DEFAULT_BEDROCK_AWS_REGION
from hatch.aws_preflight import BedrockAwsAuthError
from hatch.aws_preflight import preflight_bedrock_aws
from hatch.backends import Backend
from hatch.credentials import SECRET_SPECS
from hatch.credentials import OPENROUTER_CREDENTIAL
from hatch.credentials import credential_status
from hatch.credentials import ensure_opencode_credentials
from hatch.credentials import resolve_env_secret
from hatch.credentials import _load_secret_from_helper
from hatch.models import SURFACE_LABELS
from hatch.models import SURFACE_NAMES
from hatch.models import TOOL_TO_PROVIDER
from hatch.models import resolve_tool_model
from hatch.models import tool_model_choices
from hatch.observatory import apply_observatory_trust_env
from hatch.observatory import observatory_ca_path
from hatch.observatory import observatory_trust_signature
from hatch.opencode_stream import OpenCodeStreamAccumulator
from hatch.opencode_stream import extract_opencode_log_error


OPENCODE_BIN = os.environ.get("HATCH_MCP_OPENCODE_BIN", "opencode")
ATTACH_URL_ENV = "HATCH_MCP_OPENCODE_ATTACH_URL"
OPENCODE_PORT_ENV = "HATCH_MCP_OPENCODE_PORT"
OPENCODE_ROOT_ENV = "HATCH_MCP_OPENCODE_ROOT"
OPENCODE_CONFIG_ENV = "HATCH_MCP_OPENCODE_CONFIG"
OPENCODE_IDLE_SHUTDOWN_ENV = "HATCH_MCP_OPENCODE_IDLE_SHUTDOWN_S"
SESSION_FETCH_TIMEOUT_ENV = "HATCH_MCP_SESSION_FETCH_TIMEOUT_S"
DEFAULT_GEMINI_MODEL = os.environ.get(
    "HATCH_MCP_GEMINI_MODEL",
    "google/gemini-3-flash-preview",
)
SERVER_START_TIMEOUT_S = 15.0
SERVER_IDLE_SHUTDOWN_S = 60.0
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
    assistant_message_id: str | None = None
    selected_message_id: str | None = None
    selected_finish_reason: str | None = None
    output_source: str = "stdout"
    session_messages: list[dict[str, Any]] | None = None
    session_fetch_error: str | None = None
    artifact_path: str | None = None


@dataclass
class OpenCodeSessionSnapshot:
    output: str
    completed: bool
    finish_reason: str | None = None
    message_id: str | None = None


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

    for key in [
        "HATCH_CREDENTIAL_ROULETTE",
        "HATCH_ROULETTE_OPENAI_API_KEY",
        "HATCH_ROULETTE_AWS_PROFILE",
        "HATCH_ROULETTE_AWS_REGION",
        "OPENAI_API_KEY",
        "CODEX_API_KEY",
        "AWS_PROFILE",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "CLAUDE_CODE_USE_BEDROCK",
        "ANTHROPIC_MODEL",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
    ]:
        env.pop(key, None)
    for key in list(env):
        if key.startswith("OPENCODE"):
            env.pop(key, None)

    secret = _maybe_load_secret(Backend.CODEX)
    if secret:
        env["OPENAI_API_KEY"] = secret

    openrouter_secret = resolve_env_secret(OPENROUTER_CREDENTIAL, env)
    if openrouter_secret:
        env["OPENROUTER_API_KEY"] = openrouter_secret
    else:
        env.pop("OPENROUTER_API_KEY", None)

    env["AWS_PROFILE"] = DEFAULT_BEDROCK_AWS_PROFILE
    env["AWS_REGION"] = DEFAULT_BEDROCK_AWS_REGION
    for key, value in runtime_paths.items():
        env[key] = str(value)
    env["OPENCODE_CONFIG"] = str(_runtime_config_path())
    env.pop("OPENCODE_CONFIG_CONTENT", None)
    apply_observatory_trust_env(env)
    return env


def _build_run_env(model: str) -> dict[str, str]:
    env = _build_server_env()

    if not model.startswith("openai/"):
        env.pop("OPENAI_API_KEY", None)
    if not model.startswith("openrouter/"):
        env.pop("OPENROUTER_API_KEY", None)

    return env


def _opencode_event_summary(stdout: str) -> tuple[tuple[str, ...], str | None, str | None]:
    event_types: list[str] = []
    session_id: str | None = None
    assistant_message_id: str | None = None

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
        part = payload.get("part") or {}
        if (
            assistant_message_id is None
            and payload_type == "step_start"
            and isinstance(part, dict)
            and isinstance(part.get("messageID"), str)
        ):
            assistant_message_id = part["messageID"]

    return tuple(event_types), session_id, assistant_message_id


def _session_fetch_timeout_s() -> float:
    configured = os.environ.get(SESSION_FETCH_TIMEOUT_ENV, "").strip()
    if not configured:
        return 5.0
    try:
        timeout = float(configured)
    except ValueError:
        return 5.0
    return max(0.1, timeout)


def _idle_shutdown_s() -> float:
    configured = os.environ.get(OPENCODE_IDLE_SHUTDOWN_ENV, "").strip()
    if not configured:
        return SERVER_IDLE_SHUTDOWN_S
    try:
        return float(configured)
    except ValueError:
        return SERVER_IDLE_SHUTDOWN_S


def _terminate_process(proc: subprocess.Popen[str] | None, timeout_s: float = 2.0) -> None:
    if proc is None:
        return
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout_s)
        return
    except subprocess.TimeoutExpired:
        proc.kill()
    except Exception:
        with contextlib.suppress(Exception):
            proc.kill()
    with contextlib.suppress(Exception):
        proc.wait(timeout=timeout_s)


def _fetch_attached_session_messages(
    attach_url: str,
    session_id: str,
) -> tuple[list[dict[str, Any]] | None, str | None]:
    """Fetch the full OpenCode session message payload for an attached run."""
    encoded_session_id = urllib.parse.quote(session_id, safe="")
    url = f"{attach_url.rstrip('/')}/session/{encoded_session_id}/message"

    try:
        with urllib.request.urlopen(url, timeout=_session_fetch_timeout_s()) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        return None, str(exc)

    if not isinstance(payload, list):
        return None, f"Unexpected OpenCode session payload type: {type(payload).__name__}"

    messages = [message for message in payload if isinstance(message, dict)]
    return messages, None


def _extract_message_output(message: dict[str, Any]) -> OpenCodeSessionSnapshot | None:
    info = message.get("info") or {}
    if not isinstance(info, dict) or info.get("role") != "assistant":
        return None
    message_id = info.get("id") if isinstance(info.get("id"), str) else None

    chunks: list[str] = []
    time_info = info.get("time")
    completed = bool(time_info.get("completed")) if isinstance(time_info, dict) else False
    finish_reason: str | None = None
    parts = message.get("parts") or []
    if not isinstance(parts, list):
        return None

    for part in parts:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text" and isinstance(part.get("text"), str):
            chunks.append(part["text"])
        if part.get("type") == "step-finish" and part.get("reason") == "stop":
            finish_reason = part.get("reason")
            completed = True
        elif part.get("type") == "step-finish" and isinstance(part.get("reason"), str):
            finish_reason = part.get("reason")

    text = "".join(chunks).strip()
    if not text:
        return None
    return OpenCodeSessionSnapshot(
        output=text,
        completed=completed,
        finish_reason=finish_reason,
        message_id=message_id,
    )


def _extract_attached_session_output(
    messages: list[dict[str, Any]],
    assistant_message_id: str | None = None,
) -> OpenCodeSessionSnapshot | None:
    """Return final assistant text for this run.

    OpenCode can emit several assistant messages during one run: short pre-tool
    narration, more tool-use narration, then the final answer. The JSON stream's
    first step_start message id is useful for correlation, but it is not
    necessarily the final assistant message.
    """
    snapshots: list[OpenCodeSessionSnapshot] = []
    for message in messages:
        info = message.get("info") or {}
        if not isinstance(info, dict) or info.get("role") != "assistant":
            continue
        snapshot = _extract_message_output(message)
        if snapshot:
            snapshots.append(snapshot)

    for snapshot in reversed(snapshots):
        if snapshot.finish_reason == "stop":
            return snapshot

    if assistant_message_id:
        for snapshot in snapshots:
            if snapshot.message_id == assistant_message_id:
                return snapshot

    for snapshot in reversed(snapshots):
        return snapshot

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
    assistant_message_id: str | None,
    selected_message_id: str | None,
    selected_finish_reason: str | None,
    output_source: str,
    session_messages: list[dict[str, Any]] | None,
    session_fetch_error: str | None,
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
        "assistant_message_id": assistant_message_id,
        "selected_message_id": selected_message_id,
        "selected_finish_reason": selected_finish_reason,
        "output_source": output_source,
        "session_messages": session_messages,
        "session_fetch_error": session_fetch_error,
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


def _healthcheck_status(url: str, timeout_s: float = 1.0) -> tuple[bool, str | None]:
    try:
        with urllib.request.urlopen(f"{url}/global/health", timeout=timeout_s) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code} {exc.reason}"
    except (OSError, urllib.error.URLError) as exc:
        return False, str(exc)
    except json.JSONDecodeError as exc:
        return False, f"invalid health JSON: {exc}"
    healthy = bool(payload.get("healthy"))
    return healthy, None if healthy else f"unhealthy payload: {payload!r}"


def _healthcheck(url: str, timeout_s: float = 1.0) -> bool:
    healthy, _ = _healthcheck_status(url, timeout_s=timeout_s)
    return healthy


class OpenCodeServerManager:
    """Own a single persistent local OpenCode server per hatch MCP process."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[str] | None = None
        self._url: str | None = None
        self._managed = False
        self._trust_signature: tuple[tuple[str, str], ...] = ()
        self._active_runs = 0
        self._idle_timer: threading.Timer | None = None
        atexit.register(self.shutdown)

    @property
    def current_url(self) -> str | None:
        return self._url

    def ensure_server(self) -> str:
        configured_url = os.environ.get(ATTACH_URL_ENV, "").strip()
        if configured_url:
            healthy, health_error = _healthcheck_status(configured_url)
            if not healthy:
                raise HatchMcpRuntimeError(
                    f"Configured attach url is not healthy: {configured_url}"
                    f" ({health_error or 'unknown healthcheck failure'})"
                )
            with self._lock:
                proc = self._take_shutdown_proc_locked()
                self._url = configured_url
                self._managed = False
                self._trust_signature = ()
            _terminate_process(proc)
            return configured_url

        with self._lock:
            self._cancel_idle_timer_locked()
            desired_signature = observatory_trust_signature(_build_server_env())
            if self._proc and self._proc.poll() is None and self._url and _healthcheck(self._url):
                if self._trust_signature == desired_signature:
                    return self._url
                _terminate_process(self._take_shutdown_proc_locked())

            self._start_locked()
            assert self._url is not None
            return self._url

    def acquire_server(self) -> str:
        with self._lock:
            self._active_runs += 1
            self._cancel_idle_timer_locked()
        try:
            return self.ensure_server()
        except Exception:
            self.release_server()
            raise

    def release_server(self) -> None:
        should_shutdown_now = False
        with self._lock:
            if self._active_runs > 0:
                self._active_runs -= 1
            if self._active_runs != 0:
                return
            if self._proc is None or self._proc.poll() is not None or not self._managed:
                return

            delay_s = _idle_shutdown_s()
            if delay_s < 0:
                return
            self._cancel_idle_timer_locked()
            if delay_s == 0:
                should_shutdown_now = True
            else:
                self._idle_timer = threading.Timer(delay_s, self._shutdown_if_idle)
                self._idle_timer.daemon = True
                self._idle_timer.start()

        if should_shutdown_now:
            self.shutdown()

    def _shutdown_if_idle(self) -> None:
        with self._lock:
            self._idle_timer = None
            if self._active_runs != 0:
                return
        self.shutdown()

    def _cancel_idle_timer_locked(self) -> None:
        timer = self._idle_timer
        self._idle_timer = None
        if timer is not None:
            timer.cancel()

    def _take_shutdown_proc_locked(self) -> subprocess.Popen[str] | None:
        self._cancel_idle_timer_locked()
        proc = self._proc
        should_terminate = proc is not None and proc.poll() is None and self._managed
        self._proc = None
        self._url = None
        self._managed = False
        self._trust_signature = ()
        self._active_runs = 0
        return proc if should_terminate else None

    def _start_locked(self) -> None:
        port = int(os.environ.get(OPENCODE_PORT_ENV) or _find_free_port())
        url = f"http://127.0.0.1:{port}"
        _ensure_runtime_paths()
        env = _build_server_env()

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
            env=env,
            text=True,
        )

        deadline = time.monotonic() + SERVER_START_TIMEOUT_S
        last_health_error: str | None = None
        while time.monotonic() < deadline:
            healthy, health_error = _healthcheck_status(url)
            if healthy:
                self._proc = proc
                self._url = url
                self._managed = True
                self._trust_signature = observatory_trust_signature(env)
                return
            last_health_error = health_error
            if proc.poll() is not None:
                raise HatchMcpRuntimeError(f"OpenCode server exited during startup ({proc.returncode})")
            time.sleep(0.2)

        _terminate_process(proc)
        raise HatchMcpRuntimeError(
            "Timed out waiting for OpenCode server to become healthy"
            f" at {url}; last healthcheck: {last_health_error or 'unknown failure'}"
        )

    def shutdown(self) -> None:
        with self._lock:
            proc = self._take_shutdown_proc_locked()

        _terminate_process(proc)


SERVER_MANAGER = OpenCodeServerManager()
_SIGNAL_HANDLERS_INSTALLED = False


def install_shutdown_signal_handlers() -> None:
    """Ensure managed OpenCode servers are torn down on normal MCP termination."""
    global _SIGNAL_HANDLERS_INSTALLED
    if _SIGNAL_HANDLERS_INSTALLED:
        return

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handler = signal.getsignal(signum)

        def _handler(
            received: int,
            frame: Any,
            previous_handler: Any = previous_handler,
        ) -> None:
            SERVER_MANAGER.shutdown()
            if callable(previous_handler):
                previous_handler(received, frame)
                return
            if previous_handler == signal.SIG_IGN:
                return
            raise SystemExit(128 + received)

        signal.signal(signum, _handler)

    _SIGNAL_HANDLERS_INSTALLED = True


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
        "--dangerously-skip-permissions",
        # Surface provider/transport errors on stderr (see configure_opencode).
        "--print-logs",
        "--log-level",
        "ERROR",
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
    event_types, session_id, assistant_message_id = _opencode_event_summary(stdout)
    final_output = accumulator.final_output
    completed = accumulator.completed
    output_source = "stdout"
    session_messages: list[dict[str, Any]] | None = None
    session_fetch_error: str | None = None
    selected_message_id: str | None = None
    selected_finish_reason: str | None = None

    if session_id:
        session_messages, session_fetch_error = _fetch_attached_session_messages(attach_url, session_id)
        if session_messages:
            snapshot = _extract_attached_session_output(session_messages, assistant_message_id)
            if snapshot:
                final_output = snapshot.output
                completed = snapshot.completed
                selected_message_id = snapshot.message_id
                selected_finish_reason = snapshot.finish_reason
                output_source = "session"

    error_message = accumulator.error_message
    # Recover provider failures (Bedrock 503/throttling, etc.) from the
    # --print-logs ERROR lines when OpenCode exits without a final answer.
    if error_message is None and final_output is None:
        error_message = extract_opencode_log_error(stderr)

    return OpenCodeRunResult(
        stdout=stdout,
        stderr=stderr,
        return_code=-1 if timed_out else proc.returncode,
        timed_out=timed_out,
        final_output=final_output,
        error_message=error_message,
        attach_url=attach_url,
        model=model,
        completed=completed,
        event_types=event_types,
        session_id=session_id,
        assistant_message_id=assistant_message_id,
        selected_message_id=selected_message_id,
        selected_finish_reason=selected_finish_reason,
        output_source=output_source,
        session_messages=session_messages,
        session_fetch_error=session_fetch_error,
    )


def doctor() -> dict[str, Any]:
    opencode_path = shutil.which(OPENCODE_BIN)
    attach_url = os.environ.get(ATTACH_URL_ENV, "").strip() or SERVER_MANAGER.current_url
    env = _build_server_env()
    return {
        "ok": bool(opencode_path),
        "opencode_path": opencode_path,
        "attach_url": attach_url,
        "attach_url_healthy": bool(attach_url and _healthcheck(attach_url)),
        "gemini_model": DEFAULT_GEMINI_MODEL,
        "opencode_root": str(_runtime_root()),
        "opencode_config": str(_runtime_config_path()),
        "run_artifacts_dir": str(_run_artifacts_dir()),
        "observatory_ca_path": str(observatory_ca_path(env) or ""),
        "observatory_trust_env": dict(observatory_trust_signature(env)),
        "credentials": credential_status(),
        "secret_helper_disabled": os.environ.get("HATCH_DISABLE_SECRET_HELPER", "").strip() == "1",
        "infisical_access_token_present": (
            Path.home() / ".config" / "infisical" / "access-token"
        ).is_file(),
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
    start = time.perf_counter()
    resolved_model = _resolve_model(tool_name, model)
    try:
        run_env = _build_run_env(resolved_model)
        preflight_bedrock_aws(resolved_model, run_env)
        ensure_opencode_credentials(resolved_model, run_env)
    except (BedrockAwsAuthError, ValueError) as exc:
        duration_ms = int((time.perf_counter() - start) * 1000)
        return {
            "ok": False,
            "status": "config_error",
            "output": "",
            "exit_code": 4,
            "duration_ms": duration_ms,
            "error": str(exc),
            "stderr": None,
            "surface": SURFACE_NAMES[tool_name],
            "model": model,
            "resolved_model": resolved_model,
            "cwd": cwd,
            "attach_url": None,
            "cmd": None,
        }

    attach_url = SERVER_MANAGER.acquire_server()
    try:
        cmd = build_run_command(
            tool_name=tool_name,
            prompt=prompt,
            attach_url=attach_url,
            model=model,
            cwd=cwd,
            reasoning_effort=reasoning_effort,
        )

        result = run_attached_command(
            cmd,
            model=resolved_model,
            attach_url=attach_url,
            timeout_s=timeout_s,
            progress_label=SURFACE_LABELS[tool_name],
            progress_handler=progress_handler,
        )
    finally:
        SERVER_MANAGER.release_server()
    duration_ms = int((time.perf_counter() - start) * 1000)
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
        assistant_message_id=result.assistant_message_id,
        selected_message_id=result.selected_message_id,
        selected_finish_reason=result.selected_finish_reason,
        output_source=result.output_source,
        session_messages=result.session_messages,
        session_fetch_error=result.session_fetch_error,
    )
    artifact_note = (
        "Full OpenCode run/session output is persisted at artifact_path; inspect it "
        "if output looks incomplete, confusing, or the run needs forensic recovery."
    )
    run_metadata = {
        "output_source": result.output_source,
        "session_id": result.session_id,
        "assistant_message_id": result.assistant_message_id,
        "selected_message_id": result.selected_message_id,
        "selected_finish_reason": result.selected_finish_reason,
        "session_fetch_error": result.session_fetch_error,
        "event_types": list(result.event_types),
        "artifact_path": artifact_path,
        "artifact_note": artifact_note,
    }

    if result.timed_out:
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
            **run_metadata,
        }

    if result.return_code != 0:
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
            **run_metadata,
        }

    if not (result.final_output or "").strip():
        status, error = _missing_final_error(result)
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
            **run_metadata,
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
        **run_metadata,
    }
