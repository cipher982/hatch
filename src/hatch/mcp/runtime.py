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
import urllib.error
import urllib.request
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
        env=_build_server_env(),
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

    return OpenCodeRunResult(
        stdout="".join(stdout_chunks),
        stderr="".join(stderr_chunks),
        return_code=-1 if timed_out else proc.returncode,
        timed_out=timed_out,
        final_output=accumulator.final_output,
        error_message=accumulator.error_message,
        attach_url=attach_url,
        model=model,
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
        }

    if not (result.final_output or "").strip():
        return {
            "ok": False,
            "status": "transport_error",
            "output": "",
            "exit_code": result.return_code,
            "duration_ms": duration_ms,
            "error": result.error_message or "OpenCode returned no final output",
            "stderr": result.stderr or None,
            "surface": SURFACE_NAMES[tool_name],
            "model": model,
            "resolved_model": resolved_model,
            "cwd": cwd,
            "attach_url": attach_url,
            "cmd": cmd[:-1],
            "raw_stdout": result.stdout[:4000],
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
