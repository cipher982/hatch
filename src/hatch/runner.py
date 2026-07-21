"""Core subprocess execution for agent CLIs."""

from __future__ import annotations

import asyncio
import json
import os
import queue
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Callable

from hatch.backends import Backend
from hatch.backends import get_config
from hatch.context import detect_context
from hatch.credentials import credential_backend_for
from hatch.credentials import hydrate_backend_kwargs
from hatch.longhouse_origin import mark_longhouse_automation_env
from hatch.longhouse_origin import maybe_write_opencode_origin_sidecar_from_line
from hatch.opencode_stream import OpenCodeStreamAccumulator
from hatch.opencode_stream import extract_opencode_log_error


@dataclass
class AgentResult:
    """Result from running an agent."""

    ok: bool
    output: str
    exit_code: int
    duration_ms: int
    error: str | None = None
    stderr: str | None = None
    artifact_path: str | None = None
    session_id: str | None = None
    resume_command: str | None = None

    @property
    def status(self) -> str:
        """Simple status string."""
        if self.ok:
            return "ok"
        if self.exit_code == -1:
            return "timeout"
        if self.exit_code == -2:
            return "not_found"
        return "error"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "ok": self.ok,
            "status": self.status,
            "output": self.output,
            "exit_code": self.exit_code,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "stderr": self.stderr,
            "artifact_path": self.artifact_path,
            "session_id": self.session_id,
            "resume_command": self.resume_command,
        }


@dataclass
class ClaudeStreamRunResult:
    """Captured result from a Claude stream-json subprocess."""

    stdout: str
    stderr: str
    return_code: int
    timed_out: bool
    final_output: str | None = None


@dataclass
class OpenCodeStreamRunResult:
    """Captured result from an OpenCode JSON event stream subprocess."""

    stdout: str
    stderr: str
    return_code: int
    timed_out: bool
    final_output: str | None = None
    error_message: str | None = None
    artifact_path: str | None = None
    session_id: str | None = None
    resume_command: str | None = None


@dataclass
class CursorStreamRunResult:
    """Captured result from a Cursor stream-json subprocess."""

    stdout: str
    stderr: str
    return_code: int
    timed_out: bool
    final_output: str | None = None
    error_message: str | None = None


class CursorStreamAccumulator:
    """Parse Cursor NDJSON into a canonical result and terse progress."""

    def __init__(self) -> None:
        self.final_output: str | None = None
        self.error_message: str | None = None
        self._seen_tool_ids: set[str] = set()

    def handle_line(self, line: str) -> list[str]:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            return []
        if not isinstance(payload, dict):
            return []

        payload_type = payload.get("type")
        if payload_type == "system" and payload.get("subtype") == "init":
            model = payload.get("model") or "unknown-model"
            session_id = payload.get("session_id")
            suffix = f", session {session_id[:8]}" if isinstance(session_id, str) else ""
            return [f"[hatch] Cursor started ({model}{suffix})"]

        if payload_type == "tool_call" and payload.get("subtype") == "started":
            call_id = payload.get("call_id")
            if isinstance(call_id, str):
                if call_id in self._seen_tool_ids:
                    return []
                self._seen_tool_ids.add(call_id)
            tool_call = payload.get("tool_call")
            return [_summarize_cursor_tool_call(tool_call if isinstance(tool_call, dict) else {})]

        if payload_type == "result":
            if payload.get("subtype") != "success" or payload.get("is_error") is True:
                self.error_message = str(payload.get("result") or "Cursor returned an error result")
                return []
            result = payload.get("result")
            if not isinstance(result, str) or not result.strip():
                self.error_message = "Cursor result event did not contain output"
                return []
            self.final_output = result
            duration_ms = payload.get("duration_ms")
            if isinstance(duration_ms, int):
                return [f"[hatch] Cursor completed in {duration_ms / 1000:.1f}s"]
            return ["[hatch] Cursor completed"]

        return []


def _summarize_cursor_tool_call(tool_call: dict[str, Any]) -> str:
    """Render a compact Cursor tool-start line without exposing raw event data."""
    if not tool_call:
        return "[hatch] Cursor tool"
    name, details = next(iter(tool_call.items()))
    args = details.get("args") if isinstance(details, dict) else None
    if isinstance(args, dict):
        for key in ("path", "filePath", "command", "query"):
            value = args.get(key)
            if value:
                return f"[hatch] Cursor {name}: {value}"
    return f"[hatch] Cursor {name}"


class _ClaudeStreamAccumulator:
    """Parse Claude stream-json lines into a final output and terse progress."""

    def __init__(self) -> None:
        self.final_output: str | None = None
        self.last_text_output: str | None = None
        self._announced_thinking = False
        self._seen_tool_ids: set[str] = set()

    def handle_line(self, line: str) -> list[str]:
        stripped = line.strip()
        if not stripped:
            return []

        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            return []

        payload_type = payload.get("type")
        messages: list[str] = []

        if payload_type == "system" and payload.get("subtype") == "init":
            model = payload.get("model") or "unknown-model"
            session_id = payload.get("session_id")
            if session_id:
                messages.append(f"[hatch] Claude started ({model}, session {session_id[:8]})")
            else:
                messages.append(f"[hatch] Claude started ({model})")
            return messages

        if payload_type == "stream_event":
            event = payload.get("event") or {}
            event_type = event.get("type")
            if event_type == "message_start":
                self._announced_thinking = False
            elif event_type == "content_block_start":
                block = event.get("content_block") or {}
                if block.get("type") == "thinking" and not self._announced_thinking:
                    self._announced_thinking = True
                    messages.append("[hatch] Claude thinking")
            return messages

        if payload_type == "assistant":
            message = payload.get("message") or {}
            for block in message.get("content") or []:
                block_type = block.get("type")
                if block_type == "tool_use":
                    tool_id = block.get("id")
                    if tool_id and tool_id in self._seen_tool_ids:
                        continue
                    if tool_id:
                        self._seen_tool_ids.add(tool_id)
                    summary = _summarize_claude_tool_use(block)
                    if summary:
                        messages.append(summary)
                elif block_type == "text":
                    text = (block.get("text") or "").strip()
                    if text:
                        self.last_text_output = text
            return messages

        if payload_type == "result":
            result_text = payload.get("result")
            if isinstance(result_text, str) and result_text.strip():
                self.final_output = result_text
            duration_ms = payload.get("duration_ms")
            if isinstance(duration_ms, int):
                messages.append(f"[hatch] Claude completed in {duration_ms / 1000:.1f}s")
            else:
                messages.append("[hatch] Claude completed")
            return messages

        return messages


def _summarize_claude_tool_use(tool_use: dict[str, Any]) -> str | None:
    """Render a compact progress line for a Claude tool call."""
    name = tool_use.get("name") or "tool"
    input_data = tool_use.get("input") or {}

    # Claude can surface Slack auth MCP tools even in headless automation.
    # They are noisy, irrelevant to hatch users, and often repeat many times.
    if name.startswith("mcp__claude_ai_Slack__"):
        return None

    if name == "Bash":
        command = input_data.get("command")
        description = input_data.get("description")
        if command and description:
            return f"[hatch] Bash: {description} ({command})"
        if command:
            return f"[hatch] Bash: {command}"

    for key in ("file_path", "path"):
        value = input_data.get(key)
        if value:
            return f"[hatch] {name}: {value}"

    description = input_data.get("description")
    if description:
        return f"[hatch] {name}: {description}"

    return f"[hatch] {name}"


def _run_subprocess(
    cmd: list[str],
    stdin_data: bytes | None,
    env: dict[str, str],
    cwd: str | None,
    timeout_s: int,
) -> tuple[str, str, int, bool]:
    """Run subprocess with proper timeout and cleanup.

    Returns: (stdout, stderr, return_code, timed_out)
    """
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE if stdin_data else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=cwd,
        text=True,
    )

    try:
        stdout, stderr = proc.communicate(
            input=stdin_data.decode() if stdin_data else None,
            timeout=timeout_s,
        )
        return stdout or "", stderr or "", proc.returncode, False
    except subprocess.TimeoutExpired:
        # Kill the process on timeout
        proc.kill()
        stdout, stderr = proc.communicate()
        return stdout or "", stderr or "", -1, True


def _stream_reader(
    stream: Any,
    source: str,
    output_queue: queue.Queue[tuple[str, str | None]],
) -> None:
    """Read a text stream line-by-line into a queue."""
    try:
        for line in iter(stream.readline, ""):
            output_queue.put((source, line))
    finally:
        output_queue.put((source, None))


def run_claude_stream_sync(
    cmd: list[str],
    stdin_data: bytes | None,
    env: dict[str, str],
    cwd: str | None,
    timeout_s: int,
    *,
    progress_handler: Callable[[str], None] | None = None,
    heartbeat_s: int = 30,
) -> ClaudeStreamRunResult:
    """Run Claude in stream-json mode and preserve only the final answer."""
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE if stdin_data else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=cwd,
        text=True,
        bufsize=1,
    )

    if stdin_data and proc.stdin:
        proc.stdin.write(stdin_data.decode("utf-8"))
        proc.stdin.close()

    if proc.stdout is None or proc.stderr is None:
        raise RuntimeError("Claude process streams were not created")

    output_queue: queue.Queue[tuple[str, str | None]] = queue.Queue()
    readers = [
        threading.Thread(target=_stream_reader, args=(proc.stdout, "stdout", output_queue), daemon=True),
        threading.Thread(target=_stream_reader, args=(proc.stderr, "stderr", output_queue), daemon=True),
    ]
    for reader in readers:
        reader.start()

    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    accumulator = _ClaudeStreamAccumulator()
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

    final_output = accumulator.final_output or accumulator.last_text_output
    return ClaudeStreamRunResult(
        stdout="".join(stdout_chunks),
        stderr="".join(stderr_chunks),
        return_code=-1 if timed_out else proc.returncode,
        timed_out=timed_out,
        final_output=final_output,
    )


def run_cursor_stream_sync(
    cmd: list[str],
    stdin_data: bytes | None,
    env: dict[str, str],
    cwd: str | None,
    timeout_s: int,
    *,
    progress_handler: Callable[[str], None] | None = None,
    heartbeat_s: int = 30,
) -> CursorStreamRunResult:
    """Run Cursor stream-json and preserve its terminal result event."""
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE if stdin_data else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=cwd,
        text=True,
        bufsize=1,
    )
    if stdin_data and proc.stdin:
        proc.stdin.write(stdin_data.decode("utf-8"))
        proc.stdin.close()
    if proc.stdout is None or proc.stderr is None:
        raise RuntimeError("Cursor process streams were not created")

    output_queue: queue.Queue[tuple[str, str | None]] = queue.Queue()
    readers = [
        threading.Thread(target=_stream_reader, args=(proc.stdout, "stdout", output_queue), daemon=True),
        threading.Thread(target=_stream_reader, args=(proc.stderr, "stderr", output_queue), daemon=True),
    ]
    for reader in readers:
        reader.start()

    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    accumulator = CursorStreamAccumulator()
    start = time.monotonic()
    last_progress_at = start
    stdout_open = True
    stderr_open = True
    timed_out = False
    while stdout_open or stderr_open:
        remaining = timeout_s - (time.monotonic() - start)
        if remaining <= 0:
            timed_out = True
            proc.kill()
            proc.wait()
            break
        try:
            source, line = output_queue.get(timeout=min(0.1, remaining))
        except queue.Empty:
            if progress_handler and proc.poll() is None and (time.monotonic() - last_progress_at) >= heartbeat_s:
                progress_handler(f"[hatch] Cursor still running ({int(time.monotonic() - start)}s)")
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
            for message in accumulator.handle_line(line):
                if progress_handler:
                    progress_handler(message)
                last_progress_at = time.monotonic()
        else:
            stderr_chunks.append(line)

    for reader in readers:
        reader.join(timeout=1)
    if not timed_out and proc.returncode is None:
        proc.wait()
    return CursorStreamRunResult(
        stdout="".join(stdout_chunks),
        stderr="".join(stderr_chunks),
        return_code=-1 if timed_out else proc.returncode,
        timed_out=timed_out,
        final_output=accumulator.final_output,
        error_message=accumulator.error_message,
    )


def run_opencode_stream_sync(
    cmd: list[str],
    stdin_data: bytes | None,
    env: dict[str, str],
    cwd: str | None,
    timeout_s: int,
    *,
    model: str | None = None,
    progress_label: str = "Agent",
    progress_handler: Callable[[str], None] | None = None,
    heartbeat_s: int = 30,
) -> OpenCodeStreamRunResult:
    """Run OpenCode with per-invocation writable state.

    OpenCode's SQLite store has no busy timeout, so sharing XDG_DATA_HOME
    between concurrent one-shot Hatch processes can fail immediately with
    ``database is locked``. Hatch passes provider credentials explicitly and
    does not resume OpenCode sessions, so its data and state are disposable.
    Config and cache remain on the reviewed shared paths supplied by callers.
    """
    runtime_path = Path(tempfile.mkdtemp(prefix="hatch-opencode-"))
    try:
        data_path = runtime_path / "data"
        state_path = runtime_path / "state"
        data_path.mkdir()
        state_path.mkdir()
        isolated_env = dict(env)
        isolated_env["XDG_DATA_HOME"] = str(data_path)
        isolated_env["XDG_STATE_HOME"] = str(state_path)
        result = _run_opencode_stream_sync(
            cmd,
            stdin_data,
            isolated_env,
            cwd,
            timeout_s,
            progress_label=progress_label,
            progress_handler=progress_handler,
            heartbeat_s=heartbeat_s,
        )
        if result.timed_out:
            artifact_root = Path(
                os.environ.get("HATCH_TIMEOUT_ARTIFACT_ROOT", "~/.local/state/hatch/timeouts")
            ).expanduser()
            artifact_root.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.chmod(artifact_root, 0o700)
            session_label = (result.session_id or "unknown-session").replace("/", "-")
            destination = artifact_root / f"{int(time.time())}-{session_label}"
            shutil.move(str(runtime_path), destination)
            (destination / "stdout.jsonl").write_text(result.stdout, encoding="utf-8")
            (destination / "stderr.log").write_text(result.stderr, encoding="utf-8")
            provider = model.split("/", 1)[0] if model and "/" in model else None
            credential_env_var = {
                "openai": "OPENAI_API_KEY",
                "openrouter": "OPENROUTER_API_KEY",
                "amazon-bedrock": "AWS_PROFILE",
            }.get(provider)
            resume_argv = None
            resume_command = None
            inspect_argv = None
            if result.session_id:
                saved_environment = [
                    f"XDG_DATA_HOME={destination / 'data'}",
                    f"XDG_STATE_HOME={destination / 'state'}",
                ]
                inspect_argv = [
                    "env",
                    *saved_environment,
                    "opencode",
                    "export",
                    result.session_id,
                ]
                resume_argv = [
                    "env",
                    *saved_environment,
                    "opencode",
                    "run",
                    "--dangerously-skip-permissions",
                ]
                if cwd:
                    resume_argv.extend(["--dir", cwd])
                resume_argv.extend([
                    "--print-logs",
                    "--log-level",
                    "ERROR",
                    "--format",
                    "json",
                ])
                if model:
                    resume_argv.extend(["-m", model])
                resume_argv.extend([
                    "--session",
                    result.session_id,
                    "Return only the concise final answer from the evidence already gathered. Do not use tools or expand the investigation.",
                ])
                resume_command = shlex.join(resume_argv)
            metadata = {
                "session_id": result.session_id,
                "cwd": cwd,
                "model": model,
                "provider": provider,
                "credential_env_var": credential_env_var,
                "timed_out_at": int(time.time()),
                "environment": {
                    "XDG_DATA_HOME": str(destination / "data"),
                    "XDG_STATE_HOME": str(destination / "state"),
                },
                "inspect_argv": inspect_argv,
                "resume_argv": resume_argv,
                "resume_command": resume_command,
            }
            (destination / "metadata.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.chmod(destination, 0o700)
            result.artifact_path = str(destination)
            result.resume_command = resume_command
        return result
    finally:
        if runtime_path.exists():
            shutil.rmtree(runtime_path)


def _run_opencode_stream_sync(
    cmd: list[str],
    stdin_data: bytes | None,
    env: dict[str, str],
    cwd: str | None,
    timeout_s: int,
    *,
    progress_label: str = "Agent",
    progress_handler: Callable[[str], None] | None = None,
    heartbeat_s: int = 30,
) -> OpenCodeStreamRunResult:
    """Run OpenCode in JSON event mode and preserve only the final answer."""
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE if stdin_data else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=cwd,
        text=True,
        bufsize=1,
    )

    if stdin_data and proc.stdin:
        proc.stdin.write(stdin_data.decode("utf-8"))
        proc.stdin.close()

    if proc.stdout is None or proc.stderr is None:
        raise RuntimeError("OpenCode process streams were not created")

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
    sidecar_session_ids: set[str] = set()

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
            maybe_write_opencode_origin_sidecar_from_line(line, env, sidecar_session_ids)
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
            maybe_write_opencode_origin_sidecar_from_line(line, env, sidecar_session_ids)
            accumulator.handle_line(line)
        else:
            stderr_chunks.append(line)

    stderr_text = "".join(stderr_chunks)
    error_message = accumulator.error_message
    # OpenCode often exits 0 with no stream error event when the provider call
    # failed (e.g. Bedrock 503/throttling). Recover the real cause from the
    # --print-logs ERROR lines on stderr so callers see why it produced nothing.
    if error_message is None and accumulator.final_output is None:
        error_message = extract_opencode_log_error(stderr_text)

    return OpenCodeStreamRunResult(
        stdout="".join(stdout_chunks),
        stderr=stderr_text,
        return_code=-1 if timed_out else proc.returncode,
        timed_out=timed_out,
        final_output=accumulator.final_output,
        error_message=error_message,
        session_id=accumulator.session_id,
    )


def _validate_timeout(timeout_s: int) -> None:
    """Validate timeout."""
    if timeout_s <= 0:
        raise ValueError("timeout_s must be > 0")


def _validate_cwd(cwd: str | Path | None) -> None:
    """Validate working directory."""
    if cwd is None:
        return
    path = Path(cwd)
    if not path.exists():
        raise ValueError(f"cwd does not exist: {cwd}")
    if not path.is_dir():
        raise ValueError(f"cwd is not a directory: {cwd}")


def run_sync(
    cmd: list[str],
    stdin_data: bytes | None,
    env: dict[str, str],
    cwd: str | None,
    timeout_s: int,
) -> tuple[str, str, int, bool]:
    """Synchronous version for CLI use."""
    return _run_subprocess(cmd, stdin_data, env, cwd, timeout_s)


async def run(
    prompt: str,
    backend: Backend,
    *,
    cwd: str | Path | None = None,
    timeout_s: int = 1800,
    **backend_kwargs: Any,
) -> AgentResult:
    """Run an agent CLI and return the result.

    Args:
        prompt: The prompt to send to the agent
        backend: Which backend to use (bedrock, codex, gemini, opencode)
        cwd: Working directory for the agent
        timeout_s: Hard timeout in seconds (default 30 minutes)
        **backend_kwargs: Backend-specific options (api_key, model, etc.)

    Returns:
        AgentResult with ok, output, duration_ms, error, etc.
    """
    start = time.monotonic()

    try:
        _validate_timeout(timeout_s)
        _validate_cwd(cwd)
    except ValueError as e:
        duration_ms = int((time.monotonic() - start) * 1000)
        return AgentResult(
            ok=False,
            output="",
            exit_code=-3,
            duration_ms=duration_ms,
            error=str(e),
        )

    ctx = detect_context()
    credential_backend = credential_backend_for(backend, backend_kwargs)
    if credential_backend is None:
        resolved_backend_kwargs = dict(backend_kwargs)
    else:
        resolved_backend_kwargs = hydrate_backend_kwargs(credential_backend, backend_kwargs)
    config = get_config(backend, prompt, ctx, **resolved_backend_kwargs)
    env = config.build_env()
    mark_longhouse_automation_env(env)

    cwd_str = str(cwd) if cwd else None

    try:
        stdout, stderr, return_code, timed_out = await asyncio.to_thread(
            _run_subprocess,
            config.cmd,
            config.stdin_data,
            env,
            cwd_str,
            timeout_s,
        )
        sidecar_session_ids: set[str] = set()
        for line in stdout.splitlines():
            maybe_write_opencode_origin_sidecar_from_line(line, env, sidecar_session_ids)

        duration_ms = int((time.monotonic() - start) * 1000)

        if timed_out:
            return AgentResult(
                ok=False,
                output=stdout,
                exit_code=-1,  # Special code for timeout
                duration_ms=duration_ms,
                error=f"Agent timed out after {timeout_s}s",
                stderr=stderr or None,
            )

        if return_code != 0:
            return AgentResult(
                ok=False,
                output=stdout,
                exit_code=return_code,
                duration_ms=duration_ms,
                error=stderr or f"Exit code {return_code}",
                stderr=stderr,
            )

        if not stdout.strip():
            return AgentResult(
                ok=False,
                output="",
                exit_code=0,
                duration_ms=duration_ms,
                error="Empty output from agent",
                stderr=stderr,
            )

        return AgentResult(
            ok=True,
            output=stdout,
            exit_code=0,
            duration_ms=duration_ms,
            stderr=stderr,
        )

    except FileNotFoundError as e:
        duration_ms = int((time.monotonic() - start) * 1000)
        return AgentResult(
            ok=False,
            output="",
            exit_code=-2,  # Special code for not found
            duration_ms=duration_ms,
            error=f"CLI not found: {e}",
        )

    except Exception as e:
        duration_ms = int((time.monotonic() - start) * 1000)
        return AgentResult(
            ok=False,
            output="",
            exit_code=-3,
            duration_ms=duration_ms,
            error=str(e),
        )
