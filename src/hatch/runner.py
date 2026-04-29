"""Core subprocess execution for agent CLIs."""

from __future__ import annotations

import asyncio
import json
import queue
import shlex
import subprocess
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


@dataclass
class AgentResult:
    """Result from running an agent."""

    ok: bool
    output: str
    exit_code: int
    duration_ms: int
    error: str | None = None
    stderr: str | None = None

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


class _OpenCodeStreamAccumulator:
    """Parse OpenCode JSONL events into final output and terse progress."""

    def __init__(self, *, progress_label: str = "Agent") -> None:
        self.progress_label = progress_label
        self._announced_start = False
        self._seen_call_ids: set[str] = set()
        self._text_chunks: list[str] = []
        self._final_answer_chunks: list[str] = []
        self.completed = False
        self.error_message: str | None = None

    @property
    def final_output(self) -> str | None:
        if self._final_answer_chunks:
            text = "".join(self._final_answer_chunks).strip()
        else:
            text = "".join(self._text_chunks).strip()
        return text or None

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

        if payload_type == "step_start" and not self._announced_start:
            session_id = payload.get("sessionID")
            if session_id:
                messages.append(f"[hatch] {self.progress_label} started (session {session_id[:8]})")
            else:
                messages.append(f"[hatch] {self.progress_label} started")
            self._announced_start = True
            return messages

        if payload_type == "tool_use":
            part = payload.get("part") or {}
            call_id = part.get("callID")
            if call_id and call_id in self._seen_call_ids:
                return []
            if call_id:
                self._seen_call_ids.add(call_id)
            messages.append(_summarize_opencode_tool_use(part))
            return messages

        if payload_type == "text":
            part = payload.get("part") or {}
            text = part.get("text") or ""
            if text:
                self._text_chunks.append(text)
                metadata = part.get("metadata") or {}
                openai_meta = metadata.get("openai") or {}
                if openai_meta.get("phase") == "final_answer":
                    self._final_answer_chunks.append(text)
            return messages

        if payload_type == "step_finish":
            part = payload.get("part") or {}
            if part.get("reason") == "stop":
                self.completed = True
                messages.append(f"[hatch] {self.progress_label} completed")
            return messages

        if payload_type == "error":
            error = payload.get("error") or {}
            data = error.get("data") or {}
            message = data.get("message") or error.get("message") or error.get("name")
            if message:
                self.error_message = str(message)
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


def _summarize_opencode_tool_use(part: dict[str, Any]) -> str:
    """Render a compact progress line for an OpenCode tool call."""
    name = part.get("tool") or "tool"
    state = part.get("state") or {}
    input_data = state.get("input") or {}
    title = part.get("title")
    time_range = part.get("time") or {}
    elapsed_ms = time_range.get("end") - time_range.get("start") if time_range.get("end") else None
    suffix = f" ({int(elapsed_ms / 1000)}s)" if isinstance(elapsed_ms, int | float) and elapsed_ms >= 1000 else ""

    if title:
        return f"[hatch] {name}: {title}{suffix}"

    if name == "bash":
        command = input_data.get("command")
        description = input_data.get("description")
        if command and description:
            return f"[hatch] bash: {description} ({_compact_shell(command)}){suffix}"
        if command:
            return f"[hatch] bash: {_compact_shell(command)}{suffix}"

    for key in ("filePath", "path", "pattern", "query"):
        value = input_data.get(key)
        if value:
            return f"[hatch] {name}: {_compact_text(str(value))}{suffix}"

    description = input_data.get("description")
    if description:
        return f"[hatch] {name}: {_compact_text(str(description))}{suffix}"

    return f"[hatch] {name}{suffix}"


def _compact_shell(command: str, max_len: int = 180) -> str:
    """Keep shell progress readable without losing the command shape."""
    try:
        command = " ".join(shlex.split(command))
    except ValueError:
        command = " ".join(command.split())
    return _compact_text(command, max_len=max_len)


def _compact_text(text: str, max_len: int = 180) -> str:
    text = " ".join(text.split())
    if len(text) <= max_len:
        return text
    return f"{text[: max_len - 1]}..."


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
        stdin=subprocess.PIPE if stdin_data else None,
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
        stdin=subprocess.PIPE if stdin_data else None,
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


def run_opencode_stream_sync(
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
        stdin=subprocess.PIPE if stdin_data else None,
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
    accumulator = _OpenCodeStreamAccumulator(progress_label=progress_label)
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

    return OpenCodeStreamRunResult(
        stdout="".join(stdout_chunks),
        stderr="".join(stderr_chunks),
        return_code=-1 if timed_out else proc.returncode,
        timed_out=timed_out,
        final_output=accumulator.final_output,
        error_message=accumulator.error_message,
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
    timeout_s: int = 900,
    **backend_kwargs: Any,
) -> AgentResult:
    """Run an agent CLI and return the result.

    Args:
        prompt: The prompt to send to the agent
        backend: Which backend to use (bedrock, codex, gemini, opencode)
        cwd: Working directory for the agent
        timeout_s: Timeout in seconds (default 15 minutes)
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

        duration_ms = int((time.monotonic() - start) * 1000)

        if timed_out:
            return AgentResult(
                ok=False,
                output="",
                exit_code=-1,  # Special code for timeout
                duration_ms=duration_ms,
                error=f"Agent timed out after {timeout_s}s",
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
