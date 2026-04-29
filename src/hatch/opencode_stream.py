"""OpenCode JSONL stream parsing shared by CLI and MCP runtimes."""

from __future__ import annotations

import json
import shlex
from typing import Any


class OpenCodeStreamAccumulator:
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
            messages.append(summarize_opencode_tool_use(part))
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


def summarize_opencode_tool_use(part: dict[str, Any]) -> str:
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
