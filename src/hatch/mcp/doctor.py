"""Doctor helpers for hatch MCP."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class McpCheck:
    ok: bool
    tools: list[str] = field(default_factory=list)
    error: str | None = None


async def _start_server():
    return await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "hatch",
        "mcp",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(Path.cwd()),
    )


async def _send(proc, msg: dict) -> None:
    assert proc.stdin is not None
    proc.stdin.write((json.dumps(msg) + "\n").encode())
    await proc.stdin.drain()


async def _recv(proc, timeout_s: int) -> dict:
    assert proc.stdout is not None
    line = await asyncio.wait_for(proc.stdout.readline(), timeout=timeout_s)
    return json.loads(line)


async def _recv_response(proc, response_id: int, timeout_s: int) -> dict:
    """Read until the requested JSON-RPC response, skipping notifications."""
    deadline = time.monotonic() + timeout_s
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise asyncio.TimeoutError
        msg = await _recv(proc, int(max(1, remaining)))
        if msg.get("id") == response_id:
            return msg


async def _initialize(proc, timeout_s: int) -> dict:
    await _send(
        proc,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "hatch-mcp-doctor", "version": "0"},
            },
        },
    )
    return await _recv_response(proc, 1, timeout_s)


async def _stop(proc) -> None:
    if proc.stdin:
        proc.stdin.close()
    try:
        await asyncio.wait_for(proc.wait(), timeout=2)
    except Exception:
        proc.kill()


async def check_mcp_server_tools(timeout_s: int = 5) -> McpCheck:
    """Start hatch MCP and verify initialize + tools/list."""
    proc = await _start_server()

    try:
        init_resp = await _initialize(proc, timeout_s)
        if "result" not in init_resp:
            return McpCheck(ok=False, tools=[], error=f"initialize failed: {init_resp}")

        await _send(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        tools_resp = await _recv_response(proc, 2, timeout_s)
        if "result" not in tools_resp:
            return McpCheck(ok=False, tools=[], error=f"tools/list failed: {tools_resp}")

        tools = [tool["name"] for tool in tools_resp["result"]["tools"]]
        required = {
            "hatch_claude",
            "hatch_codex",
            "hatch_gemini",
            "hatch_openrouter",
            "hatch_doctor",
        }
        missing = sorted(required - set(tools))
        if missing:
            return McpCheck(ok=False, tools=tools, error=f"missing tools: {missing}")

        return McpCheck(ok=True, tools=sorted(tools))
    except asyncio.TimeoutError:
        return McpCheck(ok=False, tools=[], error="timeout waiting for MCP response")
    except Exception as exc:  # pragma: no cover - thin wrapper
        return McpCheck(ok=False, tools=[], error=str(exc))
    finally:
        await _stop(proc)


async def call_tool(tool_name: str, arguments: dict, timeout_s: int = 180) -> dict:
    """Call one hatch MCP tool over stdio MCP and return structured content."""
    proc = await _start_server()
    effective_timeout = max(timeout_s, int(arguments.get("timeout_s", 0)) + 60)
    try:
        init_resp = await _initialize(proc, min(effective_timeout, 15))
        if "result" not in init_resp:
            return {"ok": False, "error": f"initialize failed: {init_resp}"}

        await _send(
            proc,
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
            },
        )
        resp = await _recv_response(proc, 2, effective_timeout)
        result = resp.get("result") or {}
        if result.get("isError"):
            content = result.get("content") or []
            if content and isinstance(content[0], dict) and isinstance(content[0].get("text"), str):
                return {"ok": False, "error": content[0]["text"]}
            return {"ok": False, "error": f"Tool call failed: {resp}"}
        structured = result.get("structuredContent")
        if isinstance(structured, dict):
            return structured

        content = result.get("content") or []
        if content and isinstance(content[0], dict) and isinstance(content[0].get("text"), str):
            try:
                return json.loads(content[0]["text"])
            except json.JSONDecodeError:
                return {"ok": False, "error": content[0]["text"]}

        return {"ok": False, "error": f"Unexpected tools/call response: {resp}"}
    except asyncio.TimeoutError:
        return {"ok": False, "error": f"timeout waiting for tools/call response from {tool_name}"}
    finally:
        await _stop(proc)


async def run_smoke(cwd: str, timeout_s: int = 120) -> dict[str, dict]:
    """Run a small real MCP smoke against the common hatch tools."""
    calls = {
        "codex": ("hatch_codex", {"model": "mini", "prompt": "Reply with just MCP_CODEX_OK", "cwd": cwd, "timeout_s": timeout_s}),
        "claude": ("hatch_claude", {"model": "sonnet", "prompt": "Reply with just MCP_CLAUDE_OK", "cwd": cwd, "timeout_s": timeout_s}),
    }
    results: dict[str, dict] = {}
    for key, (tool_name, arguments) in calls.items():
        results[key] = await call_tool(tool_name, arguments, timeout_s=timeout_s + 60)
    return results


def main(argv: Sequence[str] | None = None) -> int:
    """Minimal CLI for probing hatch MCP over stdio."""
    parser = argparse.ArgumentParser(prog="hatch mcp doctor")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("tools")

    call_parser = sub.add_parser("call")
    call_parser.add_argument("tool")
    call_parser.add_argument(
        "--args",
        default="{}",
        help='JSON object of tool arguments, e.g. \'{"prompt":"hi"}\'',
    )
    call_parser.add_argument("--timeout", type=int, default=180)

    smoke_parser = sub.add_parser("smoke")
    smoke_parser.add_argument("--cwd", required=True)
    smoke_parser.add_argument("--timeout", type=int, default=120)

    args = parser.parse_args(list(argv or ()))

    if args.cmd == "tools":
        result = asyncio.run(check_mcp_server_tools())
        print(json.dumps(result.__dict__, indent=2))
        return 0 if result.ok else 1

    if args.cmd == "call":
        payload = json.loads(args.args)
        result = asyncio.run(call_tool(args.tool, payload, timeout_s=args.timeout))
        print(json.dumps(result, indent=2))
        return 0 if result.get("ok", False) else 1

    if args.cmd == "smoke":
        result = asyncio.run(run_smoke(args.cwd, timeout_s=args.timeout))
        print(json.dumps(result, indent=2))
        return 0 if all(item.get("ok", False) for item in result.values()) else 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
