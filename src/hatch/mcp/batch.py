"""Shared batch support for hatch MCP tools."""

from __future__ import annotations

import asyncio
import sys
import time
import uuid
from typing import Any


def add_batch_support(mcp_server: Any, tools: dict[str, Any], max_concurrent: int = 10):
    """Add a batch tool to a FastMCP server for parallel execution."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _execute_one(batch_id: str, index: int, total: int, tool_name: str, args: dict) -> dict:
        if tool_name not in tools:
            return {
                "ok": False,
                "index": index,
                "tool": tool_name,
                "error": {"type": "unknown_tool", "message": f"Unknown tool: {tool_name}"},
            }

        async with semaphore:
            started_at = time.perf_counter()
            print(
                f"[batch:{batch_id}] START {index}/{total} tool={tool_name}",
                file=sys.stderr,
                flush=True,
            )
            try:
                tool_func = tools[tool_name]
                if hasattr(tool_func, "fn"):
                    tool_func = tool_func.fn
                result = await tool_func(**args)
                duration_sec = time.perf_counter() - started_at
                print(
                    f"[batch:{batch_id}] FINISH {index}/{total} tool={tool_name} "
                    f"duration={duration_sec:.2f}s",
                    file=sys.stderr,
                    flush=True,
                )
                return {
                    "ok": True,
                    "index": index,
                    "tool": tool_name,
                    "duration_sec": duration_sec,
                    "value": result,
                }
            except Exception as exc:  # pragma: no cover - thin wrapper
                duration_sec = time.perf_counter() - started_at
                print(
                    f"[batch:{batch_id}] ERROR {index}/{total} tool={tool_name} "
                    f"duration={duration_sec:.2f}s error={type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                return {
                    "ok": False,
                    "index": index,
                    "tool": tool_name,
                    "duration_sec": duration_sec,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }

    @mcp_server.tool()
    async def batch(calls: list[dict]) -> dict:
        """Execute multiple tool calls in parallel."""
        if not calls:
            return {"results": [], "succeeded": 0, "failed": 0}

        batch_id = uuid.uuid4().hex[:6]
        for i, call in enumerate(calls):
            if not isinstance(call, dict):
                return {
                    "results": [],
                    "succeeded": 0,
                    "failed": len(calls),
                    "error": f"Call {i} is not a dict",
                }
            if "tool" not in call:
                return {
                    "results": [],
                    "succeeded": 0,
                    "failed": len(calls),
                    "error": f"Call {i} missing 'tool' field",
                }

        batch_started_at = time.perf_counter()
        print(
            f"[batch:{batch_id}] START total_calls={len(calls)}",
            file=sys.stderr,
            flush=True,
        )
        tasks = [
            _execute_one(batch_id, i + 1, len(calls), call["tool"], call.get("args", {}))
            for i, call in enumerate(calls)
        ]
        results = await asyncio.gather(*tasks)

        succeeded = sum(1 for result in results if result["ok"])
        failed = len(results) - succeeded
        total_duration_sec = time.perf_counter() - batch_started_at
        print(
            f"[batch:{batch_id}] FINISH total_calls={len(calls)} succeeded={succeeded} "
            f"failed={failed} duration={total_duration_sec:.2f}s",
            file=sys.stderr,
            flush=True,
        )

        return {
            "results": list(results),
            "succeeded": succeeded,
            "failed": failed,
            "batch_id": batch_id,
            "duration_sec": total_duration_sec,
        }

    return batch

