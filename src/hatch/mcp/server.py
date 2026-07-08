"""MCP server for hatch."""

from __future__ import annotations

import asyncio
import contextlib
from typing import Annotated
from typing import Literal

from fastmcp import Context
from fastmcp import FastMCP

from hatch.expert import DEFAULT_EXPERT_MODEL
from hatch.expert import ExpertReasoningEffort
from hatch.expert import run_expert_sync
from hatch.mcp.batch import add_batch_support
from hatch.mcp.runtime import SERVER_MANAGER
from hatch.mcp.runtime import doctor
from hatch.mcp.runtime import install_shutdown_signal_handlers
from hatch.mcp.runtime import run_surface
from hatch.models import ClaudeModelAlias
from hatch.models import CodexModelAlias
from hatch.models import CursorModelAlias
from hatch.models import OpenRouterModelAlias


mcp = FastMCP(
    "hatch",
    instructions="""
Stable MCP front door for hatch.

hatch_claude runs through the official local Claude Code CLI using the user's
Claude login/subscription. hatch_cursor runs through the local Cursor Agent CLI
using the user's Cursor login/subscription (Grok 4.5 by default). hatch_codex,
hatch_gemini, and hatch_openrouter run through the OpenCode runtime. Use these
full-agent tools for web research, repo code work, and multi-step tasks.

hatch_expert is a SINGLE-CALL LLM consultation (OpenAI Responses API, GPT-5.5-pro),
NOT an agent. No tool loops, no multi-step work, no file access. It answers in one
pass with optional built-in web search. Use only for simple single-turn Q&A where
you specifically want the Responses API — otherwise use hatch_codex/claude/cursor.

Recommended defaults:
- Codex: model="mini"
- Claude: model="sonnet". Use model="fable" only when the user explicitly asks for the highest-capability local Claude tier.
- Cursor: model="grok" (Grok 4.5 HiFast via Cursor Agent)
- OpenRouter: model="deepseek-v4-pro"
- Expert: reasoning_effort="medium", web_search=true; use low for faster calls

Pass cwd for repo work. Omit cwd for one-off prompts.
Agent tool results preserve a stable hatch-style JSON envelope with status,
output, stderr, exit_code, duration, surface, resolved_model, and attach_url.
Expert results return status, output, duration, model, reasoning_effort,
web_search, usage, response_id, citations, and artifact_path.

Use batch() when you need multiple independent hatch calls from this server.
""",
)


async def _run_with_progress(
    *,
    tool_name: str,
    prompt: str,
    model: str | None = None,
    cwd: str | None = None,
    timeout_s: int = 900,
    reasoning_effort: str | None = None,
    ctx: Context | None = None,
) -> dict:
    if ctx is None:
        return await asyncio.to_thread(
            run_surface,
            tool_name=tool_name,
            prompt=prompt,
            model=model,
            cwd=cwd,
            timeout_s=timeout_s,
            reasoning_effort=reasoning_effort,
        )

    loop = asyncio.get_running_loop()
    progress_queue: asyncio.Queue[str | None] = asyncio.Queue()

    def progress_handler(message: str) -> None:
        loop.call_soon_threadsafe(progress_queue.put_nowait, message)

    def run_surface_with_progress() -> dict:
        try:
            return run_surface(
                tool_name=tool_name,
                prompt=prompt,
                model=model,
                cwd=cwd,
                timeout_s=timeout_s,
                reasoning_effort=reasoning_effort,
                progress_handler=progress_handler,
            )
        finally:
            loop.call_soon_threadsafe(progress_queue.put_nowait, None)

    async def forward_progress() -> None:
        while True:
            message = await progress_queue.get()
            if message is None:
                return
            await ctx.info(message)
            await ctx.report_progress(progress=0, total=1, message=message)

    progress_task = asyncio.create_task(forward_progress())
    run_task = asyncio.create_task(asyncio.to_thread(run_surface_with_progress))

    await ctx.report_progress(progress=0, total=1)
    try:
        result = await run_task
        await progress_task
    finally:
        if not progress_task.done():
            progress_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await progress_task

    await ctx.report_progress(progress=1, total=1)
    return result


async def _run_expert_with_progress(
    *,
    prompt: str,
    model: str = DEFAULT_EXPERT_MODEL,
    reasoning_effort: ExpertReasoningEffort = "medium",
    web_search: bool = True,
    timeout_s: int = 900,
    ctx: Context | None = None,
) -> dict:
    if ctx is None:
        result = await asyncio.to_thread(
            run_expert_sync,
            prompt=prompt,
            model=model,
            reasoning_effort=reasoning_effort,
            web_search=web_search,
            timeout_s=timeout_s,
        )
        return result.to_dict()

    await ctx.report_progress(progress=0, total=1)
    await ctx.info(
        f"[hatch] expert call started: model={model} "
        f"reasoning={reasoning_effort} web_search={str(web_search).lower()}"
    )

    loop = asyncio.get_running_loop()
    progress_queue: asyncio.Queue[str | None] = asyncio.Queue()

    def progress_handler(message: str) -> None:
        loop.call_soon_threadsafe(progress_queue.put_nowait, message)

    def run_expert_with_progress():
        try:
            return run_expert_sync(
                prompt=prompt,
                model=model,
                reasoning_effort=reasoning_effort,
                web_search=web_search,
                timeout_s=timeout_s,
                progress_handler=progress_handler,
            )
        finally:
            loop.call_soon_threadsafe(progress_queue.put_nowait, None)

    async def forward_progress() -> None:
        while True:
            message = await progress_queue.get()
            if message is None:
                return
            await ctx.info(message)
            await ctx.report_progress(progress=0, total=1, message=message)

    progress_task = asyncio.create_task(forward_progress())
    run_task = asyncio.create_task(asyncio.to_thread(run_expert_with_progress))
    try:
        result = await run_task
        await progress_task
    finally:
        if not progress_task.done():
            progress_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await progress_task

    await ctx.report_progress(progress=1, total=1)
    return result.to_dict()


@mcp.tool()
async def hatch_claude(
    model: Annotated[ClaudeModelAlias, "Claude tier. Start with sonnet. Use fable for Mythos-class (highest capability)."],
    prompt: Annotated[str, "Prompt to send to Claude via the surfaced hatch path."],
    cwd: Annotated[str | None, "Absolute repo path for repo-aware work. Omit for one-off prompts."] = None,
    timeout_s: Annotated[int, "Inner runtime timeout in seconds. Default 900."] = 900,
    ctx: Context | None = None,
) -> dict:
    """Run Claude through the official local Claude Code CLI using the user's subscription/OAuth login."""
    return await _run_with_progress(
        tool_name="hatch_claude",
        prompt=prompt,
        model=model,
        cwd=cwd,
        timeout_s=timeout_s,
        ctx=ctx,
    )


@mcp.tool()
async def hatch_cursor(
    model: Annotated[CursorModelAlias, "Cursor model. Use grok (Grok 4.5 HiFast)."],
    prompt: Annotated[str, "Prompt to send to Cursor Agent via the surfaced hatch path."],
    cwd: Annotated[str | None, "Absolute repo path for repo-aware work. Omit for one-off prompts."] = None,
    timeout_s: Annotated[int, "Inner runtime timeout in seconds. Default 900."] = 900,
    ctx: Context | None = None,
) -> dict:
    """Run Cursor Agent CLI using the user's Cursor login/subscription (Grok 4.5 HiFast)."""
    return await _run_with_progress(
        tool_name="hatch_cursor",
        prompt=prompt,
        model=model,
        cwd=cwd,
        timeout_s=timeout_s,
        ctx=ctx,
    )


@mcp.tool()
async def hatch_codex(
    model: Annotated[CodexModelAlias, "Codex tier. Start with mini."],
    prompt: Annotated[str, "Prompt to send to Codex via the surfaced hatch path."],
    cwd: Annotated[str | None, "Absolute repo path for repo-aware work. Omit for one-off prompts."] = None,
    timeout_s: Annotated[int, "Inner runtime timeout in seconds. Default 900."] = 900,
    reasoning_effort: Annotated[
        Literal["low", "medium", "high", "xhigh"] | None,
        "Optional Codex reasoning effort. Omit unless you need to tune it.",
    ] = None,
    ctx: Context | None = None,
) -> dict:
    """Run Codex (OpenAI GPT-5) as a full agent with web search, file access, and tools via OpenCode runtime."""
    return await _run_with_progress(
        tool_name="hatch_codex",
        prompt=prompt,
        model=model,
        cwd=cwd,
        timeout_s=timeout_s,
        reasoning_effort=reasoning_effort,
        ctx=ctx,
    )


@mcp.tool()
async def hatch_gemini(
    prompt: Annotated[str, "Prompt to send to Gemini through hatch's surfaced runtime path."],
    cwd: Annotated[str | None, "Absolute repo path for repo-aware work. Omit for one-off prompts."] = None,
    timeout_s: Annotated[int, "Inner runtime timeout in seconds. Default 900."] = 900,
    ctx: Context | None = None,
) -> dict:
    """Run Gemini as a full agent with web search, file access, and tools via OpenCode runtime."""
    return await _run_with_progress(
        tool_name="hatch_gemini",
        prompt=prompt,
        cwd=cwd,
        timeout_s=timeout_s,
        ctx=ctx,
    )


@mcp.tool()
async def hatch_openrouter(
    model: Annotated[OpenRouterModelAlias, "OpenRouter model. Start with deepseek-v4-pro."],
    prompt: Annotated[str, "Prompt to send to the selected OpenRouter model via hatch."],
    cwd: Annotated[str | None, "Absolute repo path for repo-aware work. Omit for one-off prompts."] = None,
    timeout_s: Annotated[int, "Inner runtime timeout in seconds. Default 900."] = 900,
    ctx: Context | None = None,
) -> dict:
    """Run OpenRouter model as a full agent with web search, file access, and tools via OpenCode runtime."""
    return await _run_with_progress(
        tool_name="hatch_openrouter",
        prompt=prompt,
        model=model,
        cwd=cwd,
        timeout_s=timeout_s,
        ctx=ctx,
    )


@mcp.tool()
async def hatch_expert(
    prompt: Annotated[
        str,
        "Prompt for a single-call consultation. NOT an agent — use hatch_codex/claude for agent work including web research.",
    ],
    reasoning_effort: Annotated[
        ExpertReasoningEffort,
        "Expert reasoning effort. Use low for faster calls; medium is the default.",
    ] = "medium",
    web_search: Annotated[
        bool,
        "Allow OpenAI web search. Defaults true; set false only for sealed local-context reasoning.",
    ] = True,
    timeout_s: Annotated[int, "Inner runtime timeout in seconds. Default 900."] = 900,
    ctx: Context | None = None,
) -> dict:
    """Single-call LLM consultation (GPT-5.5-pro + OpenAI Responses API). NOT an agent — no tool loops, no multi-step. Use hatch_codex/claude for web research and agent work."""
    return await _run_expert_with_progress(
        prompt=prompt,
        reasoning_effort=reasoning_effort,
        web_search=web_search,
        timeout_s=timeout_s,
        ctx=ctx,
    )


@mcp.tool()
async def hatch_doctor() -> dict:
    """Verify the underlying OpenCode runtime path is reachable."""
    return await asyncio.to_thread(doctor)


TOOLS = {
    "hatch_claude": hatch_claude,
    "hatch_cursor": hatch_cursor,
    "hatch_codex": hatch_codex,
    "hatch_expert": hatch_expert,
    "hatch_gemini": hatch_gemini,
    "hatch_openrouter": hatch_openrouter,
    "hatch_doctor": hatch_doctor,
}

add_batch_support(mcp, TOOLS)


def main() -> None:
    """Run the MCP server over stdio."""
    install_shutdown_signal_handlers()
    try:
        mcp.run()
    finally:
        SERVER_MANAGER.shutdown()


if __name__ == "__main__":
    main()
