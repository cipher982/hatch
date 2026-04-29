from __future__ import annotations

import asyncio
from unittest import mock

from hatch.mcp.doctor import call_tool
from hatch.mcp.doctor import check_mcp_server_tools
from hatch.mcp.doctor import _recv_response
from hatch.mcp.runtime import OpenCodeRunResult
from hatch.mcp.runtime import OpenCodeServerManager
from hatch.mcp.runtime import _build_server_env
from hatch.mcp.runtime import build_run_command
from hatch.mcp.runtime import doctor
from hatch.mcp.runtime import run_surface
from hatch.mcp.server import _run_with_progress
from hatch.mcp.server import _run_expert_with_progress
from hatch.mcp.server import TOOLS


class _FakeStdout:
    def __init__(self, lines: list[str]) -> None:
        self._lines = list(lines)

    async def readline(self) -> bytes:
        return (self._lines.pop(0) if self._lines else "").encode()


class _FakeProc:
    def __init__(self, lines: list[str]) -> None:
        self.stdout = _FakeStdout(lines)


def test_build_run_command_codex_with_reasoning_and_dir():
    cmd = build_run_command(
        tool_name="hatch_codex",
        model="mini",
        prompt="review",
        attach_url="http://127.0.0.1:4196",
        cwd="/tmp",
        reasoning_effort="xhigh",
    )
    assert cmd[-1] == "review"
    assert cmd[cmd.index("--dir") + 1] == "/tmp"
    assert cmd[cmd.index("-m") + 1] == "openai/gpt-5.4-mini"
    assert cmd[cmd.index("--variant") + 1] == "high"


def test_doctor_recv_response_skips_notifications():
    proc = _FakeProc([
        '{"jsonrpc":"2.0","method":"notifications/message","params":{"level":"info"}}\n',
        '{"jsonrpc":"2.0","id":2,"result":{"ok":true}}\n',
    ])

    result = asyncio.run(_recv_response(proc, 2, timeout_s=1))

    assert result["result"]["ok"] is True


def test_build_run_command_uses_latest_frontier_aliases():
    codex_cmd = build_run_command(
        tool_name="hatch_codex",
        model="max",
        prompt="review",
        attach_url="http://127.0.0.1:4196",
    )
    claude_cmd = build_run_command(
        tool_name="hatch_claude",
        model="opus",
        prompt="review",
        attach_url="http://127.0.0.1:4196",
    )

    assert codex_cmd[codex_cmd.index("-m") + 1] == "openai/gpt-5.5"
    assert claude_cmd[claude_cmd.index("-m") + 1] == "amazon-bedrock/us.anthropic.claude-opus-4-7"


def test_build_run_command_openrouter_deepseek_alias():
    cmd = build_run_command(
        tool_name="hatch_openrouter",
        model="deepseek-v4-pro",
        prompt="review",
        attach_url="http://127.0.0.1:4196",
    )
    assert cmd[cmd.index("-m") + 1] == "openrouter/deepseek/deepseek-v4-pro"


def test_doctor_surfaces_opencode_binary():
    with mock.patch("hatch.mcp.runtime.shutil.which", return_value="/opt/homebrew/bin/opencode"):
        result = doctor()
    assert result["ok"] is True
    assert result["opencode_path"] == "/opt/homebrew/bin/opencode"
    assert result["opencode_config"].endswith("hatch/mcp/opencode.json")


def test_runtime_env_uses_isolated_xdg_paths_and_repo_config(monkeypatch, tmp_path):
    runtime_root = tmp_path / "runtime"
    monkeypatch.setenv("HATCH_MCP_OPENCODE_ROOT", str(runtime_root))
    monkeypatch.delenv("OPENCODE_CONFIG_CONTENT", raising=False)

    with mock.patch("hatch.mcp.runtime._maybe_load_secret", return_value=None):
        env = _build_server_env()

    assert env["XDG_CONFIG_HOME"] == str(runtime_root / "config")
    assert env["XDG_DATA_HOME"] == str(runtime_root / "data")
    assert env["XDG_STATE_HOME"] == str(runtime_root / "state")
    assert env["XDG_CACHE_HOME"] == str(runtime_root / "cache")
    assert env["OPENCODE_CONFIG"].endswith("hatch/mcp/opencode.json")
    assert "OPENCODE_CONFIG_CONTENT" not in env


def test_runtime_env_loads_openrouter_secret(monkeypatch, tmp_path):
    runtime_root = tmp_path / "runtime"
    monkeypatch.setenv("HATCH_MCP_OPENCODE_ROOT", str(runtime_root))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with mock.patch("hatch.mcp.runtime._maybe_load_secret") as loader:
        loader.side_effect = lambda backend: "sk-or-helper" if backend == "openrouter" else None
        env = _build_server_env()

    assert env["OPENROUTER_API_KEY"] == "sk-or-helper"


def test_configured_attach_url_shuts_down_managed_server(monkeypatch):
    manager = OpenCodeServerManager()
    fake_proc = mock.Mock()
    fake_proc.poll.return_value = None
    manager._proc = fake_proc
    manager._managed = True
    manager._url = "http://127.0.0.1:4999"

    monkeypatch.setenv("HATCH_MCP_OPENCODE_ATTACH_URL", "http://127.0.0.1:5000")
    with mock.patch("hatch.mcp.runtime._healthcheck", return_value=True):
        url = manager.ensure_server()

    assert url == "http://127.0.0.1:5000"
    fake_proc.terminate.assert_called_once()


def test_run_surface_success_uses_attach_runtime():
    fake_result = OpenCodeRunResult(
        stdout='{"type":"text"}',
        stderr="",
        return_code=0,
        timed_out=False,
        final_output="DONE",
        error_message=None,
        attach_url="http://127.0.0.1:4196",
        model="openai/gpt-5.4-mini",
    )

    with (
        mock.patch("hatch.mcp.runtime.SERVER_MANAGER.ensure_server", return_value="http://127.0.0.1:4196"),
        mock.patch("hatch.mcp.runtime.run_attached_command", return_value=fake_result),
    ):
        result = run_surface(
            tool_name="hatch_codex",
            prompt="review",
            model="mini",
            cwd="/tmp",
            timeout_s=90,
        )

    assert result["ok"] is True
    assert result["surface"] == "hatch codex"
    assert result["resolved_model"] == "openai/gpt-5.4-mini"
    assert result["attach_url"] == "http://127.0.0.1:4196"


def test_run_surface_empty_output_is_transport_error():
    fake_result = OpenCodeRunResult(
        stdout="",
        stderr="",
        return_code=0,
        timed_out=False,
        final_output=None,
        error_message=None,
        attach_url="http://127.0.0.1:4196",
        model="openai/gpt-5.4-mini",
    )

    with (
        mock.patch("hatch.mcp.runtime.SERVER_MANAGER.ensure_server", return_value="http://127.0.0.1:4196"),
        mock.patch("hatch.mcp.runtime.run_attached_command", return_value=fake_result),
    ):
        result = run_surface(
            tool_name="hatch_codex",
            prompt="hello",
            model="mini",
            timeout_s=30,
        )

    assert result["ok"] is False
    assert result["status"] == "transport_error"


def test_doctor_lists_expected_tools():
    result = asyncio.run(check_mcp_server_tools(timeout_s=5))
    assert result.ok is True
    assert "hatch_codex" in result.tools
    assert "hatch_claude" in result.tools
    assert "hatch_expert" in result.tools
    assert "hatch_openrouter" in result.tools


def test_batch_tool_map_includes_openrouter():
    assert "hatch_openrouter" in TOOLS
    assert "hatch_expert" in TOOLS


def test_build_run_command_omits_dir_when_cwd_missing():
    cmd = build_run_command(
        tool_name="hatch_codex",
        model="mini",
        prompt="hi",
        attach_url="http://127.0.0.1:4196",
    )
    assert "--dir" not in cmd


def test_run_with_progress_forwards_runtime_heartbeats():
    ctx = mock.AsyncMock()
    fake_result = {"ok": True, "status": "ok", "output": "DONE"}

    def fake_run_surface(**kwargs):
        kwargs["progress_handler"]("[hatch] Claude started")
        kwargs["progress_handler"]("[hatch] still running (30s)")
        return fake_result

    with mock.patch("hatch.mcp.server.run_surface", side_effect=fake_run_surface):
        result = asyncio.run(
            _run_with_progress(
                tool_name="hatch_claude",
                prompt="review",
                model="sonnet",
                cwd="/tmp",
                timeout_s=900,
                ctx=ctx,
            )
        )

    assert result == fake_result

    progress_calls = ctx.report_progress.await_args_list
    assert progress_calls[0].kwargs == {"progress": 0, "total": 1}
    assert any(call.kwargs.get("message") == "[hatch] Claude started" for call in progress_calls)
    assert any(call.kwargs.get("message") == "[hatch] still running (30s)" for call in progress_calls)
    assert progress_calls[-1].kwargs == {"progress": 1, "total": 1}

    info_calls = [call.args[0] for call in ctx.info.await_args_list]
    assert info_calls == ["[hatch] Claude started", "[hatch] still running (30s)"]


def test_run_expert_with_progress_forwards_heartbeats(monkeypatch):
    ctx = mock.AsyncMock()
    fake_result = mock.Mock()
    fake_result.to_dict.return_value = {"ok": True, "status": "ok", "output": "DONE"}

    async def fake_to_thread(*args, **kwargs):
        await asyncio.sleep(0.01)
        return fake_result

    monkeypatch.setattr("hatch.mcp.server.asyncio.to_thread", fake_to_thread)

    result = asyncio.run(
        _run_expert_with_progress(
            prompt="hard question",
            reasoning_effort="medium",
            web_search=False,
            timeout_s=900,
            ctx=ctx,
        )
    )

    assert result == {"ok": True, "status": "ok", "output": "DONE"}
    assert ctx.report_progress.await_args_list[0].kwargs == {"progress": 0, "total": 1}
    assert ctx.report_progress.await_args_list[-1].kwargs == {"progress": 1, "total": 1}
    info_calls = [call.args[0] for call in ctx.info.await_args_list]
    assert info_calls == [
        "[hatch] expert call started: model=gpt-5.5-pro reasoning=medium web_search=false"
    ]
