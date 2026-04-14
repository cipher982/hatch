from __future__ import annotations

import asyncio
from unittest import mock

from hatch.mcp.doctor import call_tool
from hatch.mcp.doctor import check_mcp_server_tools
from hatch.mcp.runtime import OpenCodeRunResult
from hatch.mcp.runtime import OpenCodeServerManager
from hatch.mcp.runtime import _build_server_env
from hatch.mcp.runtime import build_run_command
from hatch.mcp.runtime import doctor
from hatch.mcp.runtime import run_surface


def test_build_run_command_default():
    cmd = build_run_command(
        tool_name="hatch_default",
        prompt="hello",
        attach_url="http://127.0.0.1:4196",
    )
    assert cmd[-1] == "hello"
    assert "--attach" in cmd
    assert "--format" in cmd
    assert "--pure" in cmd
    assert "-m" in cmd


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
        model="zai/glm-5.1",
    )

    with (
        mock.patch("hatch.mcp.runtime.SERVER_MANAGER.ensure_server", return_value="http://127.0.0.1:4196"),
        mock.patch("hatch.mcp.runtime.run_attached_command", return_value=fake_result),
    ):
        result = run_surface(
            tool_name="hatch_default",
            prompt="hello",
            timeout_s=30,
        )

    assert result["ok"] is False
    assert result["status"] == "transport_error"


def test_doctor_lists_expected_tools():
    result = asyncio.run(check_mcp_server_tools(timeout_s=5))
    assert result.ok is True
    assert "hatch_default" in result.tools
    assert "hatch_codex" in result.tools
    assert "hatch_claude" in result.tools


def test_call_tool_missing_cwd_fails_fast():
    result = asyncio.run(
        call_tool(
            "hatch_codex",
            {"model": "mini", "prompt": "hi"},
            timeout_s=10,
        )
    )
    assert result["ok"] is False
    assert "cwd" in result["error"]
