"""Tests for CLI interface."""

from __future__ import annotations

import io
import json
import os
import sys
from unittest import mock

import pytest

from hatch.cli import (
    EXIT_AGENT_ERROR,
    EXIT_CONFIG_ERROR,
    EXIT_NOT_FOUND,
    EXIT_SUCCESS,
    EXIT_TIMEOUT,
    create_parser,
    get_prompt,
    infer_machine_defaults,
    main,
    normalize_argv,
    result_to_exit_code,
)
from hatch.runner import AgentResult


class TestCreateParser:
    """Tests for argument parser creation."""

    def test_parser_creates(self):
        """Parser is created successfully."""
        parser = create_parser()
        assert parser is not None
        assert parser.prog == "hatch"

    def test_default_backend(self):
        """Default backend is zai."""
        parser = create_parser()
        args = parser.parse_args(["test prompt"])
        assert args.backend == "zai"

    def test_backend_short_flag(self):
        """Short -b flag works."""
        parser = create_parser()
        args = parser.parse_args(["-b", "codex", "test"])
        assert args.backend == "codex"

    def test_backend_long_flag(self):
        """Long --backend flag works."""
        parser = create_parser()
        args = parser.parse_args(["--backend", "bedrock", "test"])
        assert args.backend == "bedrock"

    def test_all_backends_valid(self):
        """All backend choices are valid."""
        parser = create_parser()
        for backend in ["zai", "bedrock", "codex", "gemini"]:
            args = parser.parse_args(["-b", backend, "test"])
            assert args.backend == backend

    def test_invalid_backend_reaches_main_validation(self):
        """Parser leaves backend validation to main()."""
        parser = create_parser()
        args = parser.parse_args(["-b", "invalid", "test"])
        assert args.backend == "invalid"

    def test_timeout_default(self):
        """Default timeout is 900."""
        parser = create_parser()
        args = parser.parse_args(["test"])
        assert args.timeout == 900

    def test_timeout_short_flag(self):
        """Short -t flag works."""
        parser = create_parser()
        args = parser.parse_args(["-t", "60", "test"])
        assert args.timeout == 60

    def test_timeout_long_flag(self):
        """Long --timeout flag works."""
        parser = create_parser()
        args = parser.parse_args(["--timeout", "120", "test"])
        assert args.timeout == 120

    def test_cwd_flag(self):
        """--cwd flag works."""
        parser = create_parser()
        args = parser.parse_args(["--cwd", "/path/to/dir", "test"])
        assert args.cwd == "/path/to/dir"

    def test_skip_git_repo_check_flag(self):
        """--skip-git-repo-check flag works."""
        parser = create_parser()
        args = parser.parse_args(["-b", "codex", "--skip-git-repo-check", "test"])
        assert args.skip_git_repo_check is True

    def test_cwd_short_flag(self):
        """Short -C flag works."""
        parser = create_parser()
        args = parser.parse_args(["-C", "/path", "test"])
        assert args.cwd == "/path"

    def test_model_flag(self):
        """--model flag works."""
        parser = create_parser()
        args = parser.parse_args(["--model", "gpt-5", "test"])
        assert args.model == "gpt-5"

    def test_output_format_flag(self):
        """--output-format flag works."""
        parser = create_parser()
        args = parser.parse_args(["--output-format", "stream-json", "test"])
        assert args.output_format == "stream-json"

    def test_output_format_default(self):
        """Default output format is text."""
        parser = create_parser()
        args = parser.parse_args(["test"])
        assert args.output_format == "text"

    def test_include_partial_messages_flag(self):
        """--include-partial-messages flag works."""
        parser = create_parser()
        args = parser.parse_args(["--include-partial-messages", "test"])
        assert args.include_partial_messages is True

    def test_api_key_flag(self):
        """--api-key flag works."""
        parser = create_parser()
        args = parser.parse_args(["--api-key", "sk-xxx", "test"])
        assert args.api_key == "sk-xxx"

    def test_json_flag(self):
        """--json flag works."""
        parser = create_parser()
        args = parser.parse_args(["--json", "test"])
        assert args.json_output is True

    def test_json_flag_default_false(self):
        """JSON output is off by default."""
        parser = create_parser()
        args = parser.parse_args(["test"])
        assert args.json_output is False

    def test_prompt_captured(self):
        """Prompt is captured from positional argument."""
        parser = create_parser()
        args = parser.parse_args(["my test prompt"])
        assert args.prompt == ["my test prompt"]

    def test_prompt_with_spaces(self):
        """Prompt with spaces works."""
        parser = create_parser()
        args = parser.parse_args(["fix", "the", "bug", "in", "auth.py"])
        assert args.prompt == ["fix", "the", "bug", "in", "auth.py"]

    def test_prompt_optional(self):
        """Prompt is optional (can read from stdin)."""
        parser = create_parser()
        args = parser.parse_args([])
        assert args.prompt == []

    def test_dash_means_stdin(self):
        """'-' as prompt means read from stdin."""
        parser = create_parser()
        args = parser.parse_args(["-"])
        assert args.prompt == ["-"]

    def test_version_flag(self):
        """--version exits cleanly."""
        parser = create_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--version"])
        assert exc_info.value.code == 0

    def test_help_flag(self):
        """--help exits cleanly."""
        parser = create_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--help"])
        assert exc_info.value.code == 0

    def test_help_text_shows_surfaced_usage(self, capsys):
        """Help should teach the surfaced Claude/Codex forms first."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--help"])
        captured = capsys.readouterr()
        assert 'hatch claude <haiku|sonnet|opus>' in captured.out
        assert 'hatch codex <nano|mini|max>' in captured.out

    def test_help_text_hides_advanced_flags(self, capsys):
        """Hidden power-user flags should stay out of the default help surface."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--help"])
        captured = capsys.readouterr()
        assert "--model" not in captured.out
        assert "--api-key" not in captured.out
        assert "--automation" not in captured.out

    def test_advanced_help_text_shows_advanced_flags(self, capsys):
        """Advanced help should expose the raw escape-hatch flags on demand."""
        parser = create_parser(show_advanced=True)
        with pytest.raises(SystemExit):
            parser.parse_args(["--help"])
        captured = capsys.readouterr()
        assert "--model" in captured.out
        assert "--api-key" in captured.out
        assert "--automation" in captured.out


class TestSpecialCommands:
    """Tests for MCP-special command dispatch."""

    def test_mcp_dispatches_to_server(self):
        with mock.patch("hatch.mcp.server.main") as mcp_main:
            exit_code = main(["mcp"])
        mcp_main.assert_called_once_with()
        assert exit_code == EXIT_SUCCESS

    def test_mcp_doctor_dispatches(self):
        with mock.patch("hatch.mcp.doctor.main", return_value=7) as doctor_main:
            exit_code = main(["mcp", "doctor", "tools"])
        doctor_main.assert_called_once_with(["tools"])
        assert exit_code == 7

    def test_mcp_unknown_subcommand_is_config_error(self, capsys):
        exit_code = main(["mcp", "wat"])
        captured = capsys.readouterr()
        assert "unknown hatch mcp subcommand" in captured.err
        assert exit_code == EXIT_CONFIG_ERROR


class TestGetPrompt:
    """Tests for get_prompt function."""

    def test_prompt_from_args(self):
        """Returns prompt from args."""
        parser = create_parser()
        args = parser.parse_args(["test", "prompt"])
        assert get_prompt(args) == "test prompt"

    def test_stdin_on_dash(self):
        """Reads stdin when prompt is '-'."""
        parser = create_parser()
        args = parser.parse_args(["-"])

        with mock.patch.object(sys, "stdin", io.StringIO("stdin prompt\n")):
            with mock.patch.object(sys.stdin, "isatty", return_value=False):
                prompt = get_prompt(args)

        assert prompt == "stdin prompt\n"

    def test_stdin_on_none(self):
        """Reads stdin when prompt is None."""
        parser = create_parser()
        args = parser.parse_args([])

        with mock.patch.object(sys, "stdin", io.StringIO("stdin prompt")):
            with mock.patch.object(sys.stdin, "isatty", return_value=False):
                prompt = get_prompt(args)

        assert prompt == "stdin prompt"

    def test_empty_stdin_exits(self):
        """Empty stdin causes exit."""
        parser = create_parser()
        args = parser.parse_args([])

        with mock.patch.object(sys, "stdin", io.StringIO("")):
            with mock.patch.object(sys.stdin, "isatty", return_value=False):
                with pytest.raises(SystemExit) as exc_info:
                    get_prompt(args)

        assert exc_info.value.code == EXIT_CONFIG_ERROR


class TestResultToExitCode:
    """Tests for result_to_exit_code function."""

    def test_success(self):
        """OK result returns EXIT_SUCCESS."""
        result = AgentResult(ok=True, output="out", exit_code=0, duration_ms=100)
        assert result_to_exit_code(result) == EXIT_SUCCESS

    def test_timeout(self):
        """Timeout returns EXIT_TIMEOUT."""
        result = AgentResult(ok=False, output="", exit_code=-1, duration_ms=5000)
        assert result_to_exit_code(result) == EXIT_TIMEOUT

    def test_not_found(self):
        """Not found returns EXIT_NOT_FOUND."""
        result = AgentResult(ok=False, output="", exit_code=-2, duration_ms=10)
        assert result_to_exit_code(result) == EXIT_NOT_FOUND

    def test_agent_error(self):
        """Agent error returns EXIT_AGENT_ERROR."""
        result = AgentResult(ok=False, output="", exit_code=1, duration_ms=100)
        assert result_to_exit_code(result) == EXIT_AGENT_ERROR


class TestNormalizeArgv:
    """Tests for surfaced provider aliases."""

    def test_leaves_plain_prompt_unchanged(self):
        """Plain prompts still use the existing default path."""
        argv = ["review", "this"]
        assert normalize_argv(argv) == argv

    def test_empty_argv_stays_empty(self):
        """No args means parse defaults and then read stdin."""
        assert normalize_argv([]) == []

    def test_codex_requires_explicit_model(self):
        """'hatch codex ...' must name nano, mini, or max explicitly."""
        with pytest.raises(ValueError, match="codex requires an explicit model"):
            normalize_argv(["codex"])

    def test_codex_model_aliases_work(self):
        """Codex shorthand models map to the surfaced family."""
        assert normalize_argv(["codex", "nano", "review"]) == [
            "--backend",
            "opencode",
            "--model",
            "openai/gpt-5.4-nano",
            "review",
        ]

    def test_claude_model_aliases_work(self):
        """Claude shorthand models map to Bedrock model IDs."""
        assert normalize_argv(["claude", "haiku", "summarize"]) == [
            "--backend",
            "opencode",
            "--model",
            "amazon-bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "summarize",
        ]

    def test_explicit_backend_wins(self):
        """Raw backend flags bypass the surfaced alias layer."""
        assert normalize_argv(["-b", "zai", "codex", "review"]) == [
            "-b",
            "zai",
            "codex",
            "review",
        ]

    def test_explicit_model_wins(self):
        """Existing --model overrides the surfaced default."""
        assert normalize_argv(["codex", "--model", "openai/gpt-5.4", "review"]) == [
            "--backend",
            "opencode",
            "--model",
            "openai/gpt-5.4",
            "review",
        ]

    def test_all_surfaced_aliases_map_correctly(self):
        """The public alias surface maps to the expected real model names."""
        assert normalize_argv(["codex", "mini", "review"]) == [
            "--backend",
            "opencode",
            "--model",
            "openai/gpt-5.4-mini",
            "review",
        ]
        assert normalize_argv(["codex", "max", "review"]) == [
            "--backend",
            "opencode",
            "--model",
            "openai/gpt-5.4",
            "review",
        ]
        assert normalize_argv(["claude", "sonnet", "review"]) == [
            "--backend",
            "opencode",
            "--model",
            "amazon-bedrock/us.anthropic.claude-sonnet-4-6",
            "review",
        ]
        assert normalize_argv(["claude", "opus", "review"]) == [
            "--backend",
            "opencode",
            "--model",
            "amazon-bedrock/us.anthropic.claude-opus-4-6-v1",
            "review",
        ]

    def test_surfaced_codex_preserves_reasoning_effort_flag(self):
        """Surfaced Codex syntax should still forward Codex-only flags."""
        assert normalize_argv(["codex", "max", "--reasoning-effort", "high", "review"]) == [
            "--backend",
            "opencode",
            "--model",
            "openai/gpt-5.4",
            "--reasoning-effort",
            "high",
            "review",
        ]

    def test_invalid_surfaced_model_is_rejected(self):
        """Surfaced providers reject unknown model names."""
        with pytest.raises(ValueError, match="invalid codex model 'full'"):
            normalize_argv(["codex", "full", "review"])
        with pytest.raises(ValueError, match="invalid claude model '4.6'"):
            normalize_argv(["claude", "4.6", "review"])

    def test_option_value_named_like_provider_is_not_rewritten(self):
        """Provider aliases inside option values do not trigger routing."""
        assert normalize_argv(["--cwd", "claude", "review"]) == [
            "--cwd",
            "claude",
            "review",
        ]
        assert normalize_argv(["--model", "claude", "review"]) == [
            "--backend",
            "opencode",
            "--model",
            "claude",
            "review",
        ]

    def test_double_dash_forces_literal_prompt(self):
        """'--' disables provider alias rewriting for prompt text."""
        assert normalize_argv(["--", "claude", "review", "this"]) == [
            "--",
            "claude",
            "review",
            "this",
        ]

    def test_help_flags_bypass_surfaced_alias_errors(self):
        """Surfaced providers should allow help without forcing a model first."""
        assert normalize_argv(["codex", "--help"]) == ["codex", "--help"]
        assert normalize_argv(["claude", "--advanced-help"]) == ["claude", "--advanced-help"]


class TestInferMachineDefaults:
    """Tests for inferred machine-friendly defaults."""

    def test_non_tty_defaults_to_json_and_automation(self):
        """Real non-interactive callers should not need to remember these flags."""
        assert infer_machine_defaults(["codex", "mini", "review"], stdout_is_tty=False) == (
            True,
            True,
        )

    def test_tty_keeps_human_defaults(self):
        """Terminal users still get plain text and no sidechain marker by default."""
        assert infer_machine_defaults(["codex", "mini", "review"], stdout_is_tty=True) == (
            False,
            False,
        )

    def test_explicit_json_and_automation_are_respected(self):
        """Explicit flags still win."""
        assert infer_machine_defaults(["--json", "--automation", "review"], stdout_is_tty=True) == (
            True,
            True,
        )


class TestMain:
    """Tests for main function."""

    @pytest.fixture
    def mock_run_sync(self):
        """Mock run_sync to avoid actual subprocess calls."""
        with mock.patch("hatch.cli.run_sync") as m:
            m.return_value = ("output", "", 0, False)
            yield m

    @pytest.fixture
    def mock_run_claude_stream_sync(self):
        """Mock Claude stream runner to avoid real subprocess calls."""
        from hatch.runner import ClaudeStreamRunResult

        with mock.patch("hatch.cli.run_claude_stream_sync") as m:
            m.return_value = ClaudeStreamRunResult(
                stdout='{"type":"result","result":"output"}\n',
                stderr="",
                return_code=0,
                timed_out=False,
                final_output="output",
            )
            yield m

    @pytest.fixture
    def mock_run_opencode_stream_sync(self):
        """Mock OpenCode stream runner to avoid real subprocess calls."""
        from hatch.runner import OpenCodeStreamRunResult

        with mock.patch("hatch.cli.run_opencode_stream_sync") as m:
            m.return_value = OpenCodeStreamRunResult(
                stdout='{"type":"text","part":{"text":"output"}}\n',
                stderr="",
                return_code=0,
                timed_out=False,
                final_output="output",
            )
            yield m

    @pytest.fixture
    def mock_get_config(self):
        """Mock get_config."""
        from hatch.backends import BackendConfig

        config = BackendConfig(
            cmd=["test"], env={}, stdin_data=b"prompt"
        )
        with mock.patch("hatch.cli.get_config", return_value=config):
            yield config

    @pytest.fixture
    def mock_hydrate_backend_kwargs(self):
        """Mock credential hydration."""
        with mock.patch("hatch.cli.hydrate_backend_kwargs", side_effect=lambda backend, kwargs: dict(kwargs)) as m:
            yield m

    @pytest.fixture
    def mock_detect_context(self):
        """Mock detect_context."""
        from hatch.context import ExecutionContext

        ctx = ExecutionContext(in_container=False, home_writable=True)
        with mock.patch("hatch.cli.detect_context", return_value=ctx):
            yield ctx

    def test_success_output(
        self,
        mock_run_claude_stream_sync,
        mock_get_config,
        mock_hydrate_backend_kwargs,
        mock_detect_context,
        mock_zai_key,
        capsys,
    ):
        """Successful run outputs result."""
        from hatch.runner import ClaudeStreamRunResult

        mock_run_claude_stream_sync.return_value = ClaudeStreamRunResult(
            stdout='{"type":"result","result":"Hello World"}\n',
            stderr="",
            return_code=0,
            timed_out=False,
            final_output="Hello World",
        )

        exit_code = main(["test prompt"])

        assert exit_code == EXIT_SUCCESS
        captured = capsys.readouterr()
        assert "Hello World" in captured.out

    def test_json_output(
        self,
        mock_run_claude_stream_sync,
        mock_get_config,
        mock_hydrate_backend_kwargs,
        mock_detect_context,
        mock_zai_key,
        capsys,
    ):
        """JSON output mode works."""
        from hatch.runner import ClaudeStreamRunResult

        mock_run_claude_stream_sync.return_value = ClaudeStreamRunResult(
            stdout='{"type":"result","result":"output text"}\n',
            stderr="",
            return_code=0,
            timed_out=False,
            final_output="output text",
        )

        exit_code = main(["--json", "test prompt"])

        assert exit_code == EXIT_SUCCESS
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["ok"] is True
        assert data["output"] == "output text"

    def test_error_to_stderr(
        self,
        mock_run_claude_stream_sync,
        mock_get_config,
        mock_hydrate_backend_kwargs,
        mock_detect_context,
        mock_zai_key,
        capsys,
    ):
        """Errors go to stderr."""
        from hatch.runner import ClaudeStreamRunResult

        mock_run_claude_stream_sync.return_value = ClaudeStreamRunResult(
            stdout='{"type":"assistant","message":{"content":[{"type":"text","text":"partial"}]}}\n',
            stderr="error msg",
            return_code=1,
            timed_out=False,
            final_output=None,
        )

        exit_code = main(["test prompt"])

        assert exit_code == EXIT_AGENT_ERROR
        captured = capsys.readouterr()
        assert "Error" in captured.err

    def test_json_error(
        self,
        mock_run_claude_stream_sync,
        mock_get_config,
        mock_hydrate_backend_kwargs,
        mock_detect_context,
        mock_zai_key,
        capsys,
    ):
        """JSON mode includes errors."""
        from hatch.runner import ClaudeStreamRunResult

        mock_run_claude_stream_sync.return_value = ClaudeStreamRunResult(
            stdout='{"type":"assistant","message":{"content":[{"type":"text","text":"partial"}]}}\n',
            stderr="error msg",
            return_code=1,
            timed_out=False,
            final_output=None,
        )

        exit_code = main(["--json", "test prompt"])

        assert exit_code == EXIT_AGENT_ERROR
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["ok"] is False
        assert data["error"] is not None

    def test_opencode_structured_error_surfaces_in_json(
        self,
        mock_run_opencode_stream_sync,
        mock_get_config,
        mock_hydrate_backend_kwargs,
        mock_detect_context,
        capsys,
    ):
        """OpenCode error events become the top-level hatch error."""
        from hatch.runner import OpenCodeStreamRunResult

        mock_run_opencode_stream_sync.return_value = OpenCodeStreamRunResult(
            stdout='{"type":"error"}\n',
            stderr="",
            return_code=0,
            timed_out=False,
            final_output=None,
            error_message="AWS session expired",
        )

        exit_code = main(["--json", "codex", "mini", "test prompt"])

        assert exit_code == EXIT_AGENT_ERROR
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["ok"] is False
        assert data["error"] == "AWS session expired"

    def test_timeout_exit_code(
        self,
        mock_run_claude_stream_sync,
        mock_get_config,
        mock_hydrate_backend_kwargs,
        mock_detect_context,
        mock_zai_key,
    ):
        """Timeout returns correct exit code."""
        from hatch.runner import ClaudeStreamRunResult

        mock_run_claude_stream_sync.return_value = ClaudeStreamRunResult(
            stdout="",
            stderr="",
            return_code=-1,
            timed_out=True,
            final_output=None,
        )

        exit_code = main(["test prompt"])

        assert exit_code == EXIT_TIMEOUT

    def test_missing_api_key(self, clean_env, mock_detect_context, capsys):
        """Missing API key returns config error."""
        with mock.patch(
            "hatch.cli.hydrate_backend_kwargs",
            side_effect=ValueError("ZAI_API_KEY not set"),
        ):
            exit_code = main(["-b", "zai", "test prompt"])

        assert exit_code == EXIT_CONFIG_ERROR
        captured = capsys.readouterr()
        assert "ZAI_API_KEY" in captured.err or "Error" in captured.err

    def test_missing_api_key_json(self, clean_env, mock_detect_context, capsys):
        """Missing API key in JSON mode."""
        with mock.patch(
            "hatch.cli.hydrate_backend_kwargs",
            side_effect=ValueError("ZAI_API_KEY not set"),
        ):
            exit_code = main(["--json", "-b", "zai", "test prompt"])

        assert exit_code == EXIT_CONFIG_ERROR
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["ok"] is False
        assert "ZAI_API_KEY" in data["error"]

    def test_cli_not_found(
        self, mock_get_config, mock_hydrate_backend_kwargs, mock_detect_context, mock_zai_key, capsys
    ):
        """CLI not found returns correct exit code."""
        with mock.patch(
            "hatch.cli.run_claude_stream_sync",
            side_effect=FileNotFoundError("claude not found"),
        ):
            exit_code = main(["test prompt"])

        assert exit_code == EXIT_NOT_FOUND

    def test_backend_passed_correctly(
        self, mock_run_claude_stream_sync, mock_hydrate_backend_kwargs, mock_detect_context, mock_zai_key
    ):
        """Backend is passed to get_config."""

        with mock.patch("hatch.cli.get_config") as mock_config:
            from hatch.backends import BackendConfig, Backend

            mock_config.return_value = BackendConfig(
                cmd=["test"], env={}, stdin_data=b"test"
            )

            main(["-b", "bedrock", "test"])

            mock_config.assert_called_once()
            args = mock_config.call_args[0]
            assert args[0] == Backend.BEDROCK

    def test_explicit_model_routes_to_opencode_backend(
        self,
        mock_run_opencode_stream_sync,
        mock_hydrate_backend_kwargs,
        mock_detect_context,
    ):
        """Explicit models without a raw backend flag should use the surfaced runtime."""
        with mock.patch("hatch.cli.get_config") as mock_config:
            from hatch.backends import BackendConfig, Backend

            mock_config.return_value = BackendConfig(
                cmd=["test"], env={}, stdin_data=b"test"
            )

            main(["--model", "openai/gpt-5.4-mini", "test prompt"])

            args = mock_config.call_args[0]
            kwargs = mock_config.call_args[1]
            assert args[0] == Backend.OPENCODE
            assert kwargs["model"] == "openai/gpt-5.4-mini"

    def test_model_kwarg_passed(
        self, mock_run_opencode_stream_sync, mock_hydrate_backend_kwargs, mock_detect_context
    ):
        """Model is passed as kwarg."""
        with mock.patch("hatch.cli.get_config") as mock_config:
            from hatch.backends import BackendConfig

            mock_config.return_value = BackendConfig(
                cmd=["test"], env={}, stdin_data=b"test"
            )

            main(["--model", "custom-model", "test"])

            kwargs = mock_config.call_args[1]
            assert kwargs.get("model") == "custom-model"

    def test_api_key_kwarg_passed(
        self, mock_run_claude_stream_sync, mock_detect_context
    ):
        """API key is passed into credential hydration."""
        with mock.patch("hatch.cli.get_config") as mock_config:
            from hatch.backends import BackendConfig

            mock_config.return_value = BackendConfig(
                cmd=["test"], env={}, stdin_data=b"test"
            )

            with mock.patch(
                "hatch.cli.hydrate_backend_kwargs",
                side_effect=lambda backend, kwargs: dict(kwargs),
            ) as mock_hydrate:
                main(["--api-key", "sk-test", "-b", "zai", "test"])

            hydrated_kwargs = mock_hydrate.call_args[0][1]
            assert hydrated_kwargs.get("api_key") == "sk-test"

    def test_timeout_passed(
        self, mock_run_sync, mock_get_config, mock_hydrate_backend_kwargs, mock_detect_context, mock_zai_key
    ):
        """Timeout is passed to run_sync."""
        mock_run_sync.return_value = ("output", "", 0, False)

        main(["-b", "codex", "-t", "60", "test prompt"])

        call_args = mock_run_sync.call_args[0]
        assert call_args[4] == 60  # timeout_s is 5th positional arg

    def test_invalid_timeout(self, capsys):
        """Non-positive timeout returns config error."""
        exit_code = main(["-t", "0", "test prompt"])
        assert exit_code == EXIT_CONFIG_ERROR
        captured = capsys.readouterr()
        assert "timeout" in captured.err.lower()

    def test_invalid_timeout_json(self, capsys):
        """Non-positive timeout returns config error in JSON mode."""
        exit_code = main(["--json", "-t", "-5", "test prompt"])
        assert exit_code == EXIT_CONFIG_ERROR
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["ok"] is False
        assert "timeout" in data["error"].lower()

    def test_invalid_backend(self, capsys):
        """Unknown backends return a config error."""
        exit_code = main(["-b", "invalid", "test prompt"])
        assert exit_code == EXIT_CONFIG_ERROR
        captured = capsys.readouterr()
        assert "invalid backend" in captured.err.lower()

    def test_invalid_backend_json(self, capsys):
        """Unknown backends return structured config errors in JSON mode."""
        exit_code = main(["--json", "-b", "invalid", "test prompt"])
        assert exit_code == EXIT_CONFIG_ERROR
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["ok"] is False
        assert "invalid backend" in data["error"].lower()

    def test_skip_git_repo_check_rejected_for_surfaced_codex(self, capsys):
        """Surfaced Codex should reject raw-Codex-only flags instead of silently ignoring them."""
        exit_code = main(["codex", "mini", "--skip-git-repo-check", "test prompt"])
        assert exit_code == EXIT_CONFIG_ERROR
        captured = capsys.readouterr()
        assert "not supported for surfaced providers" in captured.err

    def test_reasoning_effort_rejected_for_claude(self, capsys):
        """Reasoning effort should fail loudly when used on unsupported model families."""
        exit_code = main(["claude", "sonnet", "--reasoning-effort", "high", "test prompt"])
        assert exit_code == EXIT_CONFIG_ERROR
        captured = capsys.readouterr()
        assert "only works with codex models" in captured.err.lower()

    def test_zai_model_rejected_on_shared_runtime(self, capsys):
        """z.ai should stay on the stable plain-hatch path until the surfaced runtime is reliable."""
        exit_code = main(["--model", "zai/glm-5.1", "test prompt"])
        assert exit_code == EXIT_CONFIG_ERROR
        captured = capsys.readouterr()
        assert "only available through plain hatch" in captured.err.lower()

    def test_cwd_passed(
        self, mock_run_sync, mock_get_config, mock_hydrate_backend_kwargs, mock_detect_context, mock_zai_key
    ):
        """CWD is passed to run_sync."""
        mock_run_sync.return_value = ("output", "", 0, False)

        # Use /tmp which exists on all systems
        main(["-b", "codex", "--cwd", "/tmp", "test prompt"])

        call_args = mock_run_sync.call_args[0]
        assert call_args[3] == "/tmp"  # cwd is 4th positional arg

    def test_surfaced_codex_timeout_passed_to_opencode_runner(
        self,
        mock_run_opencode_stream_sync,
        mock_get_config,
        mock_hydrate_backend_kwargs,
        mock_detect_context,
    ):
        """Surfaced Codex should pass timeout through the OpenCode runner."""
        main(["codex", "mini", "-t", "60", "test prompt"])

        call_args = mock_run_opencode_stream_sync.call_args[0]
        assert call_args[4] == 60

    def test_surfaced_codex_cwd_passed_to_opencode_runner(
        self,
        mock_run_opencode_stream_sync,
        mock_get_config,
        mock_hydrate_backend_kwargs,
        mock_detect_context,
    ):
        """Surfaced Codex should pass cwd through the OpenCode runner."""
        main(["codex", "mini", "--cwd", "/tmp", "test prompt"])

        call_args = mock_run_opencode_stream_sync.call_args[0]
        assert call_args[3] == "/tmp"

    def test_invalid_cwd(self, capsys):
        """Invalid cwd returns config error."""
        exit_code = main(["--cwd", "/does/not/exist", "test prompt"])
        assert exit_code == EXIT_CONFIG_ERROR
        captured = capsys.readouterr()
        assert "cwd" in captured.err.lower()

    def test_invalid_cwd_json(self, capsys):
        """Invalid cwd returns config error in JSON mode."""
        exit_code = main(["--json", "--cwd", "/does/not/exist", "test prompt"])
        assert exit_code == EXIT_CONFIG_ERROR
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["ok"] is False
        assert "cwd" in data["error"].lower()

    def test_missing_surfaced_model_returns_config_error(self, capsys):
        """Provider aliases without a model are rejected cleanly."""
        exit_code = main(["codex"])
        assert exit_code == EXIT_CONFIG_ERROR
        captured = capsys.readouterr()
        assert "explicit model" in captured.err.lower()

    def test_missing_surfaced_model_json_returns_config_error(self, capsys):
        """Machine callers get structured alias parse errors."""
        exit_code = main(["--json", "claude"])
        assert exit_code == EXIT_CONFIG_ERROR
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["ok"] is False
        assert "explicit model" in data["error"].lower()

    def test_strips_trailing_whitespace(
        self,
        mock_run_claude_stream_sync,
        mock_get_config,
        mock_hydrate_backend_kwargs,
        mock_detect_context,
        mock_zai_key,
        capsys,
    ):
        """Output trailing whitespace is stripped."""
        from hatch.runner import ClaudeStreamRunResult

        mock_run_claude_stream_sync.return_value = ClaudeStreamRunResult(
            stdout='{"type":"result","result":"output\\n\\n\\n"}\n',
            stderr="",
            return_code=0,
            timed_out=False,
            final_output="output\n\n\n",
        )

        main(["test prompt"])

        captured = capsys.readouterr()
        # Should have exactly one newline (from print)
        assert captured.out == "output\n"

    def test_raw_zai_defaults_to_internal_stream_json(
        self, mock_run_claude_stream_sync, mock_hydrate_backend_kwargs, mock_detect_context, mock_zai_key
    ):
        """Raw Claude backends should use stream-json transport internally."""
        with mock.patch("hatch.cli.get_config") as mock_config:
            from hatch.backends import BackendConfig

            mock_config.return_value = BackendConfig(
                cmd=["test"], env={}, stdin_data=b"test"
            )

            main(["-b", "zai", "test prompt"])

            kwargs = mock_config.call_args[1]
            assert kwargs["output_format"] == "stream-json"
            assert kwargs["include_partial_messages"] is True

    def test_hydrated_kwargs_passed_to_get_config(
        self, mock_run_sync, mock_detect_context, mock_zai_key
    ):
        """Credential hydration output is what get_config receives."""
        mock_run_sync.return_value = ("output", "", 0, False)

        with mock.patch("hatch.cli.get_config") as mock_config:
            from hatch.backends import BackendConfig

            mock_config.return_value = BackendConfig(
                cmd=["test"], env={}, stdin_data=b"test"
            )

            with mock.patch(
                "hatch.cli.hydrate_backend_kwargs",
                return_value={"api_key": "resolved-key", "model": "custom"},
            ):
                main(["-b", "codex", "--model", "custom", "test"])

            kwargs = mock_config.call_args[1]
            assert kwargs["api_key"] == "resolved-key"
            assert kwargs["model"] == "custom"

    def test_uses_sys_argv_when_no_argv_provided(
        self,
        mock_run_sync,
        mock_run_opencode_stream_sync,
        mock_get_config,
        mock_hydrate_backend_kwargs,
        mock_detect_context,
        mock_zai_key,
        capsys,
    ):
        """Real CLI invocations still read arguments from sys.argv."""
        mock_run_sync.return_value = ("Hello", "", 0, False)
        mock_run_opencode_stream_sync.return_value.final_output = "Hello"

        with mock.patch.object(sys, "argv", ["hatch", "codex", "mini", "review", "this"]):
            exit_code = main()

        assert exit_code == EXIT_SUCCESS
        captured = capsys.readouterr()
        assert "Hello" in captured.out


class TestMainWithStdin:
    """Tests for main with stdin input."""

    def test_reads_stdin(self, mock_zai_key, capsys):
        """Reads prompt from stdin when not provided."""
        with mock.patch.object(sys, "stdin", io.StringIO("stdin prompt")):
            with mock.patch.object(sys.stdin, "isatty", return_value=False):
                from hatch.runner import ClaudeStreamRunResult

                with mock.patch("hatch.cli.run_claude_stream_sync") as mock_run:
                    mock_run.return_value = ClaudeStreamRunResult(
                        stdout='{"type":"result","result":"output"}\n',
                        stderr="",
                        return_code=0,
                        timed_out=False,
                        final_output="output",
                    )

                    with mock.patch("hatch.cli.get_config") as mock_config:
                        from hatch.backends import BackendConfig

                        mock_config.return_value = BackendConfig(
                            cmd=["test"], env={}, stdin_data=b"test"
                        )

                        with mock.patch(
                            "hatch.cli.hydrate_backend_kwargs",
                            side_effect=lambda backend, kwargs: dict(kwargs),
                        ):
                            with mock.patch("hatch.cli.detect_context"):
                                main([])

                    # Verify get_config was called with stdin prompt
                    call_args = mock_config.call_args[0]
                    assert call_args[1] == "stdin prompt"
