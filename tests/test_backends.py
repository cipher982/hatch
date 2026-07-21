"""Tests for backend configurations."""

from __future__ import annotations

import json
import os
from unittest import mock

import pytest

from hatch.backends import (
    BOUNDED_RUN_CONTRACT,
    Backend,
    BackendConfig,
    configure_bedrock,
    configure_claude,
    configure_codex,
    configure_cursor,
    configure_gemini,
    configure_opencode,
    configure_zai,
    get_config,
    prepare_agent_prompt,
)
from hatch.backends import _dcg_binary as real_dcg_binary
from hatch.context import ExecutionContext


class TestBackendEnum:
    """Tests for Backend enum."""

    def test_backend_values(self):
        """Backend enum has expected values."""
        assert Backend.ZAI.value == "zai"
        assert Backend.CLAUDE.value == "claude"
        assert Backend.CURSOR.value == "cursor"
        assert Backend.BEDROCK.value == "bedrock"
        assert Backend.CODEX.value == "codex"
        assert Backend.GEMINI.value == "gemini"
        assert Backend.OPENCODE.value == "opencode"

    def test_backend_from_string(self):
        """Backend can be created from string."""
        assert Backend("zai") == Backend.ZAI
        assert Backend("claude") == Backend.CLAUDE
        assert Backend("cursor") == Backend.CURSOR
        assert Backend("bedrock") == Backend.BEDROCK
        assert Backend("codex") == Backend.CODEX
        assert Backend("gemini") == Backend.GEMINI
        assert Backend("opencode") == Backend.OPENCODE

    def test_invalid_backend(self):
        """Invalid backend string raises ValueError."""
        with pytest.raises(ValueError):
            Backend("invalid")

    def test_backend_is_string(self):
        """Backend values are strings."""
        for backend in Backend:
            assert isinstance(backend.value, str)
            assert backend == backend.value


class TestBackendConfig:
    """Tests for BackendConfig dataclass."""

    def test_build_env_merges_correctly(self):
        """build_env merges with os.environ correctly."""
        config = BackendConfig(
            cmd=["test"],
            env={"NEW_VAR": "new_value"},
        )
        env = config.build_env()

        # New var should be present
        assert env["NEW_VAR"] == "new_value"
        # PATH should still be there from os.environ
        assert "PATH" in env

    def test_build_env_unsets_vars(self):
        """build_env removes vars in env_unset."""
        with mock.patch.dict(os.environ, {"REMOVE_ME": "original"}):
            config = BackendConfig(
                cmd=["test"],
                env={},
                env_unset=["REMOVE_ME"],
            )
            env = config.build_env()
            assert "REMOVE_ME" not in env

    def test_build_env_overrides_existing(self):
        """build_env allows overriding existing env vars."""
        with mock.patch.dict(os.environ, {"OVERRIDE_ME": "original"}):
            config = BackendConfig(
                cmd=["test"],
                env={"OVERRIDE_ME": "new_value"},
            )
            env = config.build_env()
            assert env["OVERRIDE_ME"] == "new_value"

    def test_build_env_strips_dcg_bypass(self, monkeypatch):
        monkeypatch.setenv("DCG_BYPASS", "1")
        config = BackendConfig(cmd=["test"], env={})

        assert "DCG_BYPASS" not in config.build_env()
        assert config.build_env()["DCG_NO_SELF_HEAL"] == "1"

    def test_stdin_data_default_none(self):
        """stdin_data defaults to None."""
        config = BackendConfig(cmd=["test"], env={})
        assert config.stdin_data is None


class TestConfigureZai:
    """Tests for z.ai backend configuration."""

    def test_requires_api_key(self, clean_env, laptop_context):
        """Raises ValueError when no API key available."""
        with pytest.raises(ValueError, match="ZAI_API_KEY not set"):
            configure_zai("test prompt", laptop_context)

    def test_uses_env_api_key(self, laptop_context):
        """Uses ZAI_API_KEY from environment."""
        with mock.patch.dict(os.environ, {"ZAI_API_KEY": "env-key"}):
            config = configure_zai("test prompt", laptop_context)
            assert config.env["ANTHROPIC_AUTH_TOKEN"] == "env-key"

    def test_api_key_argument_overrides_env(self, laptop_context):
        """api_key argument overrides environment variable."""
        with mock.patch.dict(os.environ, {"ZAI_API_KEY": "env-key"}):
            config = configure_zai("test prompt", laptop_context, api_key="arg-key")
            assert config.env["ANTHROPIC_AUTH_TOKEN"] == "arg-key"

    def test_command_structure(self, mock_zai_key, laptop_context):
        """Command has correct structure."""
        config = configure_zai("test prompt", laptop_context)
        assert config.cmd == [
            "claude",
            "--print",
            "-",
            "--output-format",
            "text",
            "--dangerously-skip-permissions",
            "--setting-sources",
            "local",
            "--no-session-persistence",
            "--tools",
            "",
            "--effort",
            "low",
        ]

    def test_command_with_stream_json(self, mock_zai_key, laptop_context):
        """Command includes stream-json and partial messages when requested."""
        config = configure_zai(
            "test prompt",
            laptop_context,
            output_format="stream-json",
            include_partial_messages=True,
        )
        assert config.cmd == [
            "claude",
            "--verbose",
            "--print",
            "-",
            "--output-format",
            "stream-json",
            "--dangerously-skip-permissions",
            "--setting-sources",
            "local",
            "--no-session-persistence",
            "--tools",
            "",
            "--effort",
            "low",
            "--include-partial-messages",
        ]

    def test_env_vars_set_correctly(self, mock_zai_key, laptop_context):
        """Environment variables set correctly."""
        config = configure_zai("test prompt", laptop_context)
        assert config.env["ANTHROPIC_BASE_URL"] == "https://api.z.ai/api/anthropic"
        assert config.env["ANTHROPIC_AUTH_TOKEN"] == mock_zai_key
        assert config.env["ANTHROPIC_MODEL"] == "glm-5.1"

    def test_unsets_bedrock_vars(self, mock_zai_key, laptop_context):
        """Unsets CLAUDE_CODE_USE_BEDROCK and ANTHROPIC_API_KEY."""
        config = configure_zai("test prompt", laptop_context)
        assert "CLAUDE_CODE_USE_BEDROCK" in config.env_unset
        assert "ANTHROPIC_API_KEY" in config.env_unset

    def test_prompt_via_stdin(self, mock_zai_key, laptop_context):
        """Prompt passed via stdin_data."""
        config = configure_zai("my test prompt", laptop_context)
        assert config.stdin_data == b"my test prompt"

    def test_custom_model(self, mock_zai_key, laptop_context):
        """Custom model can be specified."""
        config = configure_zai("test", laptop_context, model="custom-model")
        assert config.env["ANTHROPIC_MODEL"] == "custom-model"

    def test_custom_base_url(self, mock_zai_key, laptop_context):
        """Custom base URL can be specified."""
        config = configure_zai("test", laptop_context, base_url="https://custom.api")
        assert config.env["ANTHROPIC_BASE_URL"] == "https://custom.api"

    def test_container_readonly_sets_home(self, mock_zai_key, container_readonly_context):
        """Sets HOME=/tmp in container with read-only home."""
        config = configure_zai("test", container_readonly_context)
        assert config.env["HOME"] == "/tmp"

    def test_container_writable_no_home_override(
        self, mock_zai_key, container_writable_context
    ):
        """Does not override HOME in container with writable home."""
        config = configure_zai("test", container_writable_context)
        assert "HOME" not in config.env


class TestConfigureBedrock:
    """Tests for Bedrock backend configuration."""

    def test_command_structure(self, laptop_context):
        """Command has correct structure."""
        config = configure_bedrock("test prompt", laptop_context)
        assert config.cmd == [
            "claude",
            "--print",
            "-",
            "--output-format",
            "text",
            "--dangerously-skip-permissions",
            "--setting-sources",
            "local",
            "--no-session-persistence",
            "--tools",
            "",
            "--effort",
            "low",
        ]

    def test_command_with_stream_json(self, laptop_context):
        """Command includes stream-json and partial messages when requested."""
        config = configure_bedrock(
            "test prompt",
            laptop_context,
            output_format="stream-json",
            include_partial_messages=True,
        )
        assert config.cmd == [
            "claude",
            "--verbose",
            "--print",
            "-",
            "--output-format",
            "stream-json",
            "--dangerously-skip-permissions",
            "--setting-sources",
            "local",
            "--no-session-persistence",
            "--tools",
            "",
            "--effort",
            "low",
            "--include-partial-messages",
        ]

    def test_env_vars_set_correctly(self, laptop_context):
        """Environment variables set correctly."""
        config = configure_bedrock("test prompt", laptop_context)
        assert config.env["CLAUDE_CODE_USE_BEDROCK"] == "1"
        assert config.env["AWS_PROFILE"] == "zh-ml-mlengineer"
        assert config.env["AWS_REGION"] == "us-east-1"
        assert config.env["ANTHROPIC_MODEL"] == "us.anthropic.claude-sonnet-4-6"

    def test_custom_aws_profile(self, laptop_context):
        """Custom AWS profile can be specified."""
        config = configure_bedrock("test", laptop_context, aws_profile="custom-profile")
        assert config.env["AWS_PROFILE"] == "custom-profile"

    def test_custom_aws_region(self, laptop_context):
        """Custom AWS region can be specified."""
        config = configure_bedrock("test", laptop_context, aws_region="eu-west-1")
        assert config.env["AWS_REGION"] == "eu-west-1"

    def test_custom_model(self, laptop_context):
        """Custom model can be specified."""
        config = configure_bedrock("test", laptop_context, model="anthropic.claude-v2")
        assert config.env["ANTHROPIC_MODEL"] == "anthropic.claude-v2"


class TestConfigureOpenCode:
    """Tests for OpenCode backend configuration."""

    def test_requires_explicit_model(self, laptop_context):
        """OpenCode needs a concrete provider/model target."""
        with pytest.raises(ValueError, match="requires an explicit model"):
            configure_opencode("test prompt", laptop_context)

    def test_command_structure_for_openai(self, laptop_context):
        """OpenCode command uses JSON event mode and pure runtime."""
        config = configure_opencode(
            "test prompt",
            laptop_context,
            model="openai/gpt-5.4",
            api_key="sk-test",
        )
        assert config.cmd == [
            "opencode",
            "run",
            "--dangerously-skip-permissions",
            "--pure",
            "--print-logs",
            "--log-level",
            "ERROR",
            "--format",
            "json",
            "-m",
            "openai/gpt-5.4",
            "test prompt",
        ]
        assert config.env["OPENAI_API_KEY"] == "sk-test"
        assert config.stdin_data is None

    def test_dcg_uses_isolated_config_without_pure(self, monkeypatch, tmp_path, laptop_context):
        """DCG loads from a dedicated XDG/config root without user or project plugins."""
        home = tmp_path / "home"
        plugin = home / ".config" / "hatch" / "dcg" / "opencode" / "plugins" / "dcg-guard.js"
        plugin.parent.mkdir(parents=True)
        source = home / "git" / "me" / "config" / "dcg" / "opencode-plugin.js"
        source.parent.mkdir(parents=True)
        source.write_text("export const DcgGuard = async () => ({});\n")
        plugin.symlink_to(source)
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setattr("hatch.backends._dcg_binary", real_dcg_binary)
        binary = home / ".local" / "bin" / "dcg"
        binary.parent.mkdir(parents=True)
        binary.write_text("#!/bin/sh\n")
        binary.chmod(0o755)
        isolated_binary = home / ".config" / "hatch" / "dcg" / "bin" / "dcg"
        isolated_binary.parent.mkdir(parents=True)
        isolated_binary.symlink_to(binary)

        config = configure_opencode(
            "test prompt",
            laptop_context,
            model="openai/gpt-5.4",
            api_key="sk-test",
        )

        assert "--pure" not in config.cmd
        assert "DCG_BIN" not in config.env
        assert config.env["XDG_CONFIG_HOME"] == str(home / ".config" / "hatch" / "dcg" / "xdg")
        assert config.env["XDG_DATA_HOME"] == str(home / ".config" / "hatch" / "dcg" / "data")
        assert config.env["XDG_CACHE_HOME"] == str(home / ".config" / "hatch" / "dcg" / "cache")
        assert config.env["XDG_STATE_HOME"] == str(home / ".config" / "hatch" / "dcg" / "state")
        assert config.env["OPENCODE_CONFIG_DIR"] == str(home / ".config" / "hatch" / "dcg" / "opencode")
        assert config.env["OPENCODE_DISABLE_PROJECT_CONFIG"] == "1"

    def test_required_dcg_rejects_missing_or_foreign_opencode_plugin(
        self, monkeypatch, tmp_path, laptop_context,
    ):
        home = tmp_path / "home"
        declaration = home / "git" / "me" / "config" / "dcg" / "release.json"
        declaration.parent.mkdir(parents=True)
        declaration.write_text('{"required": true}\n')
        binary = home / ".local" / "bin" / "dcg"
        binary.parent.mkdir(parents=True)
        binary.write_text("#!/bin/sh\n")
        binary.chmod(0o755)
        isolated_binary = home / ".config" / "hatch" / "dcg" / "bin" / "dcg"
        isolated_binary.parent.mkdir(parents=True)
        isolated_binary.symlink_to(binary)
        plugin = home / ".config" / "hatch" / "dcg" / "opencode" / "plugins" / "dcg-guard.js"
        plugin.parent.mkdir(parents=True)
        plugin.write_text("export const Foreign = true;\n")
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setattr("hatch.backends._dcg_binary", real_dcg_binary)

        with pytest.raises(ValueError, match="agents guard install"):
            configure_opencode(
                "test prompt", laptop_context, model="openai/gpt-5.4", api_key="sk-test",
            )

    @pytest.mark.parametrize("symlink_source", [False, True])
    def test_required_dcg_rejects_indirect_or_symlinked_plugin_source(
        self, monkeypatch, tmp_path, laptop_context, symlink_source,
    ):
        home = tmp_path / "home"
        source = home / "git" / "me" / "config" / "dcg" / "opencode-plugin.js"
        source.parent.mkdir(parents=True)
        real_source = tmp_path / "real-plugin.js"
        real_source.write_text("export const DcgGuard = true;\n")
        if symlink_source:
            source.symlink_to(real_source)
        else:
            source.write_text(real_source.read_text())
        declaration = source.with_name("release.json")
        declaration.write_text('{"required": true}\n')
        binary = home / ".local" / "bin" / "dcg"
        binary.parent.mkdir(parents=True)
        binary.write_text("#!/bin/sh\n")
        binary.chmod(0o755)
        isolated_binary = home / ".config" / "hatch" / "dcg" / "bin" / "dcg"
        isolated_binary.parent.mkdir(parents=True)
        isolated_binary.symlink_to(binary)
        plugin = home / ".config" / "hatch" / "dcg" / "opencode" / "plugins" / "dcg-guard.js"
        plugin.parent.mkdir(parents=True)
        plugin.symlink_to(real_source if not symlink_source else source)
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setattr("hatch.backends._dcg_binary", real_dcg_binary)

        with pytest.raises(ValueError, match="agents guard install"):
            configure_opencode(
                "test prompt", laptop_context, model="openai/gpt-5.4", api_key="sk-test",
            )

    def test_command_structure_for_openrouter(self, laptop_context):
        """OpenRouter-backed OpenCode models receive the OpenRouter API key."""
        config = configure_opencode(
            "test prompt",
            laptop_context,
            model="openrouter/deepseek/deepseek-v4-pro",
            api_key="sk-or-test",
        )
        assert config.cmd[config.cmd.index("-m") + 1] == "openrouter/deepseek/deepseek-v4-pro"
        assert config.env["OPENROUTER_API_KEY"] == "sk-or-test"

    def test_command_structure_for_openrouter_kimi_k3(self, laptop_context):
        """Kimi K3 on OpenRouter receives the OpenRouter API key."""
        config = configure_opencode(
            "test prompt",
            laptop_context,
            model="openrouter/moonshotai/kimi-k3",
            api_key="sk-or-test",
        )
        assert config.cmd[config.cmd.index("-m") + 1] == "openrouter/moonshotai/kimi-k3"
        assert config.env["OPENROUTER_API_KEY"] == "sk-or-test"

    def test_opencode_explicitly_sets_workspace_directory(self, laptop_context):
        """OpenCode must retain the requested workspace under isolated config."""
        config = configure_opencode(
            "test prompt",
            laptop_context,
            model="openai/gpt-5.6-terra",
            cwd="/workspace/project",
        )
        assert config.cmd[config.cmd.index("--dir") + 1] == "/workspace/project"

    def test_reasoning_effort_maps_to_variant(self, laptop_context):
        """OpenAI reasoning effort maps onto OpenCode variants."""
        config = configure_opencode(
            "test prompt",
            laptop_context,
            model="openai/gpt-5.4",
            reasoning_effort="xhigh",
        )
        assert config.cmd == [
            "opencode",
            "run",
            "--dangerously-skip-permissions",
            "--pure",
            "--print-logs",
            "--log-level",
            "ERROR",
            "--format",
            "json",
            "-m",
            "openai/gpt-5.4",
            "--variant",
            "xhigh",
            "test prompt",
        ]

    def test_agent_flag_is_forwarded(self, laptop_context):
        """Named OpenCode agents are forwarded."""
        config = configure_opencode(
            "review this",
            laptop_context,
            model="openai/gpt-5.4-mini",
            agent="review",
        )
        assert "--agent" in config.cmd
        assert "review" in config.cmd

    def test_bedrock_defaults(self, monkeypatch, laptop_context):
        """Bedrock-backed OpenCode models fall back to the hatch default profile."""
        monkeypatch.delenv("AWS_PROFILE", raising=False)
        monkeypatch.delenv("AWS_REGION", raising=False)
        config = configure_opencode(
            "test prompt",
            laptop_context,
            model="amazon-bedrock/us.anthropic.claude-sonnet-4-6",
        )
        assert config.env["AWS_PROFILE"] == "zh-ml-mlengineer"
        assert config.env["AWS_REGION"] == "us-east-1"

    def test_bedrock_ignores_ambient_aws_profile(self, monkeypatch, laptop_context):
        """Ambient AWS_PROFILE does not silently override the stable default."""
        monkeypatch.setenv("AWS_PROFILE", "zh-qa-engineer")
        config = configure_opencode(
            "test prompt",
            laptop_context,
            model="amazon-bedrock/us.anthropic.claude-sonnet-4-6",
        )
        assert config.env["AWS_PROFILE"] == "zh-ml-mlengineer"

    def test_bedrock_uses_marked_roulette_profile(self, monkeypatch, laptop_context):
        """The shell roulette wrapper has an explicit one-process Bedrock handoff."""
        monkeypatch.setenv("HATCH_CREDENTIAL_ROULETTE", "1")
        monkeypatch.setenv("HATCH_ROULETTE_AWS_PROFILE", "zh-qa-aiengineer")
        monkeypatch.setenv("HATCH_ROULETTE_AWS_REGION", "us-east-1")
        config = configure_opencode(
            "test prompt",
            laptop_context,
            model="amazon-bedrock/us.anthropic.claude-sonnet-4-6",
        )
        assert config.env["AWS_PROFILE"] == "zh-qa-aiengineer"

    def test_discovers_observatory_ca_without_inherited_env(self, monkeypatch, tmp_path, laptop_context):
        """OpenCode gets Observatory trust even if Hatch itself lacked inherited env."""
        home = tmp_path / "home"
        ca = home / ".local" / "state" / "agent-observatory" / "ca" / "observatory-ca.pem"
        ca.parent.mkdir(parents=True)
        ca.write_text("FAKE CA\n")
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.delenv("NODE_EXTRA_CA_CERTS", raising=False)
        monkeypatch.delenv("CODEX_CA_CERTIFICATE", raising=False)

        config = configure_opencode("test prompt", laptop_context, model="openai/gpt-5.4")

        assert config.env["NODE_EXTRA_CA_CERTS"] == str(ca)
        assert config.env["CODEX_CA_CERTIFICATE"] == str(ca)

    def test_container_readonly_sets_home(self, container_readonly_context):
        """OpenCode uses the effective writable home in read-only containers."""
        config = configure_opencode(
            "test prompt",
            container_readonly_context,
            model="openai/gpt-5.4",
        )
        assert config.env["HOME"] == container_readonly_context.effective_home

    def test_prompt_via_stdin(self, laptop_context):
        """Prompt passed via stdin_data."""
        config = configure_bedrock("my bedrock prompt", laptop_context)
        assert config.stdin_data == b"my bedrock prompt"

    def test_unsets_incompatible_anthropic_vars(self, laptop_context):
        """Clears incompatible Anthropic/z.ai env when using Bedrock."""
        config = configure_bedrock("test", laptop_context)
        assert config.env_unset == [
            "AWS_PROFILE",
            "AWS_REGION",
            "AWS_DEFAULT_REGION",
            "ANTHROPIC_AUTH_TOKEN",
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_BASE_URL",
            "HATCH_CREDENTIAL_ROULETTE",
            "HATCH_ROULETTE_OPENAI_API_KEY",
            "HATCH_ROULETTE_AWS_PROFILE",
            "HATCH_ROULETTE_AWS_REGION",
        ]

    def test_container_readonly_sets_home(self, container_readonly_context):
        """Sets HOME=/tmp in container with read-only home."""
        config = configure_bedrock("test", container_readonly_context)
        assert config.env["HOME"] == "/tmp"


class TestConfigureClaude:
    """Tests for Claude Code subscription/OAuth backend configuration."""

    def test_command_structure(self, laptop_context):
        """Claude uses local CLI OAuth mode with an explicit model."""
        config = configure_claude("test prompt", laptop_context, model="haiku")
        assert config.cmd == [
            "claude",
            "--print",
            "-",
            "--output-format",
            "text",
            "--model",
            "haiku",
            "--dangerously-skip-permissions",
            "--setting-sources",
            "local",
            "--no-session-persistence",
            "--tools",
            "default",
            "--effort",
            "low",
        ]

    def test_dcg_settings_overlay_preserves_local_only_sources(self, monkeypatch, tmp_path, laptop_context):
        """Claude receives only the explicit guard overlay, not full user settings."""
        monkeypatch.setattr("hatch.backends._dcg_binary", real_dcg_binary)
        monkeypatch.setenv("HOME", str(tmp_path))
        binary = tmp_path / ".local" / "bin" / "dcg"
        binary.parent.mkdir(parents=True)
        binary.write_text("#!/bin/sh\n")
        binary.chmod(0o755)

        config = configure_claude("test prompt", laptop_context, model="haiku")

        source_index = config.cmd.index("--setting-sources")
        assert config.cmd[source_index + 1] == "local"
        settings_index = config.cmd.index("--settings")
        settings = json.loads(config.cmd[settings_index + 1])
        assert settings["hooks"]["PreToolUse"][0]["hooks"][0]["command"] == str(binary)

    def test_required_dcg_missing_fails_closed(self, monkeypatch, tmp_path, laptop_context):
        declaration = tmp_path / "git" / "me" / "config" / "dcg" / "release.json"
        declaration.parent.mkdir(parents=True)
        declaration.write_text('{"required": true}\n')
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setattr("hatch.backends._dcg_binary", real_dcg_binary)
        monkeypatch.setenv("PATH", "/tmp/path-containing-a-different-dcg")

        with pytest.raises(ValueError, match="agents guard install"):
            configure_claude("test prompt", laptop_context, model="haiku")

    @pytest.mark.parametrize("declaration_value", ["null", "[]", '"yes"', "{}", '{"required":"true"}'])
    def test_malformed_dcg_declaration_fails_closed(
        self, monkeypatch, tmp_path, laptop_context, declaration_value,
    ):
        declaration = tmp_path / "git" / "me" / "config" / "dcg" / "release.json"
        declaration.parent.mkdir(parents=True)
        declaration.write_text(declaration_value + "\n")
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setattr("hatch.backends._dcg_binary", real_dcg_binary)

        with pytest.raises(ValueError, match="boolean `required`"):
            configure_claude("test prompt", laptop_context, model="haiku")

    def test_command_with_stream_json(self, laptop_context):
        """Claude MCP mode uses stream-json with partial messages."""
        config = configure_claude(
            "test prompt",
            laptop_context,
            model="sonnet",
            output_format="stream-json",
            include_partial_messages=True,
        )
        assert config.cmd[:7] == [
            "claude",
            "--verbose",
            "--print",
            "-",
            "--output-format",
            "stream-json",
            "--model",
        ]
        assert config.cmd[7] == "sonnet"
        assert "--include-partial-messages" in config.cmd

    def test_strips_api_key_and_provider_env(self, laptop_context):
        """Claude local subscription mode must fail closed, not use API-key providers."""
        config = configure_claude("test", laptop_context, model="haiku")
        with mock.patch.dict(os.environ, {
            "OPENROUTER_API_KEY": "or-key",
            "ANTHROPIC_API_KEY": "anthropic-key",
            "ANTHROPIC_AUTH_TOKEN": "anthropic-token",
            "ANTHROPIC_BASE_URL": "https://example.test",
            "CLAUDE_CODE_USE_BEDROCK": "1",
            "AWS_PROFILE": "some-profile",
        }):
            env = config.build_env()

        for key in [
            "OPENROUTER_API_KEY",
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_AUTH_TOKEN",
            "ANTHROPIC_BASE_URL",
            "CLAUDE_CODE_USE_BEDROCK",
            "AWS_PROFILE",
        ]:
            assert key not in env

    def test_prompt_via_stdin(self, laptop_context):
        """Prompt passed via stdin_data."""
        config = configure_claude("my claude prompt", laptop_context)
        assert config.stdin_data == b"my claude prompt"

    def test_container_readonly_sets_home(self, container_readonly_context):
        """Sets HOME in container with read-only home."""
        config = configure_claude("test", container_readonly_context)
        assert config.env["HOME"] == container_readonly_context.effective_home


class TestConfigureCursor:
    """Tests for Cursor Agent CLI backend configuration."""

    def test_command_structure(self, laptop_context):
        """Cursor uses cursor-agent -p with argv prompt and force."""
        config = configure_cursor(
            "test prompt",
            laptop_context,
            model="cursor-grok-4.5-high",
        )
        assert config.cmd == [
            "cursor-agent",
            "--print",
            "--trust",
            "--model",
            "cursor-grok-4.5-high",
            "--output-format",
            "stream-json",
            "--force",
            "test prompt",
        ]
        assert config.stdin_data is None

    def test_optional_api_key(self, laptop_context):
        """CURSOR_API_KEY is optional; passed through when present."""
        with mock.patch.dict(os.environ, {"CURSOR_API_KEY": "cursor-key"}):
            config = configure_cursor("hi", laptop_context)
            assert config.env["CURSOR_API_KEY"] == "cursor-key"

    def test_strips_competing_provider_env(self, laptop_context):
        """Cursor login path should not inherit OpenAI/Anthropic/OpenRouter keys."""
        config = configure_cursor("test", laptop_context)
        with mock.patch.dict(os.environ, {
            "OPENAI_API_KEY": "openai-key",
            "OPENROUTER_API_KEY": "or-key",
            "ANTHROPIC_API_KEY": "anthropic-key",
        }):
            env = config.build_env()
        for key in ["OPENAI_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY"]:
            assert key not in env

    def test_container_readonly_sets_home(self, container_readonly_context):
        """Sets HOME in container with read-only home."""
        config = configure_cursor("test", container_readonly_context)
        assert config.env["HOME"] == container_readonly_context.effective_home


class TestConfigureCodex:
    """Tests for Codex backend configuration."""

    def test_requires_api_key(self, clean_env, laptop_context):
        """Raises ValueError when no API key available."""
        with pytest.raises(ValueError, match="OPENAI_API_KEY not set"):
            configure_codex("test prompt", laptop_context)

    def test_uses_env_api_key(self, laptop_context):
        """Uses OPENAI_API_KEY from environment."""
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}):
            config = configure_codex("test prompt", laptop_context)
            assert config.env["OPENAI_API_KEY"] == "env-key"

    def test_api_key_argument_overrides_env(self, laptop_context):
        """api_key argument overrides environment variable."""
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}):
            config = configure_codex("test", laptop_context, api_key="arg-key")
            assert config.env["OPENAI_API_KEY"] == "arg-key"

    def test_command_structure_full_auto(self, mock_openai_key, laptop_context):
        """Command has correct structure with full-auto."""
        config = configure_codex("test prompt", laptop_context)
        assert config.cmd == [
            "codex",
            "exec",
            "--dangerously-bypass-approvals-and-sandbox",
        ]

    def test_command_structure_no_full_auto(self, mock_openai_key, laptop_context):
        """Command without full-auto flag."""
        config = configure_codex("test prompt", laptop_context, full_auto=False)
        assert config.cmd == ["codex", "exec"]

    def test_custom_model(self, mock_openai_key, laptop_context):
        """Custom model adds -m flag."""
        config = configure_codex("test", laptop_context, model="gpt-5")
        assert "-m" in config.cmd
        assert "gpt-5" in config.cmd

    def test_reasoning_effort(self, mock_openai_key, laptop_context):
        """Reasoning effort adds -c flag."""
        config = configure_codex("test", laptop_context, reasoning_effort="high")
        assert "-c" in config.cmd
        assert "model_reasoning_effort=high" in config.cmd

    def test_no_reasoning_effort_by_default(self, mock_openai_key, laptop_context):
        """No reasoning effort flag when not specified."""
        config = configure_codex("test", laptop_context)
        assert "-c" not in config.cmd

    def test_skip_git_repo_check(self, mock_openai_key, laptop_context):
        """Skip git repo check adds Codex flag."""
        config = configure_codex("test", laptop_context, skip_git_repo_check=True)
        assert "--skip-git-repo-check" in config.cmd

    def test_prompt_via_stdin(self, mock_openai_key, laptop_context):
        """Prompt passed via stdin_data."""
        config = configure_codex("my codex prompt", laptop_context)
        assert config.stdin_data == b"my codex prompt"

    def test_container_readonly_sets_home(
        self, mock_openai_key, container_readonly_context
    ):
        """Sets HOME in container with read-only home."""
        config = configure_codex("test", container_readonly_context)
        assert config.env["HOME"] == "/tmp"

    def test_unsets_leaked_bedrock_env(self, laptop_context):
        """build_env strips CLAUDE_CODE_USE_BEDROCK leaked from os.environ."""
        with mock.patch.dict(os.environ, {
            "OPENAI_API_KEY": "sk-test",
            "CLAUDE_CODE_USE_BEDROCK": "1",
        }):
            config = configure_codex("test", laptop_context)
            env = config.build_env()
            assert "CLAUDE_CODE_USE_BEDROCK" not in env


class TestConfigureGemini:
    """Tests for Gemini backend configuration."""

    def test_command_structure(self, laptop_context):
        """Command has correct structure."""
        config = configure_gemini("test prompt", laptop_context)
        assert config.cmd == ["gemini", "--model", "gemini-3-pro-preview", "--yolo", "--skip-trust", "-p", "-"]

    def test_no_api_key_required(self, clean_env, laptop_context):
        """Does not require API key (uses OAuth)."""
        # Should not raise
        config = configure_gemini("test prompt", laptop_context)
        assert config.cmd is not None

    def test_prompt_via_stdin(self, laptop_context):
        """Prompt passed via stdin_data."""
        config = configure_gemini("my gemini prompt", laptop_context)
        assert config.stdin_data == b"my gemini prompt"

    def test_minimal_env(self, laptop_context):
        """Minimal environment modifications on laptop."""
        config = configure_gemini("test", laptop_context)
        # Should not add unnecessary env vars
        assert "OPENAI_API_KEY" not in config.env
        assert "ZAI_API_KEY" not in config.env

    def test_container_readonly_sets_home(self, container_readonly_context):
        """Sets HOME in container with read-only home."""
        config = configure_gemini("test", container_readonly_context)
        assert config.env["HOME"] == "/tmp"

    def test_unsets_leaked_bedrock_env(self, laptop_context):
        """build_env strips CLAUDE_CODE_USE_BEDROCK leaked from os.environ."""
        with mock.patch.dict(os.environ, {"CLAUDE_CODE_USE_BEDROCK": "1"}):
            config = configure_gemini("test", laptop_context)
            env = config.build_env()
            assert "CLAUDE_CODE_USE_BEDROCK" not in env


class TestGetConfig:
    """Tests for the get_config dispatcher."""

    def test_zai_backend_is_disabled(self, mock_zai_key, laptop_context):
        """ZAI remains defined for compatibility but is no longer dispatchable."""
        with pytest.raises(ValueError, match="disabled"):
            get_config(Backend.ZAI, "test", laptop_context)

    def test_dispatches_to_bedrock(self, laptop_context):
        """Dispatches to configure_bedrock for BEDROCK backend."""
        config = get_config(Backend.BEDROCK, "test", laptop_context)
        assert "claude" in config.cmd
        assert config.env.get("CLAUDE_CODE_USE_BEDROCK") == "1"

    def test_dispatches_to_claude(self, laptop_context):
        """Dispatches to configure_claude for CLAUDE backend."""
        config = get_config(Backend.CLAUDE, "test", laptop_context, model="haiku")
        assert config.cmd[0] == "claude"
        assert config.cmd[config.cmd.index("--model") + 1] == "haiku"
        assert "OPENROUTER_API_KEY" in config.env_unset

    def test_dispatches_to_cursor(self, laptop_context):
        """Dispatches to configure_cursor for CURSOR backend."""
        config = get_config(
            Backend.CURSOR,
            "test",
            laptop_context,
            model="cursor-grok-4.5-high",
        )
        assert config.cmd[0] == "cursor-agent"
        assert config.cmd[config.cmd.index("--model") + 1] == "cursor-grok-4.5-high"

    def test_dispatches_to_codex(self, mock_openai_key, laptop_context):
        """Dispatches to configure_codex for CODEX backend."""
        config = get_config(Backend.CODEX, "test", laptop_context)
        assert "codex" in config.cmd

    def test_dispatches_to_gemini(self, laptop_context):
        """Dispatches to configure_gemini for GEMINI backend."""
        config = get_config(Backend.GEMINI, "test", laptop_context)
        assert "gemini" in config.cmd

    def test_dispatches_to_opencode(self, laptop_context):
        """Dispatches to configure_opencode for OPENCODE backend."""
        config = get_config(Backend.OPENCODE, "test", laptop_context, model="openai/gpt-5.4")
        assert config.cmd[0] == "opencode"

    def test_passes_kwargs(self, mock_zai_key, laptop_context):
        """Passes kwargs to configure function."""
        config = get_config(
            Backend.OPENCODE,
            "test",
            laptop_context,
            model="openai/gpt-5.4-mini",
            agent="review",
        )
        assert config.cmd[config.cmd.index("-m") + 1] == "openai/gpt-5.4-mini"
        assert config.cmd[config.cmd.index("--agent") + 1] == "review"

    def test_wraps_every_dispatched_agent_prompt(self, mock_openai_key, laptop_context):
        """All agent backends receive the same bounded-run contract."""
        bedrock = get_config(Backend.BEDROCK, "review this", laptop_context)
        claude = get_config(Backend.CLAUDE, "review this", laptop_context)
        cursor = get_config(Backend.CURSOR, "review this", laptop_context)
        codex = get_config(Backend.CODEX, "review this", laptop_context)
        gemini = get_config(Backend.GEMINI, "review this", laptop_context)
        opencode = get_config(
            Backend.OPENCODE,
            "review this",
            laptop_context,
            model="openai/gpt-5.4",
        )

        for config in (bedrock, claude, codex, gemini):
            assert config.stdin_data is not None
            assert config.stdin_data.decode().startswith(BOUNDED_RUN_CONTRACT)
        assert cursor.cmd[-1].startswith(BOUNDED_RUN_CONTRACT)
        assert opencode.cmd[-1].startswith(BOUNDED_RUN_CONTRACT)

    def test_bounded_contract_preserves_explicit_depth_request(self):
        """The contract stays additive when the user explicitly asks for depth."""
        prompt = "Perform an exhaustive, deep review of this protocol."
        wrapped = prepare_agent_prompt(prompt)

        assert "honor that instead" in wrapped
        assert wrapped.endswith(prompt)


class TestUnicodePrompts:
    """Tests for handling unicode in prompts."""

    def test_zai_unicode_prompt(self, mock_zai_key, laptop_context):
        """ZAI handles unicode prompts."""
        prompt = "Fix the bug in \u65e5\u672c\u8a9e code"
        config = configure_zai(prompt, laptop_context)
        assert config.stdin_data == prompt.encode("utf-8")

    def test_codex_unicode_prompt(self, mock_openai_key, laptop_context):
        """Codex handles unicode prompts."""
        prompt = "Analyze \U0001F680 emoji usage"  # Rocket emoji
        config = configure_codex(prompt, laptop_context)
        assert config.stdin_data == prompt.encode("utf-8")

    def test_gemini_unicode_prompt(self, laptop_context):
        """Gemini handles unicode prompts."""
        prompt = "Explain \u03c0 calculation"
        config = configure_gemini(prompt, laptop_context)
        assert config.stdin_data == prompt.encode("utf-8")

    def test_opencode_unicode_prompt(self, laptop_context):
        """OpenCode keeps unicode prompts intact in argv."""
        prompt = "Review 日本語 and 🚀 usage"
        config = configure_opencode(prompt, laptop_context, model="openai/gpt-5.4")
        assert config.cmd[-1] == prompt


class TestLargePrompts:
    """Tests for handling large prompts."""

    def test_large_prompt_via_stdin(self, mock_zai_key, laptop_context):
        """Large prompts go via stdin to avoid ARG_MAX."""
        # 100KB prompt
        large_prompt = "x" * 100_000
        config = configure_zai(large_prompt, laptop_context)
        assert len(config.stdin_data) == 100_000
        # Command should use stdin, not have prompt in args
        assert large_prompt not in " ".join(config.cmd)
