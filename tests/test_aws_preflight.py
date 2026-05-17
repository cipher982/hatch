from __future__ import annotations

import subprocess
from unittest import mock

import pytest

from hatch.aws_preflight import BedrockAwsAuthError
from hatch.aws_preflight import bedrock_aws_profile
from hatch.aws_preflight import preflight_bedrock_aws
from hatch.aws_preflight import uses_bedrock_aws


def test_uses_bedrock_aws_for_opencode_bedrock_model():
    assert uses_bedrock_aws("amazon-bedrock/us.anthropic.claude-opus-4-7")


def test_uses_bedrock_aws_for_raw_claude_env():
    assert uses_bedrock_aws(env={"CLAUDE_CODE_USE_BEDROCK": "1"})


def test_bedrock_aws_profile_defaults_to_hatch_profile(monkeypatch):
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    assert bedrock_aws_profile({}) == "zh-qa-engineer"


def test_preflight_skips_non_bedrock_models():
    with mock.patch("hatch.aws_preflight.subprocess.run") as run:
        preflight_bedrock_aws("openai/gpt-5.4-mini", {})

    run.assert_not_called()


def test_preflight_raises_actionable_sso_error():
    completed = subprocess.CompletedProcess(
        args=["aws"],
        returncode=255,
        stdout="",
        stderr="The SSO session associated with this profile has expired.",
    )

    with mock.patch("hatch.aws_preflight.subprocess.run", return_value=completed):
        with pytest.raises(BedrockAwsAuthError) as exc:
            preflight_bedrock_aws(
                "amazon-bedrock/us.anthropic.claude-opus-4-7",
                {"AWS_PROFILE": "zh-qa-engineer", "AWS_REGION": "us-east-1"},
            )

    message = str(exc.value)
    assert "AWS_PROFILE=zh-qa-engineer" in message
    assert "aws sso login --profile zh-qa-engineer" in message
