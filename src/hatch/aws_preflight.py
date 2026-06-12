"""AWS credential preflight checks for Bedrock-backed Hatch surfaces."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping


# zh-ml-mlengineer is the account with the Bedrock data-retention opt-in
# (provider_data_share) required by Mythos-class models like Fable 5, and it
# also serves the haiku/sonnet/opus tiers. Keeping one profile for every tier
# matters because the MCP path runs a single shared `opencode serve` whose AWS
# env is fixed at server startup and cannot switch per-model per-request.
DEFAULT_BEDROCK_AWS_PROFILE = "zh-ml-mlengineer"
DEFAULT_BEDROCK_AWS_REGION = "us-east-1"


class BedrockAwsAuthError(ValueError):
    """Raised when Bedrock AWS credentials are not ready for a Hatch run."""


def uses_bedrock_aws(model: str | None = None, env: Mapping[str, str] | None = None) -> bool:
    """Return whether this Hatch run needs AWS Bedrock credentials."""
    if model and model.startswith("amazon-bedrock/"):
        return True
    return bool(env and env.get("CLAUDE_CODE_USE_BEDROCK") == "1")


def bedrock_aws_profile(env: Mapping[str, str] | None = None) -> str:
    """Resolve the AWS profile Hatch will use for Bedrock."""
    if env and env.get("AWS_PROFILE"):
        return str(env["AWS_PROFILE"])
    return os.environ.get("AWS_PROFILE") or DEFAULT_BEDROCK_AWS_PROFILE


def bedrock_aws_region(env: Mapping[str, str] | None = None) -> str:
    """Resolve the AWS region Hatch will use for Bedrock."""
    if env and env.get("AWS_REGION"):
        return str(env["AWS_REGION"])
    return os.environ.get("AWS_REGION") or DEFAULT_BEDROCK_AWS_REGION


def _clean_aws_error(raw: str) -> str:
    detail = " ".join(raw.strip().split())
    if not detail:
        return "AWS CLI returned a non-zero exit code"
    if len(detail) > 500:
        return f"{detail[:497]}..."
    return detail


def preflight_bedrock_aws(
    model: str | None = None,
    env: Mapping[str, str] | None = None,
    *,
    timeout_s: float = 8.0,
) -> None:
    """Fail early with an actionable message when Bedrock AWS auth is expired."""
    if not uses_bedrock_aws(model, env):
        return

    profile = bedrock_aws_profile(env)
    region = bedrock_aws_region(env)
    login_hint = f"aws sso login --profile {profile}"
    cmd = [
        "aws",
        "sts",
        "get-caller-identity",
        "--profile",
        profile,
        "--region",
        region,
        "--output",
        "json",
    ]
    preflight_env = dict(os.environ)
    preflight_env["AWS_PAGER"] = ""

    try:
        result = subprocess.run(
            cmd,
            env=preflight_env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except FileNotFoundError as exc:
        raise BedrockAwsAuthError(
            "Bedrock AWS credentials could not be checked because the AWS CLI was not found"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise BedrockAwsAuthError(
            f"Bedrock AWS credential preflight timed out for AWS_PROFILE={profile}; "
            f"refresh with: {login_hint}"
        ) from exc

    if result.returncode == 0:
        return

    detail = _clean_aws_error(result.stderr or result.stdout)
    raise BedrockAwsAuthError(
        f"Bedrock AWS credentials are not ready for AWS_PROFILE={profile}: {detail}. "
        f"Refresh with: {login_hint}"
    )
