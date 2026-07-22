package cli

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/cipher982/hatch/internal/provider"
)

const defaultAWSProfile = "zh-ml-mlengineer"
const defaultAWSRegion = "us-east-1"

func preflightBedrock(model string, invocation provider.Invocation) error {
	usesBedrock := strings.HasPrefix(model, "amazon-bedrock/") || invocation.SetEnv["CLAUDE_CODE_USE_BEDROCK"] == "1"
	if !usesBedrock {
		return nil
	}
	profile := strings.TrimSpace(invocation.SetEnv["AWS_PROFILE"])
	if profile == "" {
		profile = strings.TrimSpace(os.Getenv("AWS_PROFILE"))
	}
	if profile == "" {
		profile = defaultAWSProfile
	}
	region := strings.TrimSpace(invocation.SetEnv["AWS_REGION"])
	if region == "" {
		region = strings.TrimSpace(os.Getenv("AWS_REGION"))
	}
	if region == "" {
		region = defaultAWSRegion
	}
	loginHint := "aws sso login --profile " + profile
	ctx, cancel := context.WithTimeout(context.Background(), 8*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "aws", "sts", "get-caller-identity", "--profile", profile, "--region", region, "--output", "json")
	cmd.Env = append(os.Environ(), "AWS_PAGER=")
	stdout, err := cmd.Output()
	if ctx.Err() == context.DeadlineExceeded {
		return fmt.Errorf("Bedrock AWS credential preflight timed out for AWS_PROFILE=%s; refresh with: %s", profile, loginHint)
	}
	if err == nil {
		return nil
	}
	if _, ok := err.(*exec.Error); ok {
		return fmt.Errorf("Bedrock AWS credentials could not be checked because the AWS CLI was not found")
	}
	detail := ""
	if exit, ok := err.(*exec.ExitError); ok {
		detail = cleanAWSError(string(exit.Stderr))
	}
	if detail == "" {
		detail = cleanAWSError(string(stdout))
	}
	if detail == "" {
		detail = "AWS CLI returned a non-zero exit code"
	}
	return fmt.Errorf("Bedrock AWS credentials are not ready for AWS_PROFILE=%s: %s. Refresh with: %s", profile, detail, loginHint)
}

func cleanAWSError(value string) string {
	value = strings.Join(strings.Fields(value), " ")
	if len(value) > 500 {
		return value[:497] + "..."
	}
	return value
}
