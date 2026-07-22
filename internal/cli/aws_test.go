package cli

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cipher982/hatch/internal/provider"
)

func TestPreflightBedrockSkipsOtherProviders(t *testing.T) {
	t.Setenv("PATH", t.TempDir())
	if err := preflightBedrock("openai/gpt-5.6", provider.Invocation{}); err != nil {
		t.Fatal(err)
	}
}

func TestPreflightBedrockSuccess(t *testing.T) {
	directory := t.TempDir()
	binary := filepath.Join(directory, "aws")
	if err := os.WriteFile(binary, []byte("#!/bin/sh\nexit 0\n"), 0o700); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", directory)
	invocation := provider.Invocation{SetEnv: map[string]string{"CLAUDE_CODE_USE_BEDROCK": "1"}}
	if err := preflightBedrock("", invocation); err != nil {
		t.Fatal(err)
	}
}

func TestPreflightBedrockFailureIsActionable(t *testing.T) {
	directory := t.TempDir()
	binary := filepath.Join(directory, "aws")
	if err := os.WriteFile(binary, []byte("#!/bin/sh\necho ' expired   token ' >&2\nexit 2\n"), 0o700); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", directory)
	invocation := provider.Invocation{SetEnv: map[string]string{"AWS_PROFILE": "profile", "AWS_REGION": "region"}}
	err := preflightBedrock("amazon-bedrock/model", invocation)
	if err == nil || !strings.Contains(err.Error(), "AWS_PROFILE=profile: expired token") || !strings.Contains(err.Error(), "aws sso login --profile profile") {
		t.Fatalf("error = %v", err)
	}
}
