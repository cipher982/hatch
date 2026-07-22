package provider

import (
	"crypto/sha256"
	"encoding/hex"
	"reflect"
	"testing"
)

func TestPreparePromptOracle(t *testing.T) {
	got := []byte(PreparePrompt("oracle prompt"))
	digest := sha256.Sum256(got)
	if len(got) != 570 || hex.EncodeToString(digest[:]) != "0eb7749ad17c17ce3323e0628fc9a6fc040d640d95edb7d6839f983ac354ee54" {
		t.Fatalf("prepared prompt len=%d sha256=%s", len(got), hex.EncodeToString(digest[:]))
	}
}

func TestBuildOracleInvocations(t *testing.T) {
	tests := []struct {
		name string
		req  Request
		argv []string
	}{
		{"gemini", Request{Backend: "gemini", Prompt: "oracle prompt"}, []string{"gemini", "--model", "gemini-3-pro-preview", "--yolo", "--skip-trust", "-p", "-"}},
		{"cursor", Request{Backend: "cursor", Model: "cursor-grok-4.5-high", Prompt: "oracle prompt"}, []string{"cursor-agent", "--print", "--trust", "--model", "cursor-grok-4.5-high", "--output-format", "stream-json", "--force", PreparePrompt("oracle prompt")}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, err := Build(test.req)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got.Argv, test.argv) {
				t.Fatalf("argv = %#v, want %#v", got.Argv, test.argv)
			}
		})
	}
}

func TestBuildAdvancedBackendInvocations(t *testing.T) {
	t.Run("claude resume and explicit stream", func(t *testing.T) {
		got, err := Build(Request{
			Backend: "claude", Model: "opus", Prompt: "p", OutputFormat: "stream-json",
			IncludePartialMessages: true, Resume: "ses_1",
		})
		if err != nil {
			t.Fatal(err)
		}
		wantTail := []string{"--include-partial-messages", "--resume", "ses_1"}
		if !reflect.DeepEqual(got.Argv[len(got.Argv)-len(wantTail):], wantTail) || got.Adapter != "claude" {
			t.Fatalf("claude invocation = %#v", got)
		}
	})

	t.Run("explicit claude stream remains raw", func(t *testing.T) {
		got, err := Build(Request{Backend: "claude", Model: "opus", Prompt: "p", OutputFormat: "stream-json", RawStructuredOutput: true})
		if err != nil {
			t.Fatal(err)
		}
		if got.Adapter != "raw" || got.StreamFormat != "jsonl" {
			t.Fatalf("explicit stream invocation = %#v", got)
		}
	})

	t.Run("raw codex", func(t *testing.T) {
		got, err := Build(Request{
			Backend: "codex", Model: "gpt-5.6", Prompt: "p", APIKey: "secret",
			ReasoningEffort: "high", SkipGitRepoCheck: true,
		})
		if err != nil {
			t.Fatal(err)
		}
		want := []string{
			"codex", "exec", "--dangerously-bypass-approvals-and-sandbox", "-m", "gpt-5.6",
			"-c", "model_reasoning_effort=high", "--skip-git-repo-check",
		}
		if !reflect.DeepEqual(got.Argv, want) || got.SetEnv["OPENAI_API_KEY"] != "secret" {
			t.Fatalf("codex invocation = %#v", got)
		}
	})

	t.Run("bedrock defaults", func(t *testing.T) {
		got, err := Build(Request{Backend: "bedrock", Prompt: "p", OutputFormat: "text"})
		if err != nil {
			t.Fatal(err)
		}
		if got.SetEnv["AWS_PROFILE"] != "zh-ml-mlengineer" || got.SetEnv["AWS_REGION"] != "us-east-1" ||
			got.SetEnv["ANTHROPIC_MODEL"] != "us.anthropic.claude-sonnet-4-6" || got.Adapter != "claude" {
			t.Fatalf("bedrock invocation = %#v", got)
		}
	})
}
