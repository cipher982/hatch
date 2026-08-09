package provider

import (
	"reflect"
	"strings"
	"testing"
)

func TestPreparePromptOracle(t *testing.T) {
	got := PreparePrompt("oracle prompt")
	for _, want := range []string{
		"A human is waiting for a useful answer by the behavioral deadline",
		"Do not promise or assume an exact wall-clock duration",
		"Use focused checks by default",
		"Time-box expensive tests, scratch clones or worktrees, broad repository scans, and network or fetch work",
		"Check the budget mid-run",
		"Once evidence is sufficient, stop using tools and synthesize",
		"At the late budget threshold, stop launching tools",
		"Preserve useful partial findings and do not redo completed work",
		"Nested Hatch runs are allowed",
		"not a ban on child Hatch calls",
		"Never wait indefinitely for a child",
		"findings, confidence, unresolved questions, and the exact next action",
		"must never be presented as approved or complete",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("prepared prompt missing guardrail %q", want)
		}
	}
	if !strings.HasSuffix(got, "User task:\noracle prompt") {
		t.Fatalf("prepared prompt does not preserve user task: %q", got)
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
		if !reflect.DeepEqual(got.Argv[len(got.Argv)-len(wantTail):], wantTail) || got.Adapter != "claude" ||
			got.ReasoningPolicy != (ReasoningPolicy{Effort: "low", Source: "default", Support: "native"}) {
			t.Fatalf("claude invocation = %#v", got)
		}
	})

	t.Run("claude passes explicit effort", func(t *testing.T) {
		got, err := Build(Request{Backend: "claude", Model: "opus", Prompt: "p", ReasoningEffort: "high"})
		if err != nil {
			t.Fatal(err)
		}
		want := []string{
			"claude", "--verbose", "--print", "-", "--output-format", "stream-json",
			"--model", "opus", "--dangerously-skip-permissions", "--setting-sources", "local",
			"--no-session-persistence", "--tools", "default", "--effort", "high", "--include-partial-messages",
		}
		if !reflect.DeepEqual(got.Argv, want) || got.ReasoningPolicy != (ReasoningPolicy{Effort: "high", Source: "explicit", Support: "native"}) {
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
			"codex", "exec", "--dangerously-bypass-approvals-and-sandbox", "--ignore-user-config", "--ephemeral", "-m", "gpt-5.6",
			"-c", "model_reasoning_effort=high", "--skip-git-repo-check",
		}
		if !reflect.DeepEqual(got.Argv, want) || got.SetEnv["OPENAI_API_KEY"] != "secret" {
			t.Fatalf("codex invocation = %#v", got)
		}
	})

	t.Run("opencode defaults reasoning explicitly", func(t *testing.T) {
		got, err := Build(Request{Backend: "opencode", Model: "openai/gpt-5.6-sol", Prompt: "p", APIKey: "secret"})
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got.Argv[len(got.Argv)-3:], []string{"--variant", "medium", PreparePrompt("p")}) ||
			got.ReasoningPolicy != (ReasoningPolicy{Effort: "medium", Source: "default", Support: "native"}) {
			t.Fatalf("opencode invocation = %#v", got)
		}
	})

	for _, backend := range []string{"pi", "omp"} {
		t.Run(backend+" uses explicit headless JSON mode", func(t *testing.T) {
			got, err := Build(Request{Backend: backend, Model: "openai/gpt-5.6-sol", Prompt: "p", APIKey: "secret"})
			if err != nil {
				t.Fatal(err)
			}
			if got.Argv[0] != backend || got.StreamFormat != "jsonl" || got.Adapter != backend || got.Argv[len(got.Argv)-1] != PreparePrompt("p") ||
				got.SetEnv["OPENAI_API_KEY"] != "secret" || got.ReasoningPolicy.Effort != "medium" {
				t.Fatalf("%s invocation = %#v", backend, got)
			}
			joined := strings.Join(got.Argv, " ")
			for _, want := range []string{"--mode json", "--no-session", "--model openai/gpt-5.6-sol", "--thinking medium"} {
				if !strings.Contains(joined, want) {
					t.Fatalf("%s argv lacks %q: %#v", backend, want, got.Argv)
				}
			}
		})
	}

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
