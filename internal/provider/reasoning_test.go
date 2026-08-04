package provider

import (
	"strings"
	"testing"
)

func TestResolveReasoningPolicy(t *testing.T) {
	tests := []struct {
		name      string
		backend   string
		model     string
		requested string
		want      ReasoningPolicy
		wantErr   bool
	}{
		{name: "codex default", backend: "opencode", model: "openai/gpt-5.6-sol", want: ReasoningPolicy{Effort: "medium", Source: "default", Support: "native"}},
		{name: "codex explicit max", backend: "opencode", model: "openai/gpt-5.6-sol", requested: "max", want: ReasoningPolicy{Effort: "max", Source: "explicit", Support: "native"}},
		{name: "codex model without max", backend: "opencode", model: "openai/gpt-5.5", requested: "max", wantErr: true},
		{name: "unknown model requires explicit choice", backend: "opencode", model: "openai/provider-model", wantErr: true},
		{name: "unknown model explicit choice is visible", backend: "opencode", model: "openai/provider-model", requested: "high", want: ReasoningPolicy{Effort: "high", Source: "explicit", Support: "unknown"}},
		{name: "openrouter unsupported", backend: "opencode", model: "openrouter/example", want: ReasoningPolicy{Source: "unsupported", Support: "unsupported"}},
		{name: "openrouter rejects override", backend: "opencode", model: "openrouter/example", requested: "high", wantErr: true},
		{name: "claude default preserves low", backend: "claude", model: "opus", want: ReasoningPolicy{Effort: "low", Source: "default", Support: "native"}},
		{name: "claude explicit effort", backend: "claude", model: "opus", requested: "high", want: ReasoningPolicy{Effort: "high", Source: "explicit", Support: "native"}},
		{name: "claude rejects unsupported effort", backend: "claude", model: "opus", requested: "none", wantErr: true},
		{name: "bedrock fixed", backend: "bedrock", model: "claude", want: ReasoningPolicy{Effort: "low", Source: "fixed", Support: "fixed"}},
		{name: "bedrock rejects override", backend: "bedrock", model: "claude", requested: "high", wantErr: true},
		{name: "raw codex default", backend: "codex", model: "gpt-5.6", want: ReasoningPolicy{Effort: "medium", Source: "default", Support: "native"}},
		{name: "raw codex known model without max", backend: "codex", model: "gpt-5.5", requested: "max", wantErr: true},
		{name: "raw codex unknown model requires explicit choice", backend: "codex", model: "gpt-test", wantErr: true},
		{name: "expert known model without max", backend: "expert", model: "gpt-5.5", requested: "max", wantErr: true},
		{name: "cursor unsupported", backend: "cursor", want: ReasoningPolicy{Source: "unsupported", Support: "unsupported"}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, err := ResolveReasoning(test.backend, test.model, test.requested)
			if test.wantErr {
				if err == nil {
					t.Fatalf("ResolveReasoning() error = nil, policy = %#v", got)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if got != test.want {
				t.Fatalf("ResolveReasoning() = %#v, want %#v", got, test.want)
			}
		})
	}
}

func TestResolveReasoningRejectsInvalidEffort(t *testing.T) {
	if _, err := ResolveReasoning("codex", "gpt-5.6", "unexpected"); err == nil {
		t.Fatal("invalid effort accepted")
	}
}

func TestResolveClaudeReasoningRejectsUnsupportedEffortClearly(t *testing.T) {
	_, err := ResolveReasoning("claude", "opus", "none")
	if err == nil || !strings.Contains(err.Error(), "supported values: low, medium, high, xhigh, max") {
		t.Fatalf("ResolveReasoning() error = %v", err)
	}
}
