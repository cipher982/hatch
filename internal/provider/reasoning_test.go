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
		{name: "codex default", backend: "pi", model: "openai/gpt-6-astra", want: ReasoningPolicy{Effort: "medium", Source: "default", Support: "native"}},
		{name: "codex explicit max", backend: "pi", model: "openai/gpt-6-astra", requested: "max", want: ReasoningPolicy{Effort: "max", Source: "explicit", Support: "native"}},
		{name: "astra requires reasoning", backend: "pi", model: "openai/gpt-6-astra", requested: "none", wantErr: true},
		{name: "sol requires reasoning", backend: "pi", model: "openai/gpt-6.1-sol", requested: "none", wantErr: true},
		{name: "sol supports max", backend: "pi", model: "openai/gpt-6.1-sol", requested: "max", want: ReasoningPolicy{Effort: "max", Source: "explicit", Support: "native"}},
		{name: "terra default", backend: "pi", model: "openai/gpt-6.1-terra", want: ReasoningPolicy{Effort: "medium", Source: "default", Support: "native"}},
		{name: "terra supports no reasoning", backend: "pi", model: "openai/gpt-6.1-terra", requested: "none", want: ReasoningPolicy{Effort: "none", Source: "explicit", Support: "native"}},
		{name: "retired GPT-6 Sol requires an explicit choice", backend: "pi", model: "openai/gpt-6-sol", wantErr: true},
		{name: "luna supports no reasoning", backend: "pi", model: "openai/gpt-6-luna", requested: "none", want: ReasoningPolicy{Effort: "none", Source: "explicit", Support: "native"}},
		{name: "retired GPT-5.6 model requires an explicit choice", backend: "pi", model: "openai/gpt-5.6-sol", wantErr: true},
		{name: "codex model without max", backend: "pi", model: "openai/gpt-5.5", requested: "max", wantErr: true},
		{name: "unknown model requires explicit choice", backend: "pi", model: "openai/provider-model", wantErr: true},
		{name: "unknown model explicit choice is visible", backend: "pi", model: "openai/provider-model", requested: "high", want: ReasoningPolicy{Effort: "high", Source: "explicit", Support: "unknown"}},
		{name: "openrouter glm default", backend: "pi", model: "openrouter/z-ai/glm-5.3-flash", want: ReasoningPolicy{Source: "unsupported", Support: "unsupported"}},
		{name: "openrouter glm explicit high", backend: "pi", model: "openrouter/z-ai/glm-5.3-flash", requested: "high", want: ReasoningPolicy{Effort: "high", Source: "explicit", Support: "unknown"}},
		{name: "openrouter unsupported", backend: "pi", model: "openrouter/example", want: ReasoningPolicy{Source: "unsupported", Support: "unsupported"}},
		{name: "openrouter explicit effort is passed through", backend: "pi", model: "openrouter/example", requested: "high", want: ReasoningPolicy{Effort: "high", Source: "explicit", Support: "unknown"}},
		{name: "pi OpenAI default", backend: "pi", model: "openai/gpt-6-astra", want: ReasoningPolicy{Effort: "medium", Source: "default", Support: "native"}},
		{name: "omp OpenRouter explicit effort", backend: "omp", model: "openrouter/example", requested: "high", want: ReasoningPolicy{Effort: "high", Source: "explicit", Support: "unknown"}},
		{name: "claude default preserves low", backend: "claude", model: "claude-opus-5-5", want: ReasoningPolicy{Effort: "low", Source: "default", Support: "native"}},
		{name: "claude explicit effort", backend: "claude", model: "claude-opus-5-5", requested: "high", want: ReasoningPolicy{Effort: "high", Source: "explicit", Support: "native"}},
		{name: "claude rejects unsupported effort", backend: "claude", model: "claude-opus-5-5", requested: "none", wantErr: true},
		{name: "bedrock surface default matches claude cost profile", backend: "pi", model: "amazon-bedrock/global.anthropic.claude-sonnet-5-5", want: ReasoningPolicy{Effort: "low", Source: "default", Support: "native"}},
		{name: "bedrock openai terra default", backend: "pi", model: "amazon-bedrock/global.openai.gpt-6.1-terra", want: ReasoningPolicy{Effort: "medium", Source: "default", Support: "native"}},
		{name: "bedrock openai terra accepts none", backend: "pi", model: "amazon-bedrock/global.openai.gpt-6.1-terra", requested: "none", want: ReasoningPolicy{Effort: "none", Source: "explicit", Support: "native"}},
		{name: "bedrock openai sol rejects none", backend: "pi", model: "amazon-bedrock/global.openai.gpt-6.1-sol", requested: "none", wantErr: true},
		{name: "bedrock openai luna on omp", backend: "omp", model: "amazon-bedrock/global.openai.gpt-6-luna", requested: "xhigh", want: ReasoningPolicy{Effort: "xhigh", Source: "explicit", Support: "native"}},
		{name: "bedrock surface explicit effort", backend: "omp", model: "amazon-bedrock/global.anthropic.claude-opus-5-5", requested: "xhigh", want: ReasoningPolicy{Effort: "xhigh", Source: "explicit", Support: "native"}},
		{name: "bedrock surface rejects none", backend: "pi", model: "amazon-bedrock/global.anthropic.claude-opus-5-5", requested: "none", wantErr: true},
		{name: "unsurfaced bedrock model stays unsupported", backend: "pi", model: "amazon-bedrock/global.anthropic.claude-haiku-4-5-20251001-v1:0", want: ReasoningPolicy{Source: "unsupported", Support: "unsupported"}},
		{name: "raw codex default", backend: "codex", model: "gpt-6.1-sol", want: ReasoningPolicy{Effort: "medium", Source: "default", Support: "native"}},
		{name: "raw astra requires reasoning", backend: "codex", model: "gpt-6-astra", requested: "none", wantErr: true},
		{name: "raw codex known model without max", backend: "codex", model: "gpt-5.5", requested: "max", wantErr: true},
		{name: "raw codex unknown model requires explicit choice", backend: "codex", model: "gpt-test", wantErr: true},
		{name: "expert known model without max", backend: "expert", model: "gpt-5.5", requested: "max", wantErr: true},
		{name: "cursor unsupported", backend: "cursor", want: ReasoningPolicy{Source: "unsupported", Support: "unsupported"}},
		{name: "antigravity unsupported", backend: "antigravity", model: "gemini-4-argon-medium", want: ReasoningPolicy{Source: "unsupported", Support: "unsupported"}},
		{name: "antigravity rejects effort", backend: "antigravity", model: "gemini-4-argon-medium", requested: "high", wantErr: true},
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
	if _, err := ResolveReasoning("codex", "gpt-6.1-sol", "unexpected"); err == nil {
		t.Fatal("invalid effort accepted")
	}
}

func TestResolveClaudeReasoningRejectsUnsupportedEffortClearly(t *testing.T) {
	_, err := ResolveReasoning("claude", "opus", "none")
	if err == nil || !strings.Contains(err.Error(), "supported values: low, medium, high, xhigh, max") {
		t.Fatalf("ResolveReasoning() error = %v", err)
	}
}
