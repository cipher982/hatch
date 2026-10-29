package cli

import "testing"

func TestParseSurfacedCommands(t *testing.T) {
	tests := []struct {
		args           []string
		backend, model string
	}{
		{[]string{"claude", "haiku", "--json", "-"}, "claude", "claude-haiku-5-5"},
		{[]string{"claude", "sonnet", "--json", "-"}, "claude", "claude-sonnet-5-5"},
		{[]string{"bedrock", "opus", "--json", "-"}, "opencode", "amazon-bedrock/global.anthropic.claude-opus-5-5"},
		{[]string{"bedrock", "sonnet", "--json", "-"}, "opencode", "amazon-bedrock/global.anthropic.claude-sonnet-5-5"},
		{[]string{"claude", "fable", "--json", "-"}, "claude", "claude-fable-5-1"},
		{[]string{"claude", "fable-5.1", "--json", "-"}, "claude", "claude-fable-5-1"},
		{[]string{"cursor", "grok", "--json", "-"}, "cursor", "grok-4.7-high"},
		{[]string{"cursor", "kimi-k3", "--json", "-"}, "cursor", "kimi-k3"},
		{[]string{"cursor", "gemini-4-argon", "--json", "-"}, "cursor", "gemini-4-argon-medium"},
		{[]string{"gemini", "--json", "-"}, "omp", "google-antigravity/gemini-4-argon"},
		{[]string{"gemini", "argon", "--json", "-"}, "omp", "google-antigravity/gemini-4-argon"},
		{[]string{"codex", "astra", "--json", "-"}, "opencode", "openai/gpt-6-astra"},
		{[]string{"codex", "sol", "--json", "-"}, "opencode", "openai/gpt-6.1-sol"},
		{[]string{"codex", "terra", "--json", "-"}, "opencode", "openai/gpt-6.1-terra"},
		{[]string{"codex", "luna", "--json", "-"}, "opencode", "openai/gpt-6-luna"},
		{[]string{"openrouter", "deepseek", "--json", "-"}, "opencode", "openrouter/deepseek/deepseek-v4.1-flash-1017"},
		{[]string{"openrouter", "glm-5.3-flash", "--json", "-"}, "opencode", "openrouter/z-ai/glm-5.3-flash"},
	}
	for _, test := range tests {
		got, err := Parse(test.args, true)
		if err != nil {
			t.Fatal(err)
		}
		if got.Backend != test.backend || got.Model != test.model || len(got.PromptArgs) != 1 || got.PromptArgs[0] != "-" {
			t.Fatalf("Parse(%v) = %#v", test.args, got)
		}
	}
}
func TestParseMachineDefaults(t *testing.T) {
	got, err := Parse([]string{"-b", "gemini", "prompt"}, false)
	if err != nil {
		t.Fatal(err)
	}
	if !got.JSON || !got.Automation {
		t.Fatalf("machine defaults not enabled: %#v", got)
	}
}

func TestParseExplicitJSONIsAutomationEvenWithTTY(t *testing.T) {
	got, err := Parse([]string{"cursor", "grok", "--json", "prompt"}, true)
	if err != nil {
		t.Fatal(err)
	}
	if !got.JSON || !got.Automation {
		t.Fatalf("explicit JSON did not enable automation: %#v", got)
	}
}

func TestParseExplicitHarness(t *testing.T) {
	got, err := Parse([]string{"codex", "astra", "--harness", "omp", "--json", "prompt"}, true)
	if err != nil {
		t.Fatal(err)
	}
	if got.Backend != "opencode" || got.Harness != "omp" || got.Model != "openai/gpt-6-astra" {
		t.Fatalf("parsed harness = %#v", got)
	}
}

func TestParseCodexTierShorthand(t *testing.T) {
	got, err := Parse([]string{"astra", "--harness", "omp", "--json", "prompt"}, true)
	if err != nil {
		t.Fatal(err)
	}
	if got.Backend != "opencode" || got.Harness != "omp" || got.Model != "amazon-bedrock/global.openai.gpt-6-astra" || len(got.PromptArgs) != 1 || got.PromptArgs[0] != "prompt" {
		t.Fatalf("parsed shorthand = %#v", got)
	}
}

func TestParseModelFirstShorthands(t *testing.T) {
	tests := []struct {
		alias, backend, model string
	}{
		{"opus", "claude", "claude-opus-5-5"},
		{"sonnet", "claude", "claude-sonnet-5-5"},
		{"fable", "claude", "claude-fable-5-1"},
		{"fable-5.1", "claude", "claude-fable-5-1"},
		{"astra", "opencode", "amazon-bedrock/global.openai.gpt-6-astra"},
		{"sol", "opencode", "amazon-bedrock/global.openai.gpt-6.1-sol"},
		{"terra", "opencode", "amazon-bedrock/global.openai.gpt-6.1-terra"},
		{"luna", "opencode", "amazon-bedrock/global.openai.gpt-6-luna"},
		{"grok", "cursor", "grok-4.7-high"},
		{"kimi-k3", "cursor", "kimi-k3"},
		{"argon", "omp", "google-antigravity/gemini-4-argon"},
		{"deepseek", "opencode", "openrouter/deepseek/deepseek-v4.1-flash-1017"},
		{"glm-5.3-flash", "opencode", "openrouter/z-ai/glm-5.3-flash"},
	}
	for _, test := range tests {
		t.Run(test.alias, func(t *testing.T) {
			got, err := Parse([]string{test.alias, "prompt"}, true)
			if err != nil {
				t.Fatal(err)
			}
			if got.Backend != test.backend || got.Model != test.model || len(got.PromptArgs) != 1 || got.PromptArgs[0] != "prompt" {
				t.Fatalf("parsed %s = %#v", test.alias, got)
			}
		})
	}
}
func TestNormalizeSurfaceCompatibility(t *testing.T) {
	tests := []struct {
		name string
		args []string
		want []string
	}{
		{"explicit backend wins", []string{"-b", "zai", "codex", "review"}, []string{"-b", "zai", "codex", "review"}},
		{"explicit backend equals wins", []string{"--backend=gemini", "claude", "review"}, []string{"--backend=gemini", "claude", "review"}},
		{"explicit model wins", []string{"codex", "--model", "openai/gpt-5.4", "review"}, []string{"--backend", "opencode", "--model", "openai/gpt-5.4", "review"}},
		{"explicit model equals routes", []string{"--model=openai/gpt-5.4", "review"}, []string{"--backend", "opencode", "--model=openai/gpt-5.4", "review"}},
		{"model shorthand", []string{"astra", "review"}, []string{"--backend", "opencode", "--model", "amazon-bedrock/global.openai.gpt-6-astra", "review"}},
		{"cursor raw override", []string{"cursor", "grok", "--model", "cursor-grok-4.5-low", "review"}, []string{"--backend", "cursor", "--model", "cursor-grok-4.5-low", "review"}},
		{"option value provider", []string{"--cwd", "claude", "review"}, []string{"--cwd", "claude", "review"}},
		{"option equals provider", []string{"--cwd=claude", "review"}, []string{"--cwd=claude", "review"}},
		{"model value provider", []string{"--model", "claude", "review"}, []string{"--backend", "opencode", "--model", "claude", "review"}},
		{"double dash", []string{"--", "claude", "review"}, []string{"--", "claude", "review"}},
		{"surface help", []string{"codex", "--help"}, []string{"codex", "--help"}},
		{"surface advanced help", []string{"claude", "--advanced-help"}, []string{"claude", "--advanced-help"}},
		{"gemini default prompt", []string{"gemini", "review"}, []string{"--backend", "omp", "--model", "google-antigravity/gemini-4-argon", "review"}},
		{"gemini shorthand", []string{"argon", "review"}, []string{"--backend", "omp", "--model", "google-antigravity/gemini-4-argon", "review"}},
		{"cursor gemini surface", []string{"cursor", "gemini-4-argon", "review"}, []string{"--backend", "cursor", "--model", "gemini-4-argon-medium", "review"}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, err := normalizeSurface(test.args)
			if err != nil {
				t.Fatal(err)
			}
			if len(got) != len(test.want) {
				t.Fatalf("normalizeSurface(%v) = %v, want %v", test.args, got, test.want)
			}
			for i := range got {
				if got[i] != test.want[i] {
					t.Fatalf("normalizeSurface(%v) = %v, want %v", test.args, got, test.want)
				}
			}
		})
	}
}

func TestParseRejectsDeprecatedFable5Alias(t *testing.T) {
	_, err := Parse([]string{"claude", "fable-5", "review"}, true)
	if err == nil || err.Error() != `invalid claude model "fable-5". Choose one of: haiku, sonnet, opus, fable, fable-5.1` {
		t.Fatalf("Parse deprecated Claude alias error = %v", err)
	}
	_, err = Parse([]string{"fable-5", "review"}, true)
	if err == nil || err.Error() != `invalid claude model "fable-5". Choose one of: haiku, sonnet, opus, fable, fable-5.1` {
		t.Fatalf("Parse deprecated Claude alias shorthand error = %v", err)
	}
}
func TestParseRejectsDeprecatedOpus5Alias(t *testing.T) {
	for _, args := range [][]string{
		{"claude", "opus-5", "review"},
		{"opus-5", "review"},
	} {
		_, err := Parse(args, true)
		if err == nil || err.Error() != `invalid claude model "opus-5". Choose one of: haiku, sonnet, opus, fable, fable-5.1` {
			t.Fatalf("Parse deprecated Claude alias %v error = %v", args, err)
		}
	}
}
func TestParseRejectsDeprecatedDeepSeekAliases(t *testing.T) {
	for _, alias := range []string{"deepseek-v4.1-flash", "deepseek-v4-flash", "deepseek-v4-pro"} {
		_, err := Parse([]string{"openrouter", alias, "review"}, true)
		if err == nil || err.Error() != `invalid openrouter model "`+alias+`". Choose one of: deepseek, glm-5.3-flash` {
			t.Fatalf("Parse deprecated OpenRouter alias %q error = %v", alias, err)
		}
		_, err = Parse([]string{alias, "review"}, true)
		if err == nil || err.Error() != `invalid openrouter model "`+alias+`". Choose one of: deepseek, glm-5.3-flash` {
			t.Fatalf("Parse deprecated OpenRouter alias shorthand %q error = %v", alias, err)
		}
	}
}
func TestParseTerraRoutesToBedrockByDefault(t *testing.T) {
	tests := []struct {
		args []string
		want string
	}{
		{[]string{"terra", "review"}, "amazon-bedrock/global.openai.gpt-6.1-terra"},
		{[]string{"bedrock", "terra", "review"}, "amazon-bedrock/global.openai.gpt-6.1-terra"},
		{[]string{"codex", "terra", "review"}, "openai/gpt-6.1-terra"},
	}
	for _, test := range tests {
		got, err := Parse(test.args, true)
		if err != nil {
			t.Fatalf("Parse(%v) rejected terra: %v", test.args, err)
		}
		if got.Model != test.want || got.Backend != "opencode" {
			t.Fatalf("Parse(%v) = %#v, want %s on opencode", test.args, got, test.want)
		}
	}
}

func TestOpenRouterKimiK3IsRejected(t *testing.T) {
	if _, err := Parse([]string{"openrouter", "kimi-k3", "review"}, true); err == nil {
		t.Fatal("expected openrouter kimi-k3 to be rejected")
	}
}

func TestParseAdvancedFlagsAndLiteralPrompt(t *testing.T) {
	got, err := Parse([]string{
		"--backend=claude", "--model=opus", "--timeout=60", "--cwd=/tmp",
		"--reasoning-effort=high", "--output-format=stream-json", "--api-key=secret",
		"--resume=session", "--skip-git-repo-check", "--include-partial-messages",
		"--", "--literal", "prompt",
	}, true)
	if err != nil {
		t.Fatal(err)
	}
	if got.Backend != "claude" || got.Model != "opus" || got.TimeoutSeconds != 60 || got.CWD != "/tmp" ||
		got.ReasoningEffort != "high" || got.OutputFormat != "stream-json" || got.APIKey != "secret" ||
		got.Resume != "session" || !got.SkipGitRepoCheck || !got.IncludePartialMessages ||
		len(got.PromptArgs) != 2 || got.PromptArgs[0] != "--literal" {
		t.Fatalf("Parse advanced flags = %#v", got)
	}
}

func TestParseSurfaceHelpDoesNotRequireModel(t *testing.T) {
	got, err := Parse([]string{"codex", "--help"}, true)
	if err != nil || !got.Help {
		t.Fatalf("Parse help = %#v, %v", got, err)
	}
}

func TestParseReadOnlyFlag(t *testing.T) {
	req, err := Parse([]string{"openrouter", "deepseek", "--read-only", "--timeout", "300", "prompt"}, false)
	if err != nil {
		t.Fatal(err)
	}
	if !req.ReadOnly || req.TimeoutSeconds != 300 || req.Backend != "opencode" || len(req.PromptArgs) != 1 {
		t.Fatalf("request = %#v", req)
	}
	if plain, err := Parse([]string{"openrouter", "deepseek", "prompt"}, false); err != nil || plain.ReadOnly {
		t.Fatalf("read-only must be opt-in: %#v %v", plain, err)
	}
}
