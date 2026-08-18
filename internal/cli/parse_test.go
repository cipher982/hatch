package cli

import "testing"

func TestParseSurfacedCommands(t *testing.T) {
	tests := []struct {
		args           []string
		backend, model string
	}{
		{[]string{"claude", "haiku", "--json", "-"}, "claude", "haiku"},
		{[]string{"cursor", "grok", "--json", "-"}, "cursor", "cursor-grok-4.6-high"},
		{[]string{"cursor", "kimi-k3", "--json", "-"}, "cursor", "kimi-k3"},
		{[]string{"codex", "sol", "--json", "-"}, "opencode", "openai/gpt-5.6-sol"},
		{[]string{"gemini", "--json", "-"}, "omp", "google-antigravity/gemini-3.7-flash-tiered"},
		{[]string{"gemini", "flash", "--json", "-"}, "omp", "google-antigravity/gemini-3.7-flash-tiered"},
		{[]string{"openrouter", "deepseek-v4-flash", "--json", "-"}, "opencode", "openrouter/deepseek/deepseek-v4-flash-0731"},
		{[]string{"openrouter", "deepseek-v4-pro", "--json", "-"}, "opencode", "openrouter/deepseek/deepseek-v4-pro-0813"},
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
	got, err := Parse([]string{"codex", "sol", "--harness", "omp", "--json", "prompt"}, true)
	if err != nil {
		t.Fatal(err)
	}
	if got.Backend != "opencode" || got.Harness != "omp" || got.Model != "openai/gpt-5.6-sol" {
		t.Fatalf("parsed harness = %#v", got)
	}
}

func TestParseCodexTierShorthand(t *testing.T) {
	got, err := Parse([]string{"sol", "--harness", "omp", "--json", "prompt"}, true)
	if err != nil {
		t.Fatal(err)
	}
	if got.Backend != "opencode" || got.Harness != "omp" || got.Model != "openai/gpt-5.6-sol" || len(got.PromptArgs) != 1 || got.PromptArgs[0] != "prompt" {
		t.Fatalf("parsed shorthand = %#v", got)
	}
}

func TestParseModelFirstShorthands(t *testing.T) {
	tests := []struct {
		alias, backend, model string
	}{
		{"opus", "claude", "opus"},
		{"sol", "opencode", "openai/gpt-5.6-sol"},
		{"grok", "cursor", "cursor-grok-4.6-high"},
		{"kimi-k3", "cursor", "kimi-k3"},
		{"flash", "omp", "google-antigravity/gemini-3.7-flash-tiered"},
		{"3.7", "omp", "google-antigravity/gemini-3.7-flash-tiered"},
		{"gemini-3.7-flash-tiered", "omp", "google-antigravity/gemini-3.7-flash-tiered"},
		{"deepseek-v4-flash", "opencode", "openrouter/deepseek/deepseek-v4-flash-0731"},
		{"deepseek-v4-pro", "opencode", "openrouter/deepseek/deepseek-v4-pro-0813"},
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
		{"model shorthand", []string{"sol", "review"}, []string{"--backend", "opencode", "--model", "openai/gpt-5.6-sol", "review"}},
		{"cursor raw override", []string{"cursor", "grok", "--model", "cursor-grok-4.5-low", "review"}, []string{"--backend", "cursor", "--model", "cursor-grok-4.5-low", "review"}},
		{"option value provider", []string{"--cwd", "claude", "review"}, []string{"--cwd", "claude", "review"}},
		{"option equals provider", []string{"--cwd=claude", "review"}, []string{"--cwd=claude", "review"}},
		{"model value provider", []string{"--model", "claude", "review"}, []string{"--backend", "opencode", "--model", "claude", "review"}},
		{"double dash", []string{"--", "claude", "review"}, []string{"--", "claude", "review"}},
		{"surface help", []string{"codex", "--help"}, []string{"codex", "--help"}},
		{"surface advanced help", []string{"claude", "--advanced-help"}, []string{"claude", "--advanced-help"}},
		{"gemini default prompt", []string{"gemini", "review"}, []string{"--backend", "omp", "--model", "google-antigravity/gemini-3.7-flash-tiered", "review"}},
		{"gemini shorthand", []string{"flash", "review"}, []string{"--backend", "omp", "--model", "google-antigravity/gemini-3.7-flash-tiered", "review"}},
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

func TestParseRejectsDeprecatedGeminiAlias(t *testing.T) {
	_, err := Parse([]string{"gemini", "pro", "review"}, true)
	if err == nil || err.Error() != `invalid gemini model "pro". Choose one of: flash, 3.7, gemini-3.7-flash-tiered` {
		t.Fatalf("Parse deprecated Gemini alias error = %v", err)
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
