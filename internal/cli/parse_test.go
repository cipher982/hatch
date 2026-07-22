package cli

import "testing"

func TestParseSurfacedCommands(t *testing.T) {
	tests := []struct {
		args           []string
		backend, model string
	}{
		{[]string{"claude", "haiku", "--json", "-"}, "claude", "haiku"},
		{[]string{"cursor", "grok", "--json", "-"}, "cursor", "cursor-grok-4.5-high"},
		{[]string{"openrouter", "kimi-k3", "--json", "-"}, "opencode", "openrouter/moonshotai/kimi-k3"},
		{[]string{"codex", "sol", "--json", "-"}, "opencode", "openai/gpt-5.6-sol"},
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
