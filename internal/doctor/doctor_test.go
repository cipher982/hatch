package doctor

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestParseCursorModelIDs(t *testing.T) {
	got := ParseCursorModelIDs("cursor-auto - Auto\ncursor-grok-4.5-high - Grok 4.5 High\nnoise\n")
	if _, ok := got["cursor-grok-4.5-high"]; !ok || len(got) != 2 {
		t.Fatalf("models = %#v", got)
	}
}

func TestParseOpenCodeModelIDs(t *testing.T) {
	got := ParseOpenCodeModelIDs("openai/gpt-5.6-sol\nopenai/gpt-5.6-terra\n")
	if _, ok := got["openai/gpt-5.6-sol"]; !ok || len(got) != 2 {
		t.Fatalf("models = %#v", got)
	}
}

func TestCheckOpenCodeModels(t *testing.T) {
	directory := t.TempDir()
	binary := filepath.Join(directory, "opencode")
	if err := os.WriteFile(binary, []byte("#!/bin/sh\nprintf '%s\\n' 'openrouter/deepseek/deepseek-v4-pro' 'openrouter/~moonshotai/kimi-latest'\n"), 0o700); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", directory)
	check := checkOpenCodeModels("openrouter.catalog", "openrouter", OpenRouterModels)
	if !check.OK || check.Name != "openrouter.catalog" {
		t.Fatalf("check = %#v", check)
	}
}

func TestCheckOpenCodeModelsDetectsDrift(t *testing.T) {
	directory := t.TempDir()
	binary := filepath.Join(directory, "opencode")
	if err := os.WriteFile(binary, []byte("#!/bin/sh\nprintf '%s\\n' 'openrouter/deepseek/deepseek-v4-pro'\n"), 0o700); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", directory)
	check := checkOpenCodeModels("openrouter.catalog", "openrouter", OpenRouterModels)
	if check.OK || !strings.Contains(check.Detail, "~moonshotai/kimi-latest") || !strings.Contains(check.Detail, "--refresh") {
		t.Fatalf("check = %#v", check)
	}
}

func TestCheckCursorModel(t *testing.T) {
	directory := t.TempDir()
	binary := filepath.Join(directory, "cursor-agent")
	if err := os.WriteFile(binary, []byte("#!/bin/sh\nprintf '%s\\n' 'cursor-grok-4.5-high - Grok'\n"), 0o700); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", directory)
	check := checkCursorModel()
	if !check.OK || check.Name != "cursor.grok" {
		t.Fatalf("check = %#v", check)
	}
}

func TestCheckCursorModelMissing(t *testing.T) {
	t.Setenv("PATH", t.TempDir())
	check := checkCursorModel()
	if check.OK || !strings.Contains(check.Detail, "not installed") {
		t.Fatalf("check = %#v", check)
	}
}
