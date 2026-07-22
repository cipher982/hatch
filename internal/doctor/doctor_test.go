package doctor

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cipher982/hatch/internal/provider"
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
	if err := os.WriteFile(binary, []byte("#!/bin/sh\n[ \"$OPENROUTER_API_KEY\" = expected-secret ] || exit 9\nprintf '%s\\n' 'openrouter/deepseek/deepseek-v4-pro' 'openrouter/~moonshotai/kimi-latest'\n"), 0o700); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", directory)
	check := checkOpenCodeModels("openrouter.catalog", "openrouter", "OPENROUTER_API_KEY", Credential{Value: "expected-secret"}, modelValues(provider.OpenRouterSurfaceModels))
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
	check := checkOpenCodeModels("openrouter.catalog", "openrouter", "OPENROUTER_API_KEY", Credential{Value: "expected-secret"}, modelValues(provider.OpenRouterSurfaceModels))
	if check.OK || !strings.Contains(check.Detail, "~moonshotai/kimi-latest") || !strings.Contains(check.Detail, "--refresh") {
		t.Fatalf("check = %#v", check)
	}
}

func TestCheckOpenCodeModelsDistinguishesMissingCredential(t *testing.T) {
	check := checkOpenCodeModels("codex.catalog", "openai", "OPENAI_API_KEY", Credential{}, modelValues(provider.CodexSurfaceModels))
	if check.OK || !strings.Contains(check.Detail, "OPENAI_API_KEY is unavailable") {
		t.Fatalf("check = %#v", check)
	}
}

func TestCheckOpenCodeModelsReportsCredentialResolverFailure(t *testing.T) {
	check := checkOpenCodeModels("codex.catalog", "openai", "OPENAI_API_KEY", Credential{ResolutionError: os.ErrPermission}, modelValues(provider.CodexSurfaceModels))
	if check.OK || !strings.Contains(check.Detail, "credential resolver failed") {
		t.Fatalf("check = %#v", check)
	}
}

func TestCodexDoctorCoversEverySurfaceAlias(t *testing.T) {
	models := modelValues(provider.CodexSurfaceModels)
	if len(models) != 6 {
		t.Fatalf("doctor covers %d Codex models, want 6: %v", len(models), models)
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
