package doctor

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cipher982/hatch/internal/provider"
)

func TestParseCursorModelIDs(t *testing.T) {
	got := ParseCursorModelIDs("cursor-auto - Auto\ncursor-grok-4.6-high - Grok 4.6 High\nnoise\n")
	if _, ok := got["cursor-grok-4.6-high"]; !ok || len(got) != 2 {
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
	if err := os.WriteFile(binary, []byte("#!/bin/sh\n[ \"$OPENROUTER_API_KEY\" = expected-secret ] || exit 9\nprintf '%s\\n' 'openrouter/deepseek/deepseek-v4-flash-0731' 'openrouter/deepseek/deepseek-v4-pro-0813' 'openrouter/z-ai/glm-5.3-flash'\n"), 0o700); err != nil {
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
	if err := os.WriteFile(binary, []byte("#!/bin/sh\nprintf '%s\\n' 'openrouter/other'\n"), 0o700); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", directory)
	check := checkOpenCodeModels("openrouter.catalog", "openrouter", "OPENROUTER_API_KEY", Credential{Value: "expected-secret"}, modelValues(provider.OpenRouterSurfaceModels))
	if check.OK || !strings.Contains(check.Detail, "deepseek-v4-flash-0731") || !strings.Contains(check.Detail, "deepseek-v4-pro-0813") || !strings.Contains(check.Detail, "glm-5.3-flash") || !strings.Contains(check.Detail, "--refresh") {
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
	if err := os.WriteFile(binary, []byte("#!/bin/sh\nprintf '%s\\n' 'cursor-grok-4.6-high - Grok' 'kimi-k3 - Kimi K3'\n"), 0o700); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", directory)
	check := checkCursorModel(Credential{})
	if !check.OK || check.Name != "cursor.catalog" {
		t.Fatalf("check = %#v", check)
	}
}

func TestCheckCursorModelMissing(t *testing.T) {
	t.Setenv("PATH", t.TempDir())
	check := checkCursorModel(Credential{})
	if check.OK || !strings.Contains(check.Detail, "not installed") {
		t.Fatalf("check = %#v", check)
	}
}

func TestCheckHarness(t *testing.T) {
	directory := t.TempDir()
	binary := filepath.Join(directory, "omp")
	if err := os.WriteFile(binary, []byte("#!/bin/sh\nprintf '%s\\n' 'omp v17.2.10'\n"), 0o700); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", directory)
	check := checkHarness("harness.omp", "omp")
	if !check.OK || check.Name != "harness.omp" || check.Detail != "omp v17.2.10" {
		t.Fatalf("check = %#v", check)
	}
}

func TestCheckOMPModels(t *testing.T) {
	directory := t.TempDir()
	binary := filepath.Join(directory, "omp")
	script := `#!/bin/sh
printf '%s\n' '{"models":[{"id":"gemini-3.7-flash-tiered","selector":"google-antigravity/gemini-3.7-flash-tiered"}]}'
`
	if err := os.WriteFile(binary, []byte(script), 0o700); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", directory)
	check := checkOMPModels("gemini.catalog", modelValues(provider.GeminiSurfaceModels))
	if !check.OK || check.Name != "gemini.catalog" {
		t.Fatalf("check = %#v", check)
	}
}
