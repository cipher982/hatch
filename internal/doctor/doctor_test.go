package doctor

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cipher982/hatch/internal/provider"
)

func TestParseCursorModelIDs(t *testing.T) {
	got := ParseCursorModelIDs("cursor-auto - Auto\ngrok-4.7-high - Grok 4.7 High\nnoise\n")
	if _, ok := got["grok-4.7-high"]; !ok || len(got) != 2 {
		t.Fatalf("models = %#v", got)
	}
}

func TestParseOpenCodeModelIDs(t *testing.T) {
	got := ParseOpenCodeModelIDs("openai/gpt-6-sol\nopenai/gpt-6-luna\n")
	if _, ok := got["openai/gpt-6-sol"]; !ok || len(got) != 2 {
		t.Fatalf("models = %#v", got)
	}
}

func TestCheckOpenAIGPT6ModelsAreInjectedIntoCatalogProbe(t *testing.T) {
	directory := t.TempDir()
	binary := filepath.Join(directory, "opencode")
	script := `#!/bin/sh
[ "$OPENAI_API_KEY" = expected-secret ] || exit 9
case "$OPENCODE_CONFIG_CONTENT" in *gpt-6-sol* ) ;; *) exit 8 ;; esac
case "$OPENCODE_CONFIG_CONTENT" in *gpt-6-luna* ) ;; *) exit 8 ;; esac
printf '%s\n' 'openai/gpt-6-astra' 'openai/gpt-6-sol' 'openai/gpt-6-luna' 'openai/gpt-5.4-nano' 'openai/gpt-5.4-mini' 'openai/gpt-5.5'
`
	if err := os.WriteFile(binary, []byte(script), 0o700); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", directory)
	check := checkOpenCodeModels("codex.catalog", "openai", "OPENAI_API_KEY", Credential{Value: "expected-secret"}, modelValues(provider.CodexSurfaceModels))
	if !check.OK || check.Name != "codex.catalog" {
		t.Fatalf("check = %#v", check)
	}
}

func TestCheckOpenCodeModels(t *testing.T) {
	directory := t.TempDir()
	binary := filepath.Join(directory, "opencode")
	if err := os.WriteFile(binary, []byte("#!/bin/sh\n[ \"$OPENROUTER_API_KEY\" = expected-secret ] || exit 9\nprintf '%s\\n' 'openrouter/deepseek/deepseek-v4.1-flash' 'openrouter/z-ai/glm-5.3-flash'\n"), 0o700); err != nil {
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
	if check.OK || !strings.Contains(check.Detail, "deepseek-v4.1-flash") || !strings.Contains(check.Detail, "glm-5.3-flash") || !strings.Contains(check.Detail, "--refresh") {
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
	if err := os.WriteFile(binary, []byte("#!/bin/sh\nprintf '%s\\n' 'grok-4.7-high - Grok' 'kimi-k3 - Kimi K3'\n"), 0o700); err != nil {
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
printf '%s\n' '{"models":[{"id":"gemini-3.8-flash-low","selector":"google-antigravity/gemini-3.8-flash-low"}]}'
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
