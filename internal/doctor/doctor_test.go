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

const piModelTable = `provider        model                          context  max-out  thinking  images
openrouter      z-ai/glm-5.3-flash             1.0M     943.7K   yes       yes
openrouter      deepseek/deepseek-v4.1-flash   1.0M     943.7K   yes       yes
openai          gpt-6-astra                    272K     128K     yes       yes
amazon-bedrock  global.openai.gpt-6-astra      1.1M     128K     yes       yes
`

func TestParsePiModelIDs(t *testing.T) {
	got := ParsePiModelIDs(piModelTable)
	if _, ok := got["openai/gpt-6-astra"]; !ok || len(got) != 4 {
		t.Fatalf("models = %#v", got)
	}
	if _, header := got["provider/model"]; header {
		t.Fatal("table header parsed as a model")
	}
}

func fakePi(t *testing.T, listing string) {
	t.Helper()
	directory := t.TempDir()
	script := "#!/bin/sh\nwhile IFS= read -r line; do printf '%s\\n' \"$line\"; done <<'EOF'\n" + listing + "EOF\n"
	if err := os.WriteFile(filepath.Join(directory, "pi"), []byte(script), 0o700); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", directory)
}

func TestCheckPiModelsAcceptsCatalogAndHatchDeclaredModels(t *testing.T) {
	fakePi(t, piModelTable)
	// gpt-6.1-sol and the 1017 DeepSeek snapshot are absent from the listing:
	// Hatch declares them in its own models.json.
	check := checkPiModels("bedrock.catalog", []string{"amazon-bedrock/global.openai.gpt-6-astra", "amazon-bedrock/global.openai.gpt-6.1-sol"})
	if !check.OK {
		t.Fatalf("check = %#v", check)
	}
	check = checkPiModels("openrouter.catalog", modelValues(provider.OpenRouterSurfaceModels))
	if !check.OK {
		t.Fatalf("check = %#v", check)
	}
}

func TestCheckPiModelsDetectsDrift(t *testing.T) {
	fakePi(t, "provider model\nopenrouter other 1M 1M yes yes\n")
	check := checkPiModels("openrouter.catalog", []string{"openrouter/z-ai/glm-5.3-flash"})
	if check.OK || !strings.Contains(check.Detail, "glm-5.3-flash") || !strings.Contains(check.Detail, "pi update") {
		t.Fatalf("check = %#v", check)
	}
}

func TestCheckPiModelsReportsMissingPi(t *testing.T) {
	t.Setenv("PATH", t.TempDir())
	check := checkPiModels("codex.catalog", []string{"openai/gpt-6-astra"})
	if check.OK || !strings.Contains(check.Detail, "not installed") {
		t.Fatalf("check = %#v", check)
	}
}

func TestCodexDoctorCoversEverySurfaceAlias(t *testing.T) {
	models := modelValues(provider.CodexSurfaceModels)
	if len(models) != 7 {
		t.Fatalf("doctor covers %d Codex models, want 7: %v", len(models), models)
	}
}

func TestCheckCursorModel(t *testing.T) {
	directory := t.TempDir()
	binary := filepath.Join(directory, "cursor-agent")
	if err := os.WriteFile(binary, []byte("#!/bin/sh\nprintf '%s\\n' 'grok-4.7-high - Grok' 'kimi-k3 - Kimi K3' 'gemini-4-argon-medium - Gemini 4 Argon Medium'\n"), 0o700); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", directory)
	check := checkCursorModel(Credential{})
	if !check.OK || check.Name != "cursor.catalog" || check.Detail != "3 configured models are available" {
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

func TestCheckAntigravityModels(t *testing.T) {
	directory := t.TempDir()
	binary := filepath.Join(directory, "agy")
	script := `#!/bin/sh
printf 'Fetching available models...\n'
printf 'gemini-4-carbon-medium\tGemini 4 Carbon (Medium)\n'
printf 'gemini-4-argon-medium\tGemini 4 Argon (Medium)\n'
`
	if err := os.WriteFile(binary, []byte(script), 0o700); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", directory)
	check := checkAntigravityModels("gemini.catalog", modelValues(provider.GeminiSurfaceModels))
	if !check.OK || check.Name != "gemini.catalog" {
		t.Fatalf("check = %#v", check)
	}
}

func TestCheckAntigravityModelsReportsMissingModel(t *testing.T) {
	directory := t.TempDir()
	binary := filepath.Join(directory, "agy")
	script := `#!/bin/sh
printf 'gemini-4-carbon-medium\tGemini 4 Carbon (Medium)\n'
`
	if err := os.WriteFile(binary, []byte(script), 0o700); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", directory)
	check := checkAntigravityModels("gemini.catalog", []string{"gemini-4-argon-medium"})
	if check.OK || !strings.Contains(check.Detail, "gemini-4-argon-medium") {
		t.Fatalf("check = %#v", check)
	}
}

func TestCheckCursorModelProbesUnlistedAliases(t *testing.T) {
	directory := t.TempDir()
	script := `#!/bin/sh
if [ "$1" = "models" ]; then printf '%s\n' 'grok-4.7-high - Grok' 'gemini-4-argon-medium - Gemini 4 Argon Medium'; exit 0; fi
while [ "$1" != "--model" ]; do shift; done
case "$2" in
  kimi-k3) echo "ActionRequiredError: You've reached your monthly usage limit"; exit 1 ;;
  *) echo "Cannot use this model: $2. Available models: auto"; exit 1 ;;
esac
`
	if err := os.WriteFile(filepath.Join(directory, "cursor-agent"), []byte(script), 0o700); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", directory)
	check := checkCursorModel(Credential{})
	if !check.OK || !strings.Contains(check.Detail, "1 accepted by Cursor but not listed") {
		t.Fatalf("unlisted-but-valid alias must pass: %#v", check)
	}

	original := provider.CursorSurfaceModels["kimi-k3"]
	provider.CursorSurfaceModels["kimi-k3"] = "retired-model"
	defer func() { provider.CursorSurfaceModels["kimi-k3"] = original }()
	check = checkCursorModel(Credential{})
	if check.OK || !strings.Contains(check.Detail, "retired-model") {
		t.Fatalf("rejected alias must fail: %#v", check)
	}
}
