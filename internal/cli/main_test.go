package cli

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"testing"

	"github.com/cipher982/hatch/internal/provider"
)

func TestMainRawCodexVerticalSlice(t *testing.T) {
	root := t.TempDir()
	fake := filepath.Join(root, "testprovider")
	command := exec.Command("go", "build", "-o", fake, "./internal/testprovider")
	command.Dir = filepath.Join("..", "..")
	if output, err := command.CombinedOutput(); err != nil {
		t.Fatalf("build test provider: %v\n%s", err, output)
	}
	if err := os.Symlink(fake, filepath.Join(root, "codex")); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", root+string(os.PathListSeparator)+os.Getenv("PATH"))
	t.Setenv("HATCH_RUN_ARTIFACT_ROOT", filepath.Join(root, "runs"))
	t.Setenv("HATCH_TEST_SCENARIO", "success_text")
	t.Setenv("HATCH_TEST_RECORD", filepath.Join(root, "invocation.json"))

	var stdout, stderr bytes.Buffer
	exitCode := Main([]string{"-b", "codex", "--api-key", "fake-test-key", "--json", "-"}, bytes.NewBufferString("oracle prompt"), &stdout, &stderr, true)
	if exitCode != 0 {
		t.Fatalf("exit=%d stderr=%s stdout=%s", exitCode, stderr.String(), stdout.String())
	}
	var result struct {
		OK           bool   `json:"ok"`
		Output       string `json:"output"`
		ArtifactPath string `json:"artifact_path"`
		Run          struct {
			RunID     string `json:"run_id"`
			Lifecycle string `json:"lifecycle"`
			Outcome   string `json:"outcome"`
			Surface   string `json:"surface"`
			Backend   string `json:"backend"`
			Provider  string `json:"provider"`
		} `json:"run"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if !result.OK || result.Output != "fake provider output\n" || result.Run.Lifecycle != "terminal" || result.Run.Outcome != "succeeded" ||
		result.Run.Surface != "codex.raw" || result.Run.Backend != "codex" || result.Run.Provider != "unknown" {
		t.Fatalf("unexpected result: %#v", result)
	}
	if result.ArtifactPath == "" || result.Run.RunID == "" {
		t.Fatalf("durable identity missing: %#v", result)
	}
	var inspectOut, inspectErr bytes.Buffer
	if exit := Main([]string{"runs", "inspect", result.Run.RunID, "--json"}, bytes.NewReader(nil), &inspectOut, &inspectErr, true); exit != 0 {
		t.Fatalf("inspect exit=%d stdout=%s stderr=%s", exit, inspectOut.String(), inspectErr.String())
	}
	var inspected map[string]any
	if err := json.Unmarshal(inspectOut.Bytes(), &inspected); err != nil || inspected["kind"] != "hatch_run" || inspected["manifest"] == nil {
		t.Fatalf("inspect = %#v, %v", inspected, err)
	}
}

func TestMainSelectsOMPForSurfacedCodexRun(t *testing.T) {
	root := t.TempDir()
	fake := buildTestProviderForCLI(t, root)
	if err := os.Symlink(fake, filepath.Join(root, "omp")); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", root+string(os.PathListSeparator)+os.Getenv("PATH"))
	t.Setenv("OPENAI_API_KEY", "test-key")
	t.Setenv("HATCH_TEST_SCENARIO", "success_omp")
	t.Setenv("HATCH_TEST_SESSION", "omp-session-cli")
	t.Setenv("HATCH_RUN_ARTIFACT_ROOT", filepath.Join(root, "runs"))

	var stdout, stderr bytes.Buffer
	exitCode := Main([]string{"codex", "astra", "--harness", "omp", "--json", "prompt"}, bytes.NewReader(nil), &stdout, &stderr, true)
	if exitCode != 0 {
		t.Fatalf("exit=%d stderr=%s stdout=%s", exitCode, stderr.String(), stdout.String())
	}
	var result struct {
		OK     bool   `json:"ok"`
		Output string `json:"output"`
		Run    struct {
			Surface string `json:"surface"`
			Backend string `json:"backend"`
			Model   string `json:"model"`
		} `json:"run"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if !result.OK || result.Output != "fake success_omp output" || result.Run.Surface != "codex.astra" || result.Run.Backend != "omp" || result.Run.Model != "openai/gpt-6-astra" {
		t.Fatalf("unexpected OMP result: %#v", result)
	}
	for _, test := range []struct {
		args           []string
		surface, model string
	}{
		{[]string{"sol", "--harness", "omp", "--json", "prompt"}, "bedrock.sol", "amazon-bedrock/global.openai.gpt-6.1-sol"},
		{[]string{"terra", "--harness", "omp", "--json", "prompt"}, "bedrock.terra", "amazon-bedrock/global.openai.gpt-6.1-terra"},
		{[]string{"luna", "--harness", "omp", "--json", "prompt"}, "bedrock.luna", "amazon-bedrock/global.openai.gpt-6-luna"},
		{[]string{"codex", "sol", "--harness", "omp", "--json", "prompt"}, "codex.sol", "openai/gpt-6.1-sol"},
	} {
		stdout.Reset()
		stderr.Reset()
		if exit := Main(test.args, bytes.NewReader(nil), &stdout, &stderr, true); exit != 0 {
			t.Fatalf("%v exit=%d stdout=%s stderr=%s", test.args, exit, stdout.String(), stderr.String())
		}
		if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
			t.Fatal(err)
		}
		if !result.OK || result.Run.Surface != test.surface || result.Run.Backend != "omp" || result.Run.Model != test.model {
			t.Fatalf("unexpected GPT-6 result for %v: %#v", test.args, result)
		}
	}
}

func TestMainSelectsAntigravityForSurfacedGeminiRun(t *testing.T) {
	root := t.TempDir()
	fake := buildTestProviderForCLI(t, root)
	if err := os.Symlink(fake, filepath.Join(root, "agy")); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", root+string(os.PathListSeparator)+os.Getenv("PATH"))
	t.Setenv("HATCH_TEST_SCENARIO", "success_antigravity")
	t.Setenv("HATCH_RUN_ARTIFACT_ROOT", filepath.Join(root, "runs"))

	var stdout, stderr bytes.Buffer
	exitCode := Main([]string{"gemini", "--json", "prompt"}, bytes.NewReader(nil), &stdout, &stderr, true)
	if exitCode != 0 {
		t.Fatalf("exit=%d stderr=%s stdout=%s", exitCode, stderr.String(), stdout.String())
	}
	var result struct {
		OK     bool   `json:"ok"`
		Output string `json:"output"`
		Run    struct {
			Surface string `json:"surface"`
			Backend string `json:"backend"`
			Model   string `json:"model"`
		} `json:"run"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if !result.OK || result.Output != "fake antigravity output\n" || result.Run.Surface != "gemini.carbon" ||
		result.Run.Backend != "antigravity" || result.Run.Model != "gemini-4-carbon-medium" {
		t.Fatalf("unexpected Antigravity Gemini result: %#v", result)
	}
}

func buildTestProviderForCLI(t *testing.T, root string) string {
	t.Helper()
	fake := filepath.Join(root, "testprovider")
	command := exec.Command("go", "build", "-o", fake, "./internal/testprovider")
	command.Dir = filepath.Join("..", "..")
	if output, err := command.CombinedOutput(); err != nil {
		t.Fatalf("build test provider: %v\n%s", err, output)
	}
	return fake
}

func TestIdentityUsesStableSurfaceAliases(t *testing.T) {
	for _, test := range []struct{ model, backend, want string }{
		{"openai/gpt-6-astra", "pi", "codex.astra"},
		{"openai/gpt-5.4-nano", "pi", "codex.nano"},
		{"kimi-k3", "cursor", "cursor.kimi-k3"},
		{"gemini-4-carbon-medium", "cursor", "cursor.gemini-4-carbon"},
		{"gemini-4-carbon-medium", "antigravity", "gemini.carbon"},
	} {
		if got, _ := identity(test.backend, test.model); got != test.want {
			t.Fatalf("identity(%q)=%q want=%q", test.model, got, test.want)
		}
	}
}

func TestMainExpertJSON(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		if request.Method != http.MethodPost {
			t.Fatalf("method = %s", request.Method)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id": "resp_cli", "status": "completed", "model": "gpt-resolved",
			"output": []any{map[string]any{"type": "message", "content": []any{map[string]any{"type": "output_text", "text": "expert answer"}}}},
		})
	}))
	defer server.Close()
	t.Setenv("OPENAI_API_KEY", "test-key")
	t.Setenv("HATCH_EXPERT_RESPONSES_URL", server.URL)
	t.Setenv("HATCH_RUN_ARTIFACT_ROOT", filepath.Join(t.TempDir(), "runs"))
	var stdout, stderr bytes.Buffer
	exit := Main([]string{"expert", "--json", "--no-web-search", "question"}, bytes.NewReader(nil), &stdout, &stderr, true)
	if exit != 0 {
		t.Fatalf("exit=%d stdout=%s stderr=%s", exit, stdout.String(), stderr.String())
	}
	var result map[string]any
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	run, _ := result["run"].(map[string]any)
	policy, _ := result["reasoning_policy"].(map[string]any)
	if result["ok"] != true || result["output"] != "expert answer" || result["artifact_path"] == nil ||
		run["surface"] != "expert" || run["backend"] != "responses" || run["provider"] != "openai" ||
		policy["effort"] != "medium" || policy["source"] != "default" || policy["support"] != "native" {
		t.Fatalf("result = %#v", result)
	}
}

func TestMainFailsClosedBeforeProviderWhenArtifactRootUnavailable(t *testing.T) {
	root := filepath.Join(t.TempDir(), "root-file")
	if err := os.WriteFile(root, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("HATCH_RUN_ARTIFACT_ROOT", root)
	var stdout, stderr bytes.Buffer
	exitCode := Main([]string{"-b", "codex", "--api-key", "fake-test-key", "--json", "prompt"}, bytes.NewReader(nil), &stdout, &stderr, true)
	if exitCode == 0 {
		t.Fatalf("unexpected success: %s", stdout.String())
	}
	var result map[string]any
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if result["ok"] != false {
		t.Fatalf("unexpected result: %#v", result)
	}
}

func TestMainDoctorJSON(t *testing.T) {
	directory := t.TempDir()
	cursorBinary := filepath.Join(directory, "cursor-agent")
	if err := os.WriteFile(cursorBinary, []byte("#!/bin/sh\nprintf '%s\\n' 'grok-4.7-high - Grok' 'kimi-k3 - Kimi K3' 'gemini-4-carbon-medium - Gemini 4 Carbon Medium'\n"), 0o700); err != nil {
		t.Fatal(err)
	}
	piBinary := filepath.Join(directory, "pi")
	piScript := `#!/bin/sh
if [ "$1" = "--version" ]; then printf '%s\n' 'pi test'; exit 0; fi
printf '%s\n' 'provider model context' \
 'amazon-bedrock global.anthropic.claude-opus-5-5 1M' 'amazon-bedrock global.anthropic.claude-sonnet-5-5 1M' \
 'amazon-bedrock global.openai.gpt-6-astra 1M' 'amazon-bedrock global.openai.gpt-6-luna 1M' \
 'openai gpt-6-astra 272K' 'openai gpt-6-luna 272K' 'openai gpt-5.4-nano 272K' 'openai gpt-5.4-mini 272K' 'openai gpt-5.5 272K' \
 'openrouter z-ai/glm-5.3-flash 1M'
`
	if err := os.WriteFile(piBinary, []byte(piScript), 0o700); err != nil {
		t.Fatal(err)
	}
	ompBinary := filepath.Join(directory, "omp")
	if err := os.WriteFile(ompBinary, []byte("#!/bin/sh\nprintf '%s\\n' 'omp test'\n"), 0o700); err != nil {
		t.Fatal(err)
	}
	agyBinary := filepath.Join(directory, "agy")
	agyScript := `#!/bin/sh
if [ "$1" = "--version" ]; then printf '%s\n' 'agy test'; exit 0; fi
printf 'gemini-4-carbon-medium\tGemini 4 Carbon (Medium)\n'
`
	if err := os.WriteFile(agyBinary, []byte(agyScript), 0o700); err != nil {
		t.Fatal(err)
	}
	helper := filepath.Join(directory, "credential-helper")
	if err := os.WriteFile(helper, []byte("#!/bin/sh\nprintf 'test-secret\\n'\n"), 0o700); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", directory)
	t.Setenv("CURSOR_API_KEY", "")
	t.Setenv(credentialHelperEnv, helper)
	var stdout, stderr bytes.Buffer
	if exit := Main([]string{"doctor", "--json"}, bytes.NewReader(nil), &stdout, &stderr, true); exit != 0 {
		t.Fatalf("exit=%d stdout=%s stderr=%s", exit, stdout.String(), stderr.String())
	}
	var result struct {
		OK     bool `json:"ok"`
		Checks []struct {
			Name string `json:"name"`
			OK   bool   `json:"ok"`
		} `json:"checks"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if !result.OK || len(result.Checks) != 9 {
		t.Fatalf("doctor = %#v", result)
	}
	for _, check := range result.Checks {
		if !check.OK {
			t.Fatalf("doctor = %#v", result)
		}
	}
}

func TestMainCatalogJSON(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if exit := Main([]string{"catalog", "--json"}, bytes.NewReader(nil), &stdout, &stderr, true); exit != 0 {
		t.Fatalf("exit=%d stdout=%s stderr=%s", exit, stdout.String(), stderr.String())
	}
	var catalog []provider.CatalogEntry
	if err := json.Unmarshal(stdout.Bytes(), &catalog); err != nil {
		t.Fatal(err)
	}
	if !slices.Contains(catalog, provider.CatalogEntry{Surface: "claude", Alias: "opus", Model: "claude-opus-5-5"}) ||
		!slices.Contains(catalog, provider.CatalogEntry{Surface: "claude", Alias: "fable", Model: "claude-fable-5-2"}) ||
		!slices.Contains(catalog, provider.CatalogEntry{Surface: "claude", Alias: "fable-5.2", Model: "claude-fable-5-2"}) ||
		!slices.Contains(catalog, provider.CatalogEntry{Surface: "cursor", Alias: "gemini-4-carbon", Model: "gemini-4-carbon-medium"}) ||
		!slices.Contains(catalog, provider.CatalogEntry{Surface: "gemini", Alias: "carbon", Model: "gemini-4-carbon-medium"}) ||
		!slices.Contains(catalog, provider.CatalogEntry{Surface: "codex", Alias: "sol", Model: "openai/gpt-6.1-sol"}) ||
		!slices.Contains(catalog, provider.CatalogEntry{Surface: "codex", Alias: "luna", Model: "openai/gpt-6-luna"}) ||
		!slices.Contains(catalog, provider.CatalogEntry{Surface: "openrouter", Alias: "deepseek", Model: "openrouter/deepseek/deepseek-v4.1-flash-1017"}) ||
		!slices.Contains(catalog, provider.CatalogEntry{Surface: "openrouter", Alias: "glm-5.3-flash", Model: "openrouter/z-ai/glm-5.3-flash"}) {
		t.Fatalf("catalog = %#v", catalog)
	}
	geminiModels := 0
	for _, entry := range catalog {
		if entry.Surface == "gemini" {
			geminiModels++
		}
	}
	if geminiModels != 1 {
		t.Fatalf("catalog contains %d Gemini models, want one: %#v", geminiModels, catalog)
	}
}

func TestMainAdvancedHelpSeparatesRawFlags(t *testing.T) {
	var normal, advanced, stderr bytes.Buffer
	if exit := Main([]string{"--help"}, bytes.NewReader(nil), &normal, &stderr, true); exit != 0 {
		t.Fatal(exit)
	}
	if exit := Main([]string{"--advanced-help"}, bytes.NewReader(nil), &advanced, &stderr, true); exit != 0 {
		t.Fatal(exit)
	}
	if bytes.Contains(normal.Bytes(), []byte("--api-key")) || !bytes.Contains(normal.Bytes(), []byte("--reasoning-effort")) ||
		!bytes.Contains(advanced.Bytes(), []byte("--api-key")) || !bytes.Contains(advanced.Bytes(), []byte("--automation")) {
		t.Fatalf("normal=%s\nadvanced=%s", normal.String(), advanced.String())
	}
}

func TestMainRejectsMaxEffortForMaxTierAlias(t *testing.T) {
	var stdout, stderr bytes.Buffer
	exit := Main([]string{"codex", "max", "--reasoning-effort", "max", "--json", "prompt"}, bytes.NewReader(nil), &stdout, &stderr, true)
	if exit != 4 || !bytes.Contains(stdout.Bytes(), []byte("not supported")) {
		t.Fatalf("exit=%d stdout=%s stderr=%s", exit, stdout.String(), stderr.String())
	}
}
