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

func TestMainRawGeminiVerticalSlice(t *testing.T) {
	root := t.TempDir()
	fake := filepath.Join(root, "testprovider")
	command := exec.Command("go", "build", "-o", fake, "./internal/testprovider")
	command.Dir = filepath.Join("..", "..")
	if output, err := command.CombinedOutput(); err != nil {
		t.Fatalf("build test provider: %v\n%s", err, output)
	}
	if err := os.Symlink(fake, filepath.Join(root, "gemini")); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", root+string(os.PathListSeparator)+os.Getenv("PATH"))
	t.Setenv("HATCH_RUN_ARTIFACT_ROOT", filepath.Join(root, "runs"))
	t.Setenv("HATCH_TEST_SCENARIO", "success_text")
	t.Setenv("HATCH_TEST_RECORD", filepath.Join(root, "invocation.json"))

	var stdout, stderr bytes.Buffer
	exitCode := Main([]string{"-b", "gemini", "--json", "-"}, bytes.NewBufferString("oracle prompt"), &stdout, &stderr, true)
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
		result.Run.Surface != "gemini.raw" || result.Run.Backend != "gemini" || result.Run.Provider != "google" {
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
	exitCode := Main([]string{"codex", "sol", "--harness", "omp", "--json", "prompt"}, bytes.NewReader(nil), &stdout, &stderr, true)
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
	if !result.OK || result.Output != "fake success_omp output" || result.Run.Surface != "codex.sol" || result.Run.Backend != "omp" || result.Run.Model != "openai/gpt-5.6-sol" {
		t.Fatalf("unexpected OMP result: %#v", result)
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
	for model, want := range map[string]string{
		"openai/gpt-5.6-sol":  "codex.sol",
		"openai/gpt-5.4-nano": "codex.nano",
		"kimi-k3":             "cursor.kimi-k3",
	} {
		backend := "opencode"
		if model == "kimi-k3" {
			backend = "cursor"
		}
		if got, _ := identity(backend, model); got != want {
			t.Fatalf("identity(%q)=%q want=%q", model, got, want)
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
	exitCode := Main([]string{"-b", "gemini", "--json", "prompt"}, bytes.NewReader(nil), &stdout, &stderr, true)
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
	if err := os.WriteFile(cursorBinary, []byte("#!/bin/sh\nprintf '%s\\n' 'cursor-grok-4.6-high - Grok' 'kimi-k3 - Kimi K3'\n"), 0o700); err != nil {
		t.Fatal(err)
	}
	opencodeBinary := filepath.Join(directory, "opencode")
	if err := os.WriteFile(opencodeBinary, []byte("#!/bin/sh\nif [ \"$1\" = \"--version\" ]; then printf '%s\\n' 'opencode test'; exit 0; fi\n[ \"$OPENAI_API_KEY\" = test-secret ] || [ \"$OPENROUTER_API_KEY\" = test-secret ] || exit 9\nprintf '%s\\n' 'openai/gpt-5.6-sol' 'openai/gpt-5.6-terra' 'openai/gpt-5.6-luna' 'openai/gpt-5.4-nano' 'openai/gpt-5.4-mini' 'openai/gpt-5.5' 'openrouter/deepseek/deepseek-v4-flash-0731' 'openrouter/deepseek/deepseek-v4-pro-0813' 'openrouter/z-ai/glm-5.3-flash'\n"), 0o700); err != nil {
		t.Fatal(err)
	}
	piBinary := filepath.Join(directory, "pi")
	if err := os.WriteFile(piBinary, []byte("#!/bin/sh\nprintf '%s\\n' 'pi test'\n"), 0o700); err != nil {
		t.Fatal(err)
	}
	ompBinary := filepath.Join(directory, "omp")
	ompScript := `#!/bin/sh
if [ "$1" = "--version" ]; then printf '%s\n' 'omp test'; exit 0; fi
printf '%s\n' '{"models":[{"id":"gemini-3.7-flash-tiered","selector":"google-antigravity/gemini-3.7-flash-tiered"}]}'
`
	if err := os.WriteFile(ompBinary, []byte(ompScript), 0o700); err != nil {
		t.Fatal(err)
	}
	helper := filepath.Join(directory, "credential-helper")
	if err := os.WriteFile(helper, []byte("#!/bin/sh\nprintf 'test-secret\\n'\n"), 0o700); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", directory)
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("OPENROUTER_API_KEY", "")
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
	if !result.OK || len(result.Checks) != 7 {
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
	if !slices.Contains(catalog, provider.CatalogEntry{Surface: "claude", Alias: "fable", Model: "claude-fable-5-1"}) ||
		!slices.Contains(catalog, provider.CatalogEntry{Surface: "claude", Alias: "fable-5.1", Model: "claude-fable-5-1"}) ||
		!slices.Contains(catalog, provider.CatalogEntry{Surface: "openrouter", Alias: "deepseek-v4-flash", Model: "openrouter/deepseek/deepseek-v4-flash-0731"}) ||
		!slices.Contains(catalog, provider.CatalogEntry{Surface: "openrouter", Alias: "deepseek-v4-pro", Model: "openrouter/deepseek/deepseek-v4-pro-0813"}) ||
		!slices.Contains(catalog, provider.CatalogEntry{Surface: "openrouter", Alias: "glm-5.3-flash", Model: "openrouter/z-ai/glm-5.3-flash"}) {
		t.Fatalf("catalog = %#v", catalog)
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
