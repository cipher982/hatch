package cli

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
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

func TestIdentityUsesStableSurfaceAliases(t *testing.T) {
	for model, want := range map[string]string{
		"openai/gpt-5.6-sol":            "codex.sol",
		"openai/gpt-5.4-nano":           "codex.nano",
		"openrouter/moonshotai/kimi-k3": "openrouter.kimi-k3",
	} {
		if got, _ := identity("opencode", model); got != want {
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
	if result["ok"] != true || result["output"] != "expert answer" || result["artifact_path"] == nil ||
		run["surface"] != "expert" || run["backend"] != "responses" || run["provider"] != "openai" {
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
	binary := filepath.Join(directory, "cursor-agent")
	if err := os.WriteFile(binary, []byte("#!/bin/sh\nprintf '%s\\n' 'cursor-grok-4.5-high - Grok'\n"), 0o700); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", directory)
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
	if !result.OK || len(result.Checks) != 1 || result.Checks[0].Name != "cursor.grok" || !result.Checks[0].OK {
		t.Fatalf("doctor = %#v", result)
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
	if bytes.Contains(normal.Bytes(), []byte("--api-key")) || !bytes.Contains(advanced.Bytes(), []byte("--api-key")) || !bytes.Contains(advanced.Bytes(), []byte("--automation")) {
		t.Fatalf("normal=%s\nadvanced=%s", normal.String(), advanced.String())
	}
}
