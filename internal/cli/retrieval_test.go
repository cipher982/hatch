package cli

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/cipher982/hatch/internal/expert"
	runner "github.com/cipher982/hatch/internal/run"
)

func TestCLILargeResultAndFailureKeepFullEvidence(t *testing.T) {
	root := t.TempDir()
	fake := buildTestProviderForCLI(t, root)
	if err := os.Symlink(fake, filepath.Join(root, "gemini")); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", root+string(os.PathListSeparator)+os.Getenv("PATH"))
	t.Setenv("HATCH_RUN_ARTIFACT_ROOT", filepath.Join(root, "runs"))
	for _, scenario := range []string{"large_answer", "large_failure"} {
		t.Run(scenario, func(t *testing.T) {
			t.Setenv("HATCH_TEST_SCENARIO", scenario)
			var stdout, stderr bytes.Buffer
			exit := Main([]string{"-b", "gemini", "--title", "Review oversized provider output safely", "--caller-session", "regression:large", "-"}, strings.NewReader("request"), &stdout, &stderr, false)
			if (scenario == "large_answer") != (exit == 0) {
				t.Fatalf("scenario %s exit=%d", scenario, exit)
			}
			if stdout.Len() > maxResponseBytes || stderr.Len() > 20<<10 || !json.Valid(stdout.Bytes()) || !utf8.Valid(stdout.Bytes()) {
				t.Fatalf("unsafe response: stdout=%d stderr=%d", stdout.Len(), stderr.Len())
			}
			var result resultDisplay
			if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
				t.Fatal(err)
			}
			if result.Run == nil {
				t.Fatal("run identity lost")
			}
			runDir := filepath.Join(root, "runs", result.Run.RunID)
			var durable runner.PublicResult
			data, err := os.ReadFile(filepath.Join(runDir, "result.json"))
			if err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal(data, &durable); err != nil {
				t.Fatal(err)
			}
			if scenario == "large_answer" {
				want := strings.Repeat("€<&\n", 1<<15)
				answer, err := os.ReadFile(filepath.Join(runDir, "result.txt"))
				if err != nil {
					t.Fatal(err)
				}
				if string(answer) != want || durable.Output != want {
					t.Fatal("display truncation changed durable output")
				}
				if !result.OutputTruncated || !strings.HasPrefix(want, result.Output) || len(result.Output) >= len(want) {
					t.Fatal("missing or invalid answer preview")
				}
				if result.Run.Title != "Review oversized provider output safely" || result.Run.Provenance.CallerSessionID != "regression:large" {
					t.Fatal("title or caller identity absent from launch result")
				}
			} else {
				if result.OK || !result.ErrorTruncated || !result.StderrTruncated {
					t.Fatal("chatty failure was not bounded and marked failed")
				}
				if durable.Error == nil || len(*durable.Error) <= len(*result.Error) || durable.Stderr == nil || len(*durable.Stderr) <= len(*result.Stderr) {
					t.Fatal("full failure diagnostics were lost")
				}
			}
		})
	}
}

func TestCLIScopedRecoveryAndExecutableContinuation(t *testing.T) {
	root := t.TempDir()
	fake := buildTestProviderForCLI(t, root)
	if err := os.Symlink(fake, filepath.Join(root, "gemini")); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", root+string(os.PathListSeparator)+os.Getenv("PATH"))
	t.Setenv("HATCH_RUN_ARTIFACT_ROOT", filepath.Join(root, "runs"))
	t.Setenv("XDG_CACHE_HOME", filepath.Join(root, "cache"))
	t.Setenv("HATCH_TEST_SCENARIO", "success_text")
	t.Setenv("HATCH_CALLER_SESSION_ID", "conversation:A")
	target := filepath.Join(root, "other-target")
	if err := os.Mkdir(target, 0700); err != nil {
		t.Fatal(err)
	}
	launch := func(session string) string {
		t.Helper()
		var out, diag bytes.Buffer
		if exit := Main([]string{"-b", "gemini", "--title", "Review provider factory release readiness", "--caller-session", session, "--caller-request", "review-pair", "-C", target, "request"}, bytes.NewReader(nil), &out, &diag, false); exit != 0 {
			t.Fatalf("launch exit=%d %s", exit, diag.String())
		}
		var result runner.PublicResult
		if err := json.Unmarshal(out.Bytes(), &result); err != nil {
			t.Fatal(err)
		}
		if result.Run.CWD == result.Run.Provenance.CallerCWD {
			t.Fatal("target override overwrote caller cwd")
		}
		return result.Run.RunID
	}
	older := launch("conversation:A")
	newer := launch("conversation:A")
	launch("conversation:B")
	list := func(args []string) (runner.RunPage, []string) {
		t.Helper()
		var out, diag bytes.Buffer
		if exit := Main(args, bytes.NewReader(nil), &out, &diag, false); exit != 0 {
			t.Fatalf("list exit=%d %s %s", exit, out.String(), diag.String())
		}
		var value struct {
			runner.RunPage
			NextCommand []string `json:"next_command"`
		}
		if err := json.Unmarshal(out.Bytes(), &value); err != nil {
			t.Fatal(err)
		}
		return value.RunPage, value.NextCommand
	}
	page, next := list([]string{"runs", "list", "--request", "review-pair", "--limit", "1"})
	if len(page.Runs) != 1 || page.Runs[0].RunID != newer || len(next) == 0 {
		t.Fatalf("wrong first page: %#v", page)
	}
	launch("conversation:A")
	page, _ = list(next[1:])
	if len(page.Runs) != 1 || page.Runs[0].RunID != older {
		t.Fatalf("new arrivals shifted continuation: %#v", page)
	}
	for _, name := range []string{"HATCH_CALLER_SESSION_ID", "LONGHOUSE_MANAGED_SESSION_ID", "LONGHOUSE_SESSION_ID", "LONGHOUSE_CHANNEL_SESSION_ID"} {
		t.Setenv(name, "")
	}
	page, next = list([]string{"runs", "list", "--limit", "1"})
	if page.Scope.Kind != "cwd" || len(next) == 0 {
		t.Fatalf("missing exact caller cwd scope: %#v", page)
	}
	list(next[1:]) // The emitted command must remain valid without the original environment.
}

func TestCLIConfigErrorsAreBounded(t *testing.T) {
	var out, diag bytes.Buffer
	exit := Main([]string{"runs", "list", "--" + strings.Repeat("x", 1<<20)}, bytes.NewReader(nil), &out, &diag, false)
	if exit == 0 || !json.Valid(out.Bytes()) || out.Len() > maxResponseBytes {
		t.Fatalf("unsafe config error, exit=%d bytes=%d", exit, out.Len())
	}
}

func TestExpertTextOutputRequiresExplicitJSON(t *testing.T) {
	var pipedOut, pipedErr bytes.Buffer
	if exit := runExpert(context.Background(), []string{"--max-output-bytes", "1", "question"}, bytes.NewReader(nil), &pipedOut, &pipedErr); exit == 0 {
		t.Fatal("invalid piped expert request unexpectedly succeeded")
	}
	if pipedOut.Len() != 0 || !strings.HasPrefix(pipedErr.String(), "Error:") {
		t.Fatalf("piped expert selected JSON: stdout=%q stderr=%q", pipedOut.String(), pipedErr.String())
	}
	t.Setenv("HATCH_EXPERT_MODEL", "")
	request, err := parseExpert([]string{"question"})
	if err != nil {
		t.Fatal(err)
	}
	if request.JSON || request.Model != "gpt-6-sol" {
		t.Fatalf("expert defaults = model:%q JSON:%t, want GPT-6 Sol and text output", request.Model, request.JSON)
	}
	var stdout, stderr bytes.Buffer
	if exit := renderExpertResult(expert.Result{OK: true, Output: "answer"}, 8192, request.JSON, &stdout, &stderr); exit != 0 {
		t.Fatalf("exit=%d stderr=%s", exit, stderr.String())
	}
	if stdout.String() != "answer\n" {
		t.Fatalf("text output = %q", stdout.String())
	}
}

func TestExpertJSONRetainsOrdinaryCitationMetadata(t *testing.T) {
	largeError := strings.Repeat("provider diagnostic ", 10000)
	result := expert.Result{
		OK: true, Output: "answer",
		Citations: []map[string]any{{"title": "A useful source", "url": "https://example.test/a"}},
		Sources:   []map[string]any{{"name": "example"}},
		Usage:     map[string]any{"input_tokens": float64(12), "output_tokens": float64(8)},
		Run:       &runner.Manifest{RunID: "expert-metadata", Result: runner.Result{Error: &largeError}},
	}
	var stdout, stderr bytes.Buffer
	if exit := renderExpertResult(result, 8192, true, &stdout, &stderr); exit != 0 {
		t.Fatalf("exit=%d stderr=%s", exit, stderr.String())
	}
	var payload struct {
		Citations []map[string]any `json:"citations"`
		Sources   []map[string]any `json:"sources"`
		Usage     map[string]any   `json:"usage"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &payload); err != nil {
		t.Fatal(err)
	}
	if len(payload.Citations) != 1 || payload.Citations[0]["title"] != "A useful source" ||
		len(payload.Sources) != 1 || payload.Usage["input_tokens"] != float64(12) {
		t.Fatalf("metadata was clipped: %#v", payload)
	}
}

func TestFailedTextInvocationDoesNotEmitPartialAnswer(t *testing.T) {
	message := "provider failed"
	result := runner.PublicResult{Status: "error", Output: "unfinished answer", Error: &message, ExitCode: 7}
	var stdout, stderr bytes.Buffer
	if exit := renderResult(result, 8192, false, &stdout, &stderr); exit == 0 {
		t.Fatal("failed run reported success")
	}
	if stdout.Len() != 0 || !strings.Contains(stderr.String(), message) {
		t.Fatalf("failed text output changed: stdout=%q stderr=%q", stdout.String(), stderr.String())
	}
}

func TestExpertOversizedMetadataDoesNotEvictOtherFields(t *testing.T) {
	result := expert.Result{
		OK: true, Output: "answer",
		Citations: []map[string]any{{"text": strings.Repeat("citation-", 3000)}},
		Sources:   []map[string]any{{"name": "retained source"}},
		Usage:     map[string]any{"total_tokens": float64(20)},
	}
	var stdout, stderr bytes.Buffer
	if exit := renderExpertResult(result, 8192, true, &stdout, &stderr); exit != 0 {
		t.Fatalf("exit=%d stderr=%s", exit, stderr.String())
	}
	var payload struct {
		Citations []map[string]any `json:"citations"`
		Sources   []map[string]any `json:"sources"`
		Usage     map[string]any   `json:"usage"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &payload); err != nil {
		t.Fatal(err)
	}
	if payload.Citations != nil || len(payload.Sources) != 1 || payload.Sources[0]["name"] != "retained source" ||
		payload.Usage["total_tokens"] != float64(20) {
		t.Fatalf("oversized citation evicted useful metadata: %#v", payload)
	}
}

func TestMinimalManifestPreservesBoundedRetrievalIdentity(t *testing.T) {
	manifest := &runner.Manifest{
		RunID:   "run_identity",
		Surface: strings.Repeat("expert-", 500),
		Model:   strings.Repeat("model-", 500),
		Title:   strings.Repeat("title-", 500),
		CWD:     strings.Repeat("/work/", 500),
		Provenance: &runner.Provenance{
			CallerCWD:       strings.Repeat("/caller/", 500),
			CallerKind:      strings.Repeat("kind-", 100),
			CallerSessionID: strings.Repeat("session-", 100),
		},
		Result: runner.Result{Output: strings.Repeat("answer-", 500)},
	}
	var stdout, stderr bytes.Buffer
	if exit := renderResult(runner.PublicResult{OK: true, Output: "answer", Run: manifest}, 8192, true, &stdout, &stderr); exit != 0 {
		t.Fatalf("render failed: %s", stderr.String())
	}
	var payload resultDisplay
	if err := json.Unmarshal(stdout.Bytes(), &payload); err != nil {
		t.Fatal(err)
	}
	if payload.Run == nil || payload.Run.RunID != "run_identity" || payload.Run.Provenance == nil ||
		!strings.HasPrefix(payload.Run.Title, "title-") || !strings.HasPrefix(payload.Run.Surface, "expert-") ||
		!strings.HasPrefix(payload.Run.Model, "model-") || !strings.HasPrefix(payload.Run.Provenance.CallerSessionID, "session-") {
		t.Fatalf("retrieval identity was lost: %#v", payload.Run)
	}
	if !payload.MetadataTruncated || stdout.Len() > maxResponseBytes {
		t.Fatalf("unsafe metadata response: bytes=%d truncated=%v", stdout.Len(), payload.MetadataTruncated)
	}
}
