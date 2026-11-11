package cli

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type followUpResult struct {
	OK       bool   `json:"ok"`
	Error    string `json:"error"`
	FollowUp string `json:"follow_up"`
	Run      struct {
		RunID   string `json:"run_id"`
		Surface string `json:"surface"`
		CWD     string `json:"cwd"`
	} `json:"run"`
}

func runHatch(t *testing.T, stdin string, args ...string) (followUpResult, int, string) {
	t.Helper()
	var stdout, stderr bytes.Buffer
	code := Main(args, bytes.NewBufferString(stdin), &stdout, &stderr, true)
	var result followUpResult
	if stdout.Len() > 0 {
		if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
			t.Fatalf("decode %q: %v", stdout.String(), err)
		}
	}
	return result, code, stderr.String()
}

func TestResumeContinuesAnEarlierRunOnAnyProvider(t *testing.T) {
	root := t.TempDir()
	fake := buildTestProviderForCLI(t, root)
	if err := os.Symlink(fake, filepath.Join(root, "codex")); err != nil {
		t.Fatal(err)
	}
	record := filepath.Join(root, "invocation.json")
	t.Setenv("PATH", root+string(os.PathListSeparator)+os.Getenv("PATH"))
	t.Setenv("HATCH_RUN_ARTIFACT_ROOT", filepath.Join(root, "runs"))
	t.Setenv("HATCH_TEST_SCENARIO", "success_text")
	t.Setenv("HATCH_TEST_RECORD", record)

	first, code, _ := runHatch(t, "", "-b", "codex", "--api-key", "fake-test-key", "--json", "-C", root, "Investigate the parking lot spin")
	if code != 0 || first.Run.RunID == "" {
		t.Fatalf("first run failed: code=%d %#v", code, first)
	}
	if !strings.Contains(first.FollowUp, "--resume "+first.Run.RunID) {
		t.Fatalf("result does not advertise the follow-up command: %q", first.FollowUp)
	}

	// No model, surface or directory: everything is inherited from the run.
	second, code, _ := runHatch(t, "", "--resume", first.Run.RunID, "--json", "Now fix it")
	if code != 0 || !second.OK {
		t.Fatalf("follow-up failed: code=%d %#v", code, second)
	}
	if second.Run.Surface != first.Run.Surface || second.Run.CWD != first.Run.CWD {
		t.Fatalf("follow-up did not inherit surface/cwd: %#v vs %#v", second.Run, first.Run)
	}
	data, err := os.ReadFile(record)
	if err != nil {
		t.Fatal(err)
	}
	var seen struct{ Stdin string }
	if err := json.Unmarshal(data, &seen); err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"Investigate the parking lot spin", "fake provider output", "New task:\nNow fix it"} {
		if !strings.Contains(seen.Stdin, want) {
			t.Fatalf("provider prompt missing %q:\n%s", want, seen.Stdin)
		}
	}

	// A chain of follow-ups stores only the new task as its request, so earlier
	// answers are not nested into every later run's artifact.
	var out bytes.Buffer
	if code := Main([]string{"runs", "read", second.Run.RunID, "--part", "request", "--json"}, bytes.NewReader(nil), &out, &bytes.Buffer{}, true); code != 0 {
		t.Fatalf("read request: %s", out.String())
	}
	var page struct{ Content string }
	if err := json.Unmarshal(out.Bytes(), &page); err != nil || page.Content != "Now fix it" {
		t.Fatalf("stored request = %q (%v)", page.Content, err)
	}
}

func TestResumeRefusesToCrossAccountsAndUnknownRuns(t *testing.T) {
	root := t.TempDir()
	fake := buildTestProviderForCLI(t, root)
	if err := os.Symlink(fake, filepath.Join(root, "codex")); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", root+string(os.PathListSeparator)+os.Getenv("PATH"))
	t.Setenv("HATCH_RUN_ARTIFACT_ROOT", filepath.Join(root, "runs"))
	t.Setenv("HATCH_TEST_SCENARIO", "success_text")

	first, code, _ := runHatch(t, "", "-b", "codex", "--api-key", "fake-test-key", "--json", "personal question")
	if code != 0 {
		t.Fatalf("first run failed: %#v", first)
	}
	result, code, _ := runHatch(t, "", "bedrock", "sonnet", "--resume", first.Run.RunID, "--json", "continue")
	if code != 4 || !strings.Contains(result.Error, "never cross") {
		t.Fatalf("personal run continued on a Zeta surface: code=%d %#v", code, result)
	}
	result, code, _ = runHatch(t, "", "--resume", "hatch_missing", "--json", "continue")
	if code != 4 || result.Error == "" {
		t.Fatalf("unknown run id accepted: code=%d %#v", code, result)
	}
}

func TestFlagsAreRejectedWhereTheBackendWouldIgnoreThem(t *testing.T) {
	for _, args := range [][]string{
		{"claude", "sonnet", "--skip-git-repo-check", "--json", "x"},
		{"cursor", "grok", "--include-partial-messages", "--json", "x"},
	} {
		result, code, _ := runHatch(t, "", args...)
		if code != 4 || !strings.Contains(result.Error, "only applies") {
			t.Fatalf("%v: code=%d %#v", args, code, result)
		}
	}
}
