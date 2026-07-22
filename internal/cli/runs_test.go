package cli

import (
	"bytes"
	"encoding/json"
	"path/filepath"
	"testing"

	runner "github.com/cipher982/hatch/internal/run"
)

func TestRunsCLIListsAndInspectsCurrentArtifacts(t *testing.T) {
	root := filepath.Join(t.TempDir(), "runs")
	t.Setenv("HATCH_RUN_ARTIFACT_ROOT", root)
	t.Setenv("XDG_CACHE_HOME", t.TempDir())
	store := runner.NewStore(root)
	artifact, err := store.Prepare(runner.PreparedRun{Surface: "test.surface", Provider: "test", Request: "prompt", RedactedArgv: []string{"provider"}})
	if err != nil {
		t.Fatal(err)
	}
	stdoutSink, stderrSink, err := store.OpenStreams(artifact)
	if err != nil {
		t.Fatal(err)
	}
	_ = stdoutSink.Close()
	_ = stderrSink.Close()
	resultFile, err := store.WriteResult(artifact, []byte("answer"))
	if err != nil {
		t.Fatal(err)
	}
	if err := store.CommitTerminal(artifact, runner.OutcomeSucceeded, 0, runner.Result{Output: "present", OutputFile: &resultFile, TerminalMarker: "not_applicable"}, runner.State{Retention: "unknown", NativeIDState: "not_exposed", Capabilities: map[string]string{}}, nil); err != nil {
		t.Fatal(err)
	}

	var listOut, listErr bytes.Buffer
	if exit := Main([]string{"runs", "list", "--status", "succeeded", "--json"}, bytes.NewReader(nil), &listOut, &listErr, true); exit != 0 {
		t.Fatalf("list exit=%d stderr=%s", exit, listErr.String())
	}
	var list struct {
		Runs []runner.Summary `json:"runs"`
	}
	if err := json.Unmarshal(listOut.Bytes(), &list); err != nil || len(list.Runs) != 1 || list.Runs[0].RunID != artifact.Manifest.RunID {
		t.Fatalf("list=%s err=%v", listOut.String(), err)
	}

	var inspectOut, inspectErr bytes.Buffer
	if exit := Main([]string{"runs", "inspect", artifact.Manifest.RunID, "--json"}, bytes.NewReader(nil), &inspectOut, &inspectErr, true); exit != 0 {
		t.Fatalf("inspect exit=%d stderr=%s", exit, inspectErr.String())
	}
	var record runner.Record
	if err := json.Unmarshal(inspectOut.Bytes(), &record); err != nil || record.Manifest == nil ||
		record.Manifest.RunID != artifact.Manifest.RunID || !containsString(record.Files, "evidence.sha256") ||
		!containsString(record.Files, "result.txt") {
		t.Fatalf("inspect=%s err=%v", inspectOut.String(), err)
	}
}

func TestRunsCLIAuditsArtifactIntegrityByDefault(t *testing.T) {
	root := filepath.Join(t.TempDir(), "runs")
	t.Setenv("HATCH_RUN_ARTIFACT_ROOT", root)
	var stdout, stderr bytes.Buffer
	exit := Main([]string{"runs", "audit", "--json"}, bytes.NewReader(nil), &stdout, &stderr, true)
	if exit != 0 {
		t.Fatalf("exit=%d stderr=%s", exit, stderr.String())
	}
	var result struct {
		Passed bool              `json:"passed"`
		Audit  runner.FieldAudit `json:"audit"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil || !result.Passed || result.Audit.Eligible != 0 {
		t.Fatalf("audit=%s err=%v", stdout.String(), err)
	}

	stdout.Reset()
	stderr.Reset()
	if exit := Main([]string{"runs", "audit", "--minimum-total", "1"}, bytes.NewReader(nil), &stdout, &stderr, true); exit != 1 || !bytes.Contains(stderr.Bytes(), []byte("sample minimum is not satisfied")) {
		t.Fatalf("unmet exit=%d stdout=%s stderr=%s", exit, stdout.String(), stderr.String())
	}
}

func containsString(values []string, wanted string) bool {
	for _, value := range values {
		if value == wanted {
			return true
		}
	}
	return false
}
