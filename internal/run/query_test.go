package run

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestQueryRecordsScopesAndCursorAreStable(t *testing.T) {
	root := t.TempDir()
	expert := filepath.Join(t.TempDir(), "expert")
	if err := os.MkdirAll(expert, 0o700); err != nil {
		t.Fatal(err)
	}
	caller := filepath.Join(t.TempDir(), "repo")
	for i, id := range []string{"run-a", "run-b", "run-c"} {
		writeQueryManifest(t, root, id, caller, "session-a", time.Date(2026, 1, 1, 0, i, 0, 0, time.UTC))
	}
	first, err := QueryRecords(root, expert, QueryOptions{All: true, Limit: 1})
	if err != nil || len(first.Runs) != 1 || first.NextCursor == "" {
		t.Fatalf("first=%#v err=%v", first, err)
	}
	writeQueryManifest(t, root, "run-new", caller, "session-a", time.Date(2027, 1, 1, 0, 0, 0, 0, time.UTC))
	next, err := QueryRecords(root, expert, QueryOptions{All: true, Limit: 1, Before: first.NextCursor})
	if err != nil || len(next.Runs) != 1 || next.Runs[0].RunID != "run-b" {
		t.Fatalf("next=%#v err=%v", next, err)
	}
}

func TestQueryRecordsUnderUsesPathBoundariesAndSession(t *testing.T) {
	root := t.TempDir()
	expert := filepath.Join(t.TempDir(), "expert")
	inside := filepath.Join(t.TempDir(), "repo")
	outside := filepath.Join(filepath.Dir(inside), "repo-other")
	writeQueryManifest(t, root, "inside", inside, "session-a", time.Now().UTC())
	writeQueryManifest(t, root, "outside", outside, "session-b", time.Now().UTC())
	page, err := QueryRecords(root, expert, QueryOptions{Under: inside, Limit: 10})
	if err != nil || len(page.Runs) != 1 || page.Runs[0].RunID != "inside" {
		t.Fatalf("page=%#v err=%v", page, err)
	}
	page, err = QueryRecords(root, expert, QueryOptions{Session: "session-a", Limit: 10})
	if err != nil || len(page.Runs) != 1 || page.Runs[0].RunID != "inside" {
		t.Fatalf("session page=%#v err=%v", page, err)
	}
}

func TestQueryRecordsAllStillFiltersCallerKind(t *testing.T) {
	root := t.TempDir()
	expert := filepath.Join(t.TempDir(), "expert")
	writeQueryManifestKind(t, root, "cli-run", filepath.Join(t.TempDir(), "cli"), "cli", "cli", time.Now().UTC())
	writeQueryManifestKind(t, root, "longhouse-run", filepath.Join(t.TempDir(), "longhouse"), "longhouse", "longhouse", time.Now().UTC())

	page, err := QueryRecords(root, expert, QueryOptions{All: true, CallerKind: "longhouse", Limit: 10})
	if err != nil || len(page.Runs) != 1 || page.Runs[0].RunID != "longhouse-run" {
		t.Fatalf("page=%#v err=%v", page, err)
	}
	if page.Scope.Kind != "all" || page.Scope.CallerKind != "longhouse" {
		t.Fatalf("scope=%#v", page.Scope)
	}
}

func TestQueryRecordsScopeCanonicalizesSymlinkEquivalentPaths(t *testing.T) {
	root := t.TempDir()
	expert := filepath.Join(t.TempDir(), "expert")
	real := filepath.Join(t.TempDir(), "real-repo")
	if err := os.MkdirAll(real, 0o700); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(t.TempDir(), "linked-repo")
	if err := os.Symlink(real, link); err != nil {
		t.Fatal(err)
	}
	writeQueryManifest(t, root, "symlinked", link, "session", time.Now().UTC())

	page, err := QueryRecords(root, expert, QueryOptions{CallerCWD: real, Limit: 10})
	if err != nil || len(page.Runs) != 1 || page.Runs[0].RunID != "symlinked" {
		t.Fatalf("page=%#v err=%v", page, err)
	}
	if page.Scope.Value != normalizePath(real) {
		t.Fatalf("scope=%#v want=%q", page.Scope, normalizePath(real))
	}
}

func TestQueryRecordsNumericExpertTimesSortAndFilter(t *testing.T) {
	root := t.TempDir()
	expert := filepath.Join(t.TempDir(), "expert")
	if err := os.MkdirAll(expert, 0o700); err != nil {
		t.Fatal(err)
	}
	writeJSONFile(t, filepath.Join(expert, "old.json"), map[string]any{
		"response_id": "expert-old", "status": "completed", "updated_at": json.Number("1600000000"),
	})
	writeJSONFile(t, filepath.Join(expert, "new.json"), map[string]any{
		"response_id": "expert-new", "status": "completed", "updated_at": json.Number("1700000000"),
	})

	page, err := QueryRecords(root, expert, QueryOptions{All: true, Limit: 10})
	if err != nil || len(page.Runs) != 2 || page.Runs[0].RunID != "expert-new" || page.Runs[1].RunID != "expert-old" {
		t.Fatalf("sorted page=%#v err=%v", page, err)
	}
	filtered, err := QueryRecords(root, expert, QueryOptions{All: true, Since: time.Unix(1650000000, 0).UTC(), Limit: 10})
	if err != nil || len(filtered.Runs) != 1 || filtered.Runs[0].RunID != "expert-new" {
		t.Fatalf("filtered page=%#v err=%v", filtered, err)
	}
}

func TestQueryRecordsPreviewReportsRequestTruncationAfterWhitespace(t *testing.T) {
	root := t.TempDir()
	expert := filepath.Join(t.TempDir(), "expert")
	writeQueryManifest(t, root, "long-request", filepath.Join(t.TempDir(), "repo"), "session", time.Now().UTC())
	request := strings.Repeat("x", 160) + " \nremaining request text"
	if err := os.WriteFile(filepath.Join(root, "long-request", "request.txt"), []byte(request), 0o600); err != nil {
		t.Fatal(err)
	}

	page, err := QueryRecords(root, expert, QueryOptions{All: true, Limit: 10})
	if err != nil || len(page.Runs) != 1 {
		t.Fatalf("page=%#v err=%v", page, err)
	}
	if !page.Runs[0].PreviewTruncated || page.Runs[0].Preview != strings.Repeat("x", 160) {
		t.Fatalf("run=%#v", page.Runs[0])
	}
}

func writeQueryManifest(t *testing.T, root, id, cwd, session string, created time.Time) {
	writeQueryManifestKind(t, root, id, cwd, session, "cli", created)
}

func writeQueryManifestKind(t *testing.T, root, id, cwd, session, callerKind string, created time.Time) {
	t.Helper()
	dir := filepath.Join(root, id)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		t.Fatal(err)
	}
	manifest := map[string]any{
		"schema_version": 1, "run_id": id, "created_at": created.Format(time.RFC3339Nano),
		"updated_at": created.Format(time.RFC3339Nano), "lifecycle": "terminal", "outcome": "succeeded",
		"surface": "test", "backend": "test", "provider": "test", "model": "test", "cwd": cwd,
		"capture": map[string]any{"state": "durable"}, "result": map[string]any{"output": "present", "output_bytes": 3},
		"provider_state": map[string]any{"native_id_state": "unknown", "retention": "unknown"},
		"provenance":     map[string]any{"caller_cwd": cwd, "caller_kind": callerKind, "caller_session_id": session},
	}
	data, err := json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "manifest.json"), data, 0o600); err != nil {
		t.Fatal(err)
	}
}
