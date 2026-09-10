package run

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"unicode/utf8"
)

func TestReadContentPagesLargeSingleLineAndUTF8Boundaries(t *testing.T) {
	root := t.TempDir()
	runDir := filepath.Join(root, "hatch_page")
	if err := os.Mkdir(runDir, 0o700); err != nil {
		t.Fatal(err)
	}
	data := []byte(strings.Repeat("ab😀", 5000))
	if err := os.WriteFile(filepath.Join(runDir, "stdout.log"), data, 0o600); err != nil {
		t.Fatal(err)
	}
	writeContentManifest(t, runDir, map[string]any{"stdout_file": "stdout.log"})
	var got []byte
	var offset int64
	for {
		page, err := ReadContent(root, t.TempDir(), "hatch_page", ContentOptions{Part: "stdout", Offset: offset, Limit: 17})
		if err != nil {
			t.Fatal(err)
		}
		if !isValidUTF8(page.Content) {
			t.Fatalf("page is not valid UTF-8: %q", page.Content)
		}
		got = append(got, []byte(page.Content)...)
		if page.NextOffset == nil {
			break
		}
		offset = *page.NextOffset
	}
	if string(got) != string(data) {
		t.Fatalf("paged content changed: got %d bytes, want %d", len(got), len(data))
	}
	if offset == 0 {
		t.Fatal("paging did not advance")
	}
}

func TestReadContentInvalidBytesAndMidRuneOffset(t *testing.T) {
	root := t.TempDir()
	runDir := filepath.Join(root, "hatch_invalid")
	if err := os.Mkdir(runDir, 0o700); err != nil {
		t.Fatal(err)
	}
	data := []byte{'a', 0xff, 0xe2, 0x82, 0xac, 'z'}
	if err := os.WriteFile(filepath.Join(runDir, "stdout.log"), data, 0o600); err != nil {
		t.Fatal(err)
	}
	writeContentManifest(t, runDir, map[string]any{"stdout_file": "stdout.log"})
	page, err := ReadContent(root, t.TempDir(), "hatch_invalid", ContentOptions{Part: "stdout", Limit: 32})
	if err != nil {
		t.Fatal(err)
	}
	if !page.ReplacedInvalidUTF8 || page.Content != "a�€z" || page.Length != int64(len(data)) {
		t.Fatalf("invalid page = %#v", page)
	}
	mid, err := ReadContent(root, t.TempDir(), "hatch_invalid", ContentOptions{Part: "stdout", Offset: 3, Limit: 32})
	if err != nil {
		t.Fatal(err)
	}
	if !mid.OffsetAdjusted || mid.Content != "z" || mid.Offset != 5 {
		t.Fatalf("mid-rune page = %#v", mid)
	}
	stray, err := ReadContent(root, t.TempDir(), "hatch_invalid", ContentOptions{Part: "stdout", Offset: 1, Limit: 4})
	if err != nil {
		t.Fatal(err)
	}
	if !stray.ReplacedInvalidUTF8 || stray.Content != "�€" {
		t.Fatalf("stray invalid page = %#v", stray)
	}
}

func TestReadContentRejectsSubFourByteLimit(t *testing.T) {
	root := t.TempDir()
	runDir := filepath.Join(root, "hatch_limit")
	if err := os.Mkdir(runDir, 0o700); err != nil {
		t.Fatal(err)
	}
	writeContentManifest(t, runDir, map[string]any{"stdout_file": "stdout.log"})
	if err := os.WriteFile(filepath.Join(runDir, "stdout.log"), []byte("😀"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := ReadContent(root, t.TempDir(), "hatch_limit", ContentOptions{Part: "stdout", Limit: 3}); err == nil {
		t.Fatal("sub-four-byte content limit was accepted")
	}
}

func TestReadContentRejectsTraversalAndSymlink(t *testing.T) {
	root := t.TempDir()
	runDir := filepath.Join(root, "hatch_safe")
	if err := os.Mkdir(runDir, 0o700); err != nil {
		t.Fatal(err)
	}
	writeContentManifest(t, runDir, map[string]any{"stdout_file": "../outside"})
	if _, err := ReadContent(root, t.TempDir(), "hatch_safe", ContentOptions{Part: "stdout"}); err == nil {
		t.Fatal("manifest traversal was accepted")
	}
	outside := filepath.Join(t.TempDir(), "outside")
	if err := os.WriteFile(outside, []byte("secret"), 0o600); err != nil {
		t.Fatal(err)
	}
	writeContentManifest(t, runDir, map[string]any{"stdout_file": "link"})
	if err := os.Symlink(outside, filepath.Join(runDir, "link")); err != nil {
		t.Fatal(err)
	}
	if _, err := ReadContent(root, t.TempDir(), "hatch_safe", ContentOptions{Part: "stdout"}); err == nil {
		t.Fatal("symlink content was accepted")
	}
}

func TestReadContentLegacyExpertExtractsOnlyOutputText(t *testing.T) {
	root := t.TempDir()
	cache := t.TempDir()
	path := filepath.Join(cache, "resp_legacy.json")
	writeJSONFile(t, path, map[string]any{
		"response_id": "resp_legacy",
		"status":      "completed",
		"response": map[string]any{
			"output": []any{map[string]any{"type": "message", "content": []any{map[string]any{"type": "output_text", "text": "answer"}}}},
			"secret": "must not be returned",
		},
	})
	page, err := ReadContent(root, cache, "resp_legacy", ContentOptions{Part: "result"})
	if err != nil {
		t.Fatal(err)
	}
	if page.Content != "answer" {
		t.Fatalf("legacy result = %q", page.Content)
	}
	manifest, err := ReadContent(root, cache, "resp_legacy", ContentOptions{Part: "manifest"})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(manifest.Content, "must not be returned") {
		t.Fatalf("legacy manifest unavailable: %q", manifest.Content)
	}
}

func TestInventoryBoundsAndSkipsSymlink(t *testing.T) {
	runDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(runDir, "a"), []byte("a"), 0o600); err != nil {
		t.Fatal(err)
	}
	nested := filepath.Join(runDir, "snapshot", "data")
	if err := os.MkdirAll(nested, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(nested, "b"), []byte("bb"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(filepath.Join(runDir, "a"), filepath.Join(runDir, "escape")); err != nil {
		t.Fatal(err)
	}
	page, err := Inventory(Record{Path: runDir}, 0, 1)

	if err != nil {
		t.Fatal(err)
	}
	if len(page.Files) != 1 || page.NextOffset == nil || !page.Truncated {
		t.Fatalf("inventory page = %#v", page)
	}
	next, err := Inventory(Record{Path: runDir}, *page.NextOffset, 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(next.Files) != 1 || next.Files[0].Name != filepath.ToSlash(filepath.Join("snapshot", "data", "b")) {
		t.Fatalf("inventory next = %#v", next)
	}
}
func TestReadContentLegacyExpertDirectPathDoesNotScanRegistry(t *testing.T) {
	cache := t.TempDir()
	root := filepath.Join(t.TempDir(), "not-a-registry")
	if err := os.WriteFile(root, []byte("not a directory"), 0o600); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(cache, "resp_direct.json")
	writeJSONFile(t, path, map[string]any{
		"response_id": "resp_direct",
		"status":      "completed",
		"response": map[string]any{
			"output": []any{map[string]any{"type": "message", "content": []any{map[string]any{"type": "output_text", "text": "direct answer"}}}},
		},
	})
	record, err := InspectRecord(root, cache, "resp_direct")
	if err != nil || record.Kind != "legacy_expert" || record.Legacy["response_id"] != "resp_direct" {
		t.Fatalf("direct legacy inspection = %#v, %v", record, err)
	}
	page, err := ReadContent(root, cache, "resp_direct", ContentOptions{Part: "result"})
	if err != nil {
		t.Fatal(err)
	}
	if page.Content != "direct answer" || page.Answer != "complete" {
		t.Fatalf("direct legacy result = %#v", page)
	}
}
func TestInventoryReportsCappedWalk(t *testing.T) {
	runDir := t.TempDir()
	for i := 0; i <= maxInventoryWalkEntries; i++ {
		name := filepath.Join(runDir, fmt.Sprintf("file-%05d", i))
		if err := os.WriteFile(name, []byte("x"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	page, err := Inventory(Record{Path: runDir}, maxInventoryWalkEntries-1, 10)
	if err != nil {
		t.Fatal(err)
	}
	if !page.WalkTruncated || len(page.Files) != 1 {
		t.Fatalf("capped inventory page = %#v", page)
	}
	last, err := Inventory(Record{Path: runDir}, maxInventoryWalkEntries, 10)
	if err != nil {
		t.Fatal(err)
	}
	if !last.WalkTruncated || len(last.Files) != 0 || last.NextOffset != nil {
		t.Fatalf("capped inventory lower bound = %#v", last)
	}
}

func TestClipTextBoundsUTF8AndInvalid(t *testing.T) {
	if got, truncated := ClipText("😀abc", 5, false); got != "😀a" || !truncated {
		t.Fatalf("head clip = %q, %v", got, truncated)
	}
	if got, truncated := ClipText("abc😀", 4, true); got != "😀" || !truncated {
		t.Fatalf("tail clip = %q, %v", got, truncated)
	}
	if got, truncated := ClipText(string([]byte{'x', 0xff}), 4, false); got != "x�" || truncated {
		t.Fatalf("invalid clip = %q, %v", got, truncated)
	}
	if got, truncated := ClipText("cancelled", 2048, true); got != "cancelled" || truncated {
		t.Fatalf("short tail = %q, %v", got, truncated)
	}
	if got, truncated := ClipText(string([]byte{0xff, 0xff, 'z'}), 4, true); got != "�z" || !truncated {
		t.Fatalf("invalid tail = %q, %v", got, truncated)
	}
}

func writeContentManifest(t *testing.T, runDir string, capture map[string]any) {
	t.Helper()
	manifest := map[string]any{
		"schema_version": 1,
		"run_id":         filepath.Base(runDir),
		"capture":        capture,
	}
	data, err := json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(runDir, "manifest.json"), data, 0o600); err != nil {
		t.Fatal(err)
	}
}

func isValidUTF8(value string) bool {
	for len(value) > 0 {
		_, size := utf8.DecodeRuneInString(value)
		if size == 1 && value[0] >= 0x80 {
			return false
		}
		value = value[size:]
	}
	return true
}
