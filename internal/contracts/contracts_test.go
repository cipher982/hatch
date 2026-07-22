package contracts

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

type ledger struct {
	SchemaVersion int `json:"schema_version"`
	Baseline      struct {
		CollectedTests int `json:"collected_tests"`
	} `json:"baseline"`
	Dispositions []string `json:"dispositions"`
	Tests        []struct {
		NodeID      string  `json:"node_id"`
		Disposition string  `json:"disposition"`
		Proof       string  `json:"proof"`
		Reason      *string `json:"reason"`
	} `json:"tests"`
}

func TestPythonTestLedger(t *testing.T) {
	var got ledger
	readJSON(t, filepath.Join(repoRoot(t), "testdata", "contracts", "python-test-ledger.json"), &got)
	if got.SchemaVersion != 1 {
		t.Fatalf("schema_version = %d, want 1", got.SchemaVersion)
	}
	if len(got.Tests) != got.Baseline.CollectedTests {
		t.Fatalf("ledger has %d tests, baseline says %d", len(got.Tests), got.Baseline.CollectedTests)
	}
	if got.Baseline.CollectedTests != 304 {
		t.Fatalf("baseline collected_tests = %d, want frozen baseline 304", got.Baseline.CollectedTests)
	}

	allowed := make(map[string]bool, len(got.Dispositions))
	for _, disposition := range got.Dispositions {
		allowed[disposition] = true
	}
	seen := make(map[string]bool, len(got.Tests))
	for _, test := range got.Tests {
		if test.NodeID == "" || seen[test.NodeID] {
			t.Fatalf("empty or duplicate node_id %q", test.NodeID)
		}
		seen[test.NodeID] = true
		if !allowed[test.Disposition] {
			t.Errorf("%s has invalid disposition %q", test.NodeID, test.Disposition)
		}
		if test.Proof == "" {
			t.Errorf("%s has no proof target", test.NodeID)
		}
		if test.Disposition == "intentional_change" && (test.Reason == nil || *test.Reason == "") {
			t.Errorf("%s has an intentional change without a reason", test.NodeID)
		}
	}
}

func TestContractCorpus(t *testing.T) {
	paths, err := filepath.Glob(filepath.Join(repoRoot(t), "testdata", "contracts", "cases", "*.json"))
	if err != nil {
		t.Fatal(err)
	}
	if len(paths) == 0 {
		t.Fatal("contract corpus is empty")
	}
	seen := make(map[string]bool, len(paths))
	for _, path := range paths {
		path := path
		t.Run(filepath.Base(path), func(t *testing.T) {
			var value struct {
				SchemaVersion int    `json:"schema_version"`
				Name          string `json:"name"`
			}
			readJSON(t, path, &value)
			if value.SchemaVersion != 1 || value.Name == "" {
				t.Fatalf("invalid contract identity: %#v", value)
			}
			if seen[value.Name] {
				t.Fatalf("duplicate contract name %q", value.Name)
			}
			seen[value.Name] = true
		})
	}
}

func readJSON(t *testing.T, path string, target any) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, target); err != nil {
		t.Fatalf("parse %s: %v", path, err)
	}
}

func repoRoot(t *testing.T) string {
	t.Helper()
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	return root
}
