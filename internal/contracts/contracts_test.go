package contracts

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/cipher982/hatch/internal/cli"
	runner "github.com/cipher982/hatch/internal/run"
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

type contractCase struct {
	Name             string            `json:"name"`
	ProviderBinaries []string          `json:"provider_binaries"`
	Arguments        []string          `json:"arguments"`
	Stdin            string            `json:"stdin"`
	Scenario         string            `json:"scenario"`
	Environment      map[string]string `json:"environment"`
	Expected         struct {
		ExitCode int `json:"exit_code"`
		Result   struct {
			OK       bool    `json:"ok"`
			Status   string  `json:"status"`
			Output   string  `json:"output"`
			ExitCode int     `json:"exit_code"`
			Error    *string `json:"error"`
			Stderr   *string `json:"stderr"`
		} `json:"result"`
		ProviderArgv        []string `json:"provider_argv"`
		ProviderStdinSHA256 string   `json:"provider_stdin_sha256"`
		ProviderStdinBytes  int      `json:"provider_stdin_bytes"`
		StderrContains      []string `json:"stderr_contains"`
	} `json:"expected"`
}

func TestLegacyParity(t *testing.T) {
	root := repoRoot(t)
	fake := filepath.Join(t.TempDir(), "testprovider")
	command := exec.Command("go", "build", "-o", fake, "./internal/testprovider")
	command.Dir = root
	if output, err := command.CombinedOutput(); err != nil {
		t.Fatalf("build fake provider: %v\n%s", err, output)
	}
	paths, err := filepath.Glob(filepath.Join(root, "testdata", "contracts", "cases", "*.json"))
	if err != nil || len(paths) == 0 {
		t.Fatalf("contract corpus: %v, count=%d", err, len(paths))
	}
	for _, path := range paths {
		var testCase contractCase
		readJSON(t, path, &testCase)
		t.Run(testCase.Name, func(t *testing.T) {
			directory := t.TempDir()
			for _, binary := range testCase.ProviderBinaries {
				if err := os.Symlink(fake, filepath.Join(directory, binary)); err != nil {
					t.Fatal(err)
				}
			}
			home := filepath.Join(directory, "home")
			if err := os.Mkdir(home, 0o700); err != nil {
				t.Fatal(err)
			}
			record := filepath.Join(directory, "invocation.json")
			t.Setenv("HOME", home)
			t.Setenv("PATH", directory+string(os.PathListSeparator)+os.Getenv("PATH"))
			t.Setenv("HATCH_RUN_ARTIFACT_ROOT", filepath.Join(directory, "artifacts"))
			t.Setenv("HATCH_TEST_RECORD", record)
			t.Setenv("HATCH_TEST_SCENARIO", testCase.Scenario)
			t.Setenv("HATCH_CREDENTIAL_HELPER", "")
			t.Setenv("OPENAI_API_KEY", "")
			t.Setenv("OPENROUTER_API_KEY", "")
			for name, value := range testCase.Environment {
				t.Setenv(name, value)
			}
			var stdout, stderr bytes.Buffer
			exit := cli.Main(testCase.Arguments, bytes.NewBufferString(testCase.Stdin), &stdout, &stderr, true)
			if exit != testCase.Expected.ExitCode {
				t.Fatalf("exit=%d want=%d stderr=%s stdout=%s", exit, testCase.Expected.ExitCode, stderr.String(), stdout.String())
			}
			for _, fragment := range testCase.Expected.StderrContains {
				if !bytes.Contains(stderr.Bytes(), []byte(fragment)) {
					t.Errorf("stderr lacks %q: %s", fragment, stderr.String())
				}
			}
			var result runner.PublicResult
			if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
				t.Fatalf("result JSON: %v\n%s", err, stdout.String())
			}
			if result.OK != testCase.Expected.Result.OK || result.Status != testCase.Expected.Result.Status ||
				result.Output != testCase.Expected.Result.Output || result.ExitCode != testCase.Expected.Result.ExitCode ||
				!reflect.DeepEqual(result.Error, testCase.Expected.Result.Error) || !reflect.DeepEqual(result.Stderr, testCase.Expected.Result.Stderr) {
				t.Errorf("legacy result mismatch: %#v expected %#v", result, testCase.Expected.Result)
			}
			if result.ArtifactPath == nil || result.Run == nil || result.Run.RunID == "" || result.Run.Capture.State != "durable" {
				t.Fatalf("target durability missing: %#v", result)
			}
			var invocation struct {
				Argv        []string          `json:"argv"`
				StdinSHA256 string            `json:"stdin_sha256"`
				StdinBytes  int               `json:"stdin_bytes"`
				Environment map[string]string `json:"environment"`
			}
			readJSON(t, record, &invocation)
			prepared := strings.TrimSuffix(string(mustRead(t, filepath.Join(root, "testdata", "contracts", "fixtures", "oracle_prepared_prompt.txt"))), "\n")
			for index, arg := range invocation.Argv {
				if arg == prepared {
					invocation.Argv[index] = "$PREPARED_PROMPT"
				}
			}
			if !reflect.DeepEqual(invocation.Argv, testCase.Expected.ProviderArgv) || invocation.StdinSHA256 != testCase.Expected.ProviderStdinSHA256 || invocation.StdinBytes != testCase.Expected.ProviderStdinBytes {
				t.Errorf("provider boundary = argv %#v stdin %s/%d", invocation.Argv, invocation.StdinSHA256, invocation.StdinBytes)
			}
			if invocation.Environment["DCG_NO_SELF_HEAL"] != "1" {
				t.Error("DCG_NO_SELF_HEAL missing")
			}
		})
	}
}

func mustRead(t *testing.T, path string) []byte {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return data
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
