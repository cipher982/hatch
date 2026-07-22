package contracts

import (
	"bytes"
	"encoding/json"
	"go/ast"
	"go/parser"
	"go/token"
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
		CollectedTests            int `json:"collected_tests"`
		PostBaselineContractTests int `json:"post_baseline_contract_tests"`
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
	if len(got.Tests) != got.Baseline.CollectedTests+got.Baseline.PostBaselineContractTests {
		t.Fatalf("ledger has %d tests, frozen baseline plus migration tests says %d", len(got.Tests), got.Baseline.CollectedTests+got.Baseline.PostBaselineContractTests)
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
		if test.Proof == "none" {
			if test.Disposition != "retired_python_library_only" && test.Disposition != "obsolete_implementation_detail" {
				t.Errorf("%s has no proof but is classified %q", test.NodeID, test.Disposition)
			}
		} else if test.Proof == "" {
			t.Errorf("%s has no proof target", test.NodeID)
		} else if strings.HasPrefix(test.Proof, "internal/contracts.TestContractLegacyParity/") {
			caseName := strings.TrimPrefix(test.Proof, "internal/contracts.TestContractLegacyParity/")
			if _, err := os.Stat(filepath.Join(repoRoot(t), "testdata", "contracts", "cases", caseName+".json")); err != nil {
				t.Errorf("%s names missing contract case proof %q", test.NodeID, test.Proof)
			}
		} else if !goProofExists(t, test.Proof) {
			t.Errorf("%s names non-executable proof target %q", test.NodeID, test.Proof)
		}
		if (test.Disposition == "intentional_change" || test.Disposition == "retired_python_library_only" || test.Disposition == "obsolete_implementation_detail") && (test.Reason == nil || *test.Reason == "") {
			t.Errorf("%s has disposition %q without a reason", test.NodeID, test.Disposition)
		}
	}
}

func goProofExists(t *testing.T, proof string) bool {
	t.Helper()
	separator := strings.LastIndex(proof, ".")
	if separator <= 0 || separator == len(proof)-1 {
		return false
	}
	directory := filepath.Join(repoRoot(t), filepath.FromSlash(proof[:separator]))
	want := proof[separator+1:]
	packages, err := parser.ParseDir(token.NewFileSet(), directory, func(info os.FileInfo) bool {
		return strings.HasSuffix(info.Name(), "_test.go")
	}, 0)
	if err != nil {
		return false
	}
	for _, pkg := range packages {
		for _, file := range pkg.Files {
			for _, declaration := range file.Decls {
				function, ok := declaration.(*ast.FuncDecl)
				if ok && function.Recv == nil && function.Name.Name == want {
					return true
				}
			}
		}
	}
	return false
}

func TestPythonTestLedgerMatchesFreshCollection(t *testing.T) {
	var got ledger
	root := repoRoot(t)
	readJSON(t, filepath.Join(root, "testdata", "contracts", "python-test-ledger.json"), &got)
	command := exec.Command("uv", "run", "pytest", "--collect-only", "-q")
	command.Dir = root
	output, err := command.CombinedOutput()
	if err != nil {
		t.Fatalf("collect Python tests: %v\n%s", err, output)
	}
	collected := map[string]bool{}
	for _, line := range strings.Split(string(output), "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "tests/") {
			collected[line] = true
		}
	}
	ledgerIDs := map[string]bool{}
	for _, row := range got.Tests {
		ledgerIDs[row.NodeID] = true
	}
	for id := range collected {
		if !ledgerIDs[id] {
			t.Errorf("fresh Python test missing from ledger: %s", id)
		}
	}
	for id := range ledgerIDs {
		if !collected[id] {
			t.Errorf("ledger test missing from fresh collection: %s", id)
		}
	}
	if len(collected) != len(ledgerIDs) {
		t.Fatalf("fresh collection=%d ledger=%d", len(collected), len(ledgerIDs))
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

func TestContractPythonOracle(t *testing.T) {
	command := exec.Command("uv", "run", "pytest", "tests/test_contract_harness.py", "-q")
	command.Dir = repoRoot(t)
	if output, err := command.CombinedOutput(); err != nil {
		t.Fatalf("Python contract oracle: %v\n%s", err, output)
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
		ExitCode            int             `json:"exit_code"`
		LegacyExitCode      *int            `json:"legacy_exit_code"`
		Result              contractResult  `json:"result"`
		LegacyResult        *contractResult `json:"legacy_result"`
		TargetResult        *contractResult `json:"target_result"`
		ProviderArgv        []string        `json:"provider_argv"`
		ProviderStdinSHA256 string          `json:"provider_stdin_sha256"`
		ProviderStdinBytes  int             `json:"provider_stdin_bytes"`
		StderrContains      []string        `json:"stderr_contains"`
	} `json:"expected"`
}

type contractResult struct {
	OK       bool    `json:"ok"`
	Status   string  `json:"status"`
	Output   string  `json:"output"`
	ExitCode int     `json:"exit_code"`
	Error    *string `json:"error"`
	Stderr   *string `json:"stderr"`
}

func TestContractLegacyParity(t *testing.T) {
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
			expectedResult := testCase.Expected.Result
			if testCase.Expected.TargetResult != nil {
				expectedResult = *testCase.Expected.TargetResult
			}
			if result.OK != expectedResult.OK || result.Status != expectedResult.Status ||
				result.Output != expectedResult.Output || result.ExitCode != expectedResult.ExitCode ||
				!reflect.DeepEqual(result.Error, expectedResult.Error) || !reflect.DeepEqual(result.Stderr, expectedResult.Stderr) {
				t.Errorf("target result mismatch: %#v expected %#v", result, expectedResult)
			}
			if result.ArtifactPath == nil || result.Run == nil || result.Run.RunID == "" || result.Run.Capture.State != "durable" {
				t.Fatalf("target durability missing: %#v", result)
			}
			if result.Run.Lifecycle != runner.LifecycleTerminal || result.Run.Capture.EvidenceSHA256 == nil {
				t.Fatalf("target terminal projection missing: %#v", result.Run)
			}
			for _, name := range []string{"manifest.json", "request.txt", result.Run.Capture.StdoutFile, result.Run.Capture.StderrFile, "result.txt", "result.json"} {
				info, err := os.Stat(filepath.Join(*result.ArtifactPath, name))
				if err != nil || info.Mode().Perm() != 0o600 {
					t.Fatalf("artifact %s mode=%v err=%v", name, info, err)
				}
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
