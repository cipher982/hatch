package contracts

import (
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/cipher982/hatch/internal/cli"
	"github.com/cipher982/hatch/internal/provider"
)

func TestContractLegacyParityCommandBuilders(t *testing.T) {
	paths, err := filepath.Glob(filepath.Join(repoRoot(t), "testdata", "contracts", "cases", "*.json"))
	if err != nil {
		t.Fatal(err)
	}
	if len(paths) == 0 {
		t.Fatal("legacy parity corpus is empty")
	}
	for _, path := range paths {
		path := path
		t.Run(strings.TrimSuffix(filepath.Base(path), filepath.Ext(path)), func(t *testing.T) {
			var testCase struct {
				Arguments []string `json:"arguments"`
				Stdin     string   `json:"stdin"`
				Expected  struct {
					ProviderArgv []string `json:"provider_argv"`
				} `json:"expected"`
			}
			readJSON(t, path, &testCase)
			parsed, err := cli.Parse(testCase.Arguments, true)
			if err != nil {
				t.Fatal(err)
			}
			prompt := strings.Join(parsed.PromptArgs, " ")
			if len(parsed.PromptArgs) == 0 || (len(parsed.PromptArgs) == 1 && parsed.PromptArgs[0] == "-") {
				prompt = testCase.Stdin
			}
			invocation, err := provider.Build(provider.Request{
				Backend: parsed.Backend, Model: parsed.Model, Prompt: prompt,
				CWD: parsed.CWD, ReasoningEffort: parsed.ReasoningEffort, APIKey: parsed.APIKey,
				OutputFormat: parsed.OutputFormat, Resume: parsed.Resume,
				RawStructuredOutput: parsed.OutputFormatExplicit && parsed.OutputFormat == "stream-json",
				SkipGitRepoCheck:    parsed.SkipGitRepoCheck, IncludePartialMessages: parsed.IncludePartialMessages,
			})
			if err != nil {
				t.Fatal(err)
			}
			expected := testCase.Expected.ProviderArgv
			argv := normalizePromptArgs(t, invocation.Argv, expected, prompt)
			if !reflect.DeepEqual(argv, expected) {
				t.Fatalf("provider flags differ: got %#v want %#v", argv, expected)
			}
		})
	}
}

// The frozen corpus owns provider flags and prompt transport, not editorial
// wording in the current bounded-run instruction. Preserve the original task.
func normalizePromptArgs(t *testing.T, actual, expected []string, prompt string) []string {
	t.Helper()
	if len(actual) != len(expected) {
		t.Fatalf("provider argument count = %d, want %d", len(actual), len(expected))
	}
	result := append([]string(nil), actual...)
	for i, arg := range expected {
		if arg == "$PREPARED_PROMPT" {
			if !strings.HasSuffix(actual[i], prompt) {
				t.Fatal("argv-prompt provider did not receive the original task intact")
			}
			result[i] = arg
		}
	}
	return result
}
