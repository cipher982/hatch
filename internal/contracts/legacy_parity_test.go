package contracts

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/cipher982/hatch/internal/cli"
	"github.com/cipher982/hatch/internal/provider"
)

func TestLegacyParityCommandBuilders(t *testing.T) {
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
				CWD: parsed.CWD, ReasoningEffort: parsed.ReasoningEffort,
			})
			if err != nil {
				t.Fatal(err)
			}
			expected := expandPrompt(t, testCase.Expected.ProviderArgv)
			if !reflect.DeepEqual(invocation.Argv, expected) {
				t.Fatalf("argv mismatch\n got: %#v\nwant: %#v", invocation.Argv, expected)
			}
		})
	}
}

func expandPrompt(t *testing.T, argv []string) []string {
	t.Helper()
	data, err := os.ReadFile(filepath.Join(repoRoot(t), "testdata", "contracts", "fixtures", "oracle_prepared_prompt.txt"))
	if err != nil {
		t.Fatal(err)
	}
	prompt := strings.TrimSuffix(string(data), "\n")
	result := append([]string(nil), argv...)
	for i, arg := range result {
		if arg == "$PREPARED_PROMPT" {
			result[i] = prompt
		}
	}
	return result
}
