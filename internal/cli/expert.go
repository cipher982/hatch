package cli

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/cipher982/hatch/internal/expert"
	runner "github.com/cipher982/hatch/internal/run"
)

type expertRequest struct {
	PromptArgs      []string
	Model           string
	ReasoningEffort string
	APIKey          string
	TimeoutSeconds  int
	WebSearch       bool
	JSON            bool
	Help            bool
}

func runExpert(args []string, stdin io.Reader, stdout, stderr io.Writer) int {
	request, err := parseExpert(args)
	if err != nil {
		return renderConfigError(request.JSON, stdout, stderr, err)
	}
	if request.Help {
		fmt.Fprint(stdout, ExpertHelp)
		return 0
	}
	prompt, err := readPrompt(request.PromptArgs, stdin)
	if err != nil {
		return renderConfigError(request.JSON, stdout, stderr, err)
	}
	apiKey, err := resolveCredential(request.APIKey, "OPENAI_API_KEY")
	if err != nil {
		return renderConfigError(request.JSON, stdout, stderr, err)
	}
	if apiKey == "" {
		return renderConfigError(request.JSON, stdout, stderr, fmt.Errorf("OPENAI_API_KEY is not set or available from the configured credential helper"))
	}
	root, err := runner.DefaultRoot()
	if err != nil {
		return renderConfigError(request.JSON, stdout, stderr, err)
	}
	fmt.Fprintf(stderr, "[hatch] expert call started: model=%s reasoning=%s web_search=%t\n", request.Model, request.ReasoningEffort, request.WebSearch)
	result := expert.Run(expert.Options{
		Prompt: prompt, Model: request.Model, ReasoningEffort: request.ReasoningEffort, WebSearch: request.WebSearch,
		Timeout: time.Duration(request.TimeoutSeconds) * time.Second, APIKey: apiKey,
		BaseURL: strings.TrimSpace(os.Getenv("HATCH_EXPERT_RESPONSES_URL")), Store: runner.NewStore(root),
		Progress: func(message string) { fmt.Fprintln(stderr, message) },
	})
	if request.JSON {
		encoder := json.NewEncoder(stdout)
		encoder.SetEscapeHTML(false)
		if err := encoder.Encode(result); err != nil {
			fmt.Fprintf(stderr, "Error: encode result: %v\n", err)
			return 1
		}
	} else if result.OK {
		fmt.Fprintln(stdout, strings.TrimRight(result.Output, "\n"))
	} else if result.Error != nil {
		fmt.Fprintf(stderr, "Error: %s\n", *result.Error)
	}
	return result.ExitCode
}

func parseExpert(args []string) (expertRequest, error) {
	model := strings.TrimSpace(os.Getenv("HATCH_EXPERT_MODEL"))
	if model == "" {
		model = expert.DefaultModel
	}
	result := expertRequest{Model: model, ReasoningEffort: "medium", TimeoutSeconds: 900, WebSearch: true}
	literal := false
	for index := 0; index < len(args); index++ {
		arg := args[index]
		if literal {
			result.PromptArgs = append(result.PromptArgs, arg)
			continue
		}
		if arg == "--" {
			literal = true
			continue
		}
		name, inline, hasInline := splitLongFlag(arg)
		if hasInline && !oneOf(name, "--model", "--reasoning-effort", "--api-key", "--timeout") {
			return result, fmt.Errorf("unrecognized argument: %s", arg)
		}
		switch name {
		case "-h", "--help":
			result.Help = true
		case "--json":
			result.JSON = true
		case "--web-search":
			result.WebSearch = true
		case "--no-web-search":
			result.WebSearch = false
		case "--model", "--reasoning-effort", "--api-key", "-t", "--timeout":
			value := inline
			if !hasInline {
				if index+1 >= len(args) {
					return result, fmt.Errorf("%s requires a value", name)
				}
				index++
				value = args[index]
			}
			switch name {
			case "--model":
				result.Model = value
			case "--reasoning-effort":
				if !oneOf(value, "none", "low", "medium", "high", "xhigh", "max") {
					return result, fmt.Errorf("invalid reasoning effort %q", value)
				}
				result.ReasoningEffort = value
			case "--api-key":
				result.APIKey = value
			case "-t", "--timeout":
				seconds, err := strconv.Atoi(value)
				if err != nil || seconds <= 0 {
					return result, fmt.Errorf("timeout must be > 0")
				}
				result.TimeoutSeconds = seconds
			}
		default:
			if strings.HasPrefix(arg, "-") && arg != "-" {
				return result, fmt.Errorf("unrecognized argument: %s", arg)
			}
			result.PromptArgs = append(result.PromptArgs, arg)
		}
	}
	return result, nil
}

const ExpertHelp = `usage: hatch expert [OPTIONS] "prompt"

Ask one slow synchronous expert question using the Responses API.

Options:
  --reasoning-effort LEVEL
  --web-search / --no-web-search
  -t, --timeout SECONDS
  --model MODEL
  --json
`
