package cli

import (
	"context"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/cipher982/hatch/internal/expert"
	"github.com/cipher982/hatch/internal/provider"
	runner "github.com/cipher982/hatch/internal/run"
)

type expertRequest struct {
	PromptArgs      []string
	Title           string
	CallerSession   string
	CallerRequest   string
	CWD             string
	MaxOutputBytes  int
	Model           string
	ReasoningEffort string
	APIKey          string
	TimeoutSeconds  int
	WebSearch       bool
	JSON            bool
	Help            bool
}

func runExpert(ctx context.Context, args []string, stdin io.Reader, stdout, stderr io.Writer) int {
	request, err := parseExpert(args)
	if err != nil {
		return renderConfigError(request.JSON, stdout, stderr, err)
	}
	if request.Help {
		fmt.Fprint(stdout, ExpertHelp)
		return 0
	}
	if err := runner.ValidateTitle(request.Title); err != nil {
		return renderConfigError(request.JSON, stdout, stderr, err)
	}
	provenance, err := runner.ResolveProvenance(request.CallerSession, request.CallerRequest)
	if err != nil {
		return renderConfigError(request.JSON, stdout, stderr, err)
	}
	if request.CWD != "" {
		info, err := os.Stat(request.CWD)
		if err != nil || !info.IsDir() {
			return renderConfigError(request.JSON, stdout, stderr, fmt.Errorf("cwd is not a directory: %s", request.CWD))
		}
	}
	policy, err := provider.ResolveReasoning("expert", request.Model, request.ReasoningEffort)
	if err != nil {
		return renderConfigError(request.JSON, stdout, stderr, err)
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
	fmt.Fprintf(stderr, "[hatch] expert call started: model=%s reasoning=%s source=%s support=%s web_search=%t\n", request.Model, policy.Effort, policy.Source, policy.Support, request.WebSearch)
	result := expert.Run(expert.Options{
		Context: ctx, Prompt: prompt, Model: request.Model, ReasoningEffort: request.ReasoningEffort, ReasoningPolicy: policy, WebSearch: request.WebSearch,
		Title: request.Title, Provenance: provenance, CWD: request.CWD,
		Timeout: time.Duration(request.TimeoutSeconds) * time.Second, APIKey: apiKey,
		BaseURL: strings.TrimSpace(os.Getenv("HATCH_EXPERT_RESPONSES_URL")), Store: runner.NewStore(root),
		Progress: newProgressSink(stderr),
	})
	return renderExpertResult(result, request.MaxOutputBytes, request.JSON, stdout, stderr)
}

func parseExpert(args []string) (expertRequest, error) {
	model := strings.TrimSpace(os.Getenv("HATCH_EXPERT_MODEL"))
	if model == "" {
		model = expert.DefaultModel
	}
	result := expertRequest{Model: model, TimeoutSeconds: 900, WebSearch: true, MaxOutputBytes: defaultOutputBytes}
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
		if hasInline && !oneOf(name, "--model", "--reasoning-effort", "--api-key", "--timeout", "--title", "--caller-session", "--caller-request", "--cwd", "--max-output-bytes") {
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
		case "--model", "--reasoning-effort", "--api-key", "-t", "--timeout", "--title", "--caller-session", "--caller-request", "-C", "--cwd", "--max-output-bytes":
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
			case "--title":
				result.Title = value
			case "--caller-session":
				result.CallerSession = value
			case "--caller-request":
				result.CallerRequest = value
			case "-C", "--cwd":
				result.CWD = value
			case "--max-output-bytes":
				n, err := parseOutputLimit(value)
				if err != nil {
					return result, err
				}
				result.MaxOutputBytes = n
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
  --title TEXT
  --caller-session ID / --caller-request ID
  -C, --cwd DIRECTORY
  --max-output-bytes N  256..32768 (default: 8192)
  --reasoning-effort LEVEL  none|low|medium|high|xhigh|max (default: medium)
  --web-search / --no-web-search
  -t, --timeout SECONDS
  --model MODEL
  --json
`
