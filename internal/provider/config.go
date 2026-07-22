package provider

import (
	"fmt"
	"strings"
)

const boundedRunContract = `Hatch execution contract:
This is a single bounded, non-interactive run with a time budget of about 15 minutes. Complete the requested scope and nothing more. Investigate proportionally to the question; once you have sufficient evidence, stop using tools and write your answer. Return a concise, decision-ready result. If you are blocked or running low on budget, return your best current findings and state what is uncertain rather than continuing to investigate. If the request explicitly asks for exhaustive or deep work, honor that instead.`

type Request struct {
	Backend                string
	Model                  string
	Prompt                 string
	CWD                    string
	ReasoningEffort        string
	OutputFormat           string
	RawStructuredOutput    bool
	APIKey                 string
	Resume                 string
	SkipGitRepoCheck       bool
	IncludePartialMessages bool
}

type Invocation struct {
	Argv             []string
	PromptArgIndices []int
	SetEnv           map[string]string
	UnsetEnv         []string
	Stdin            []byte
	StreamFormat     string
	Adapter          string
	ProviderVersion  string
}

func PreparePrompt(prompt string) string {
	return boundedRunContract + "\n\nUser task:\n" + prompt
}

func Build(req Request) (Invocation, error) {
	prompt := PreparePrompt(req.Prompt)
	switch req.Backend {
	case "claude":
		model := req.Model
		if model == "" {
			model = "sonnet"
		}
		outputFormat := req.OutputFormat
		if outputFormat == "" || outputFormat == "text" {
			outputFormat = "stream-json"
		}
		argv := []string{"claude"}
		if outputFormat == "stream-json" {
			argv = append(argv, "--verbose")
		}
		argv = append(argv,
			"--print", "-", "--output-format", outputFormat,
			"--model", model, "--dangerously-skip-permissions", "--setting-sources", "local",
			"--no-session-persistence", "--tools", "default", "--effort", "low",
		)
		if req.IncludePartialMessages || req.OutputFormat == "" || req.OutputFormat == "text" {
			argv = append(argv, "--include-partial-messages")
		}
		if req.Resume != "" {
			argv = append(argv, "--resume", req.Resume)
		}
		adapter, streamFormat := "raw", "text"
		if outputFormat == "stream-json" && !req.RawStructuredOutput {
			adapter, streamFormat = "claude", "jsonl"
		} else if outputFormat == "stream-json" {
			streamFormat = "jsonl"
		}
		return Invocation{
			Argv: argv, Stdin: []byte(prompt), StreamFormat: streamFormat, Adapter: adapter,
			UnsetEnv: []string{
				"OPENAI_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY",
				"ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "CLAUDE_CODE_USE_BEDROCK",
				"AWS_PROFILE", "AWS_REGION", "AWS_DEFAULT_REGION", "ANTHROPIC_MODEL",
			},
		}, nil
	case "bedrock":
		model := req.Model
		if model == "" {
			model = "us.anthropic.claude-sonnet-4-6"
		}
		outputFormat := req.OutputFormat
		if outputFormat == "" || outputFormat == "text" {
			outputFormat = "stream-json"
		}
		argv := []string{"claude"}
		if outputFormat == "stream-json" {
			argv = append(argv, "--verbose")
		}
		argv = append(argv, "--print", "-", "--output-format", outputFormat,
			"--dangerously-skip-permissions", "--setting-sources", "local",
			"--no-session-persistence", "--tools", "", "--effort", "low")
		if req.IncludePartialMessages || req.OutputFormat == "" || req.OutputFormat == "text" {
			argv = append(argv, "--include-partial-messages")
		}
		if req.Resume != "" {
			argv = append(argv, "--resume", req.Resume)
		}
		adapter, streamFormat := "raw", "text"
		if outputFormat == "stream-json" && !req.RawStructuredOutput {
			adapter, streamFormat = "claude", "jsonl"
		} else if outputFormat == "stream-json" {
			streamFormat = "jsonl"
		}
		return Invocation{
			Argv: argv, Stdin: []byte(prompt), StreamFormat: streamFormat, Adapter: adapter,
			SetEnv: map[string]string{
				"CLAUDE_CODE_USE_BEDROCK": "1", "AWS_PROFILE": "zh-ml-mlengineer",
				"AWS_REGION": "us-east-1", "ANTHROPIC_MODEL": model,
			},
			UnsetEnv: []string{"AWS_DEFAULT_REGION", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"},
		}, nil
	case "cursor":
		model := req.Model
		if model == "" {
			model = "cursor-grok-4.5-high"
		}
		invocation := Invocation{
			Argv: []string{
				"cursor-agent", "--print", "--trust", "--model", model,
				"--output-format", "stream-json", "--force", prompt,
			}, StreamFormat: "jsonl", Adapter: "cursor", SetEnv: map[string]string{},
			UnsetEnv: []string{
				"OPENAI_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN",
				"ANTHROPIC_BASE_URL", "CLAUDE_CODE_USE_BEDROCK",
			},
		}
		invocation.PromptArgIndices = []int{len(invocation.Argv) - 1}
		if req.APIKey != "" {
			invocation.SetEnv["CURSOR_API_KEY"] = req.APIKey
		}
		return invocation, nil
	case "opencode":
		if req.Model == "" {
			return Invocation{}, fmt.Errorf("OpenCode backend requires an explicit model")
		}
		argv := []string{"opencode", "run", "--dangerously-skip-permissions"}
		if req.CWD != "" {
			argv = append(argv, "--dir", req.CWD)
		}
		argv = append(argv, "--pure", "--print-logs", "--log-level", "ERROR", "--format", "json", "-m", req.Model)
		if req.ReasoningEffort != "" && strings.HasPrefix(req.Model, "openai/") {
			argv = append(argv, "--variant", req.ReasoningEffort)
		}
		argv = append(argv, prompt)
		invocation := Invocation{
			Argv:             argv,
			PromptArgIndices: []int{len(argv) - 1},
			SetEnv:           map[string]string{},
			StreamFormat:     "jsonl", Adapter: "opencode",
			UnsetEnv: []string{
				"AWS_PROFILE", "AWS_REGION", "AWS_DEFAULT_REGION", "OPENAI_API_KEY", "CODEX_API_KEY",
			},
		}
		if strings.HasPrefix(req.Model, "openai/") && req.APIKey != "" {
			invocation.SetEnv["OPENAI_API_KEY"] = req.APIKey
		}
		if strings.HasPrefix(req.Model, "openrouter/") && req.APIKey != "" {
			invocation.SetEnv["OPENROUTER_API_KEY"] = req.APIKey
		}
		if strings.HasPrefix(req.Model, "amazon-bedrock/") {
			invocation.SetEnv["AWS_PROFILE"] = "zh-ml-mlengineer"
			invocation.SetEnv["AWS_REGION"] = "us-east-1"
		}
		return invocation, nil
	case "codex":
		if req.APIKey == "" {
			return Invocation{}, fmt.Errorf("OPENAI_API_KEY not set and no api_key provided")
		}
		argv := []string{"codex", "exec", "--dangerously-bypass-approvals-and-sandbox"}
		if req.Model != "" {
			argv = append(argv, "-m", req.Model)
		}
		if req.ReasoningEffort != "" {
			argv = append(argv, "-c", "model_reasoning_effort="+req.ReasoningEffort)
		}
		if req.SkipGitRepoCheck {
			argv = append(argv, "--skip-git-repo-check")
		}
		return Invocation{
			Argv: argv, Stdin: []byte(prompt), StreamFormat: "text", Adapter: "raw",
			SetEnv:   map[string]string{"OPENAI_API_KEY": req.APIKey},
			UnsetEnv: []string{"CODEX_API_KEY", "CLAUDE_CODE_USE_BEDROCK"},
		}, nil
	case "gemini":
		model := req.Model
		if model == "" {
			model = "gemini-3-pro-preview"
		}
		return Invocation{
			Argv:         []string{"gemini", "--model", model, "--yolo", "--skip-trust", "-p", "-"},
			Stdin:        []byte(prompt),
			StreamFormat: "text", Adapter: "raw",
			UnsetEnv: []string{"CLAUDE_CODE_USE_BEDROCK"},
		}, nil
	default:
		return Invocation{}, fmt.Errorf("unsupported backend %q", req.Backend)
	}
}
