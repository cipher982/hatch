package provider

import (
	"fmt"
	"strings"
)

const boundedRunContract = `Hatch execution contract:
This is a single bounded, non-interactive run with a time budget of about 15 minutes. Complete the requested scope and nothing more. Investigate proportionally to the question; once you have sufficient evidence, stop using tools and write your answer. Return a concise, decision-ready result. If you are blocked or running low on budget, return your best current findings and state what is uncertain rather than continuing to investigate. If the request explicitly asks for exhaustive or deep work, honor that instead.`

type Request struct {
	Backend         string
	Model           string
	Prompt          string
	CWD             string
	ReasoningEffort string
}

type Invocation struct {
	Argv     []string
	SetEnv   map[string]string
	UnsetEnv []string
	Stdin    []byte
}

func PreparePrompt(prompt string) string {
	return boundedRunContract + "\n\nUser task:\n" + prompt
}

func Build(req Request) (Invocation, error) {
	prompt := PreparePrompt(req.Prompt)
	switch req.Backend {
	case "claude":
		return Invocation{
			Argv: []string{
				"claude", "--verbose", "--print", "-", "--output-format", "stream-json",
				"--model", req.Model, "--dangerously-skip-permissions", "--setting-sources", "local",
				"--no-session-persistence", "--tools", "default", "--effort", "low",
				"--include-partial-messages",
			},
			Stdin: []byte(prompt),
			UnsetEnv: []string{
				"OPENAI_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY",
				"ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "CLAUDE_CODE_USE_BEDROCK",
			},
		}, nil
	case "cursor":
		return Invocation{
			Argv: []string{
				"cursor-agent", "--print", "--trust", "--model", req.Model,
				"--output-format", "stream-json", "--force", prompt,
			},
		}, nil
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
		return Invocation{
			Argv: argv,
			UnsetEnv: []string{
				"AWS_PROFILE", "AWS_REGION", "AWS_DEFAULT_REGION", "OPENAI_API_KEY", "CODEX_API_KEY",
			},
		}, nil
	case "gemini":
		model := req.Model
		if model == "" {
			model = "gemini-3-pro-preview"
		}
		return Invocation{
			Argv:  []string{"gemini", "--model", model, "--yolo", "--skip-trust", "-p", "-"},
			Stdin: []byte(prompt),
		}, nil
	default:
		return Invocation{}, fmt.Errorf("unsupported backend %q", req.Backend)
	}
}
