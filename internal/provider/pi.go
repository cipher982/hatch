package provider

import "strings"

func buildPiLikeInvocation(req Request, policy ReasoningPolicy) (Invocation, error) {
	if req.Model == "" {
		return Invocation{}, invalidModelForBackend(req.Backend)
	}

	command := req.Backend
	argv := []string{command}
	if command == "pi" {
		argv = append(argv,
			"--print", "--mode", "json", "--no-session", "--approve",
			"--no-extensions", "--no-skills", "--no-prompt-templates", "--no-themes",
		)
	} else {
		argv = append(argv,
			"-p", "--mode", "json", "--no-session", "--no-title",
			"--no-extensions", "--no-skills", "--no-rules",
			"--auto-approve", "--approval-mode", "yolo",
		)
	}
	if req.CWD != "" {
		argv = append(argv, "--cwd", req.CWD)
	}
	argv = append(argv, "--model", req.Model)
	if policy.Effort != "" && policy.Support != "unsupported" {
		argv = append(argv, "--thinking", policy.Effort)
	}
	argv = append(argv, PreparePrompt(req.Prompt))

	invocation := Invocation{
		Argv:            argv,
		SetEnv:          map[string]string{},
		StreamFormat:    "jsonl",
		Adapter:         command,
		ReasoningPolicy: policy,
		UnsetEnv: []string{
			"AWS_PROFILE", "AWS_REGION", "AWS_DEFAULT_REGION",
			"OPENAI_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY",
			"ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_OAUTH_TOKEN", "ANTHROPIC_BASE_URL",
			"GEMINI_API_KEY", "CODEX_API_KEY", "PI_CODING_AGENT_DIR",
			"PI_CODING_AGENT_SESSION_DIR", "OMP_PROFILE",
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
	return redactInvocation(invocation, len(invocation.Argv)-1), nil
}

func invalidModelForBackend(backend string) error {
	return &backendModelError{backend: backend}
}

type backendModelError struct {
	backend string
}

func (e *backendModelError) Error() string {
	return e.backend + " backend requires an explicit model"
}
