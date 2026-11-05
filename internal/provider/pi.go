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
	// Pi has no --cwd flag; it works in the directory it is launched in, and the
	// coordinator launches every run in the requested directory.
	if req.CWD != "" && command == "omp" {
		argv = append(argv, "--cwd", req.CWD)
	}
	if req.ReadOnly {
		tools := piReadOnlyTools
		if command == "omp" {
			tools = ompReadOnlyTools
		}
		argv = append(argv, "--tools", tools)
	}
	argv = append(argv, "--model", req.Model)
	if policy.Effort != "" && policy.Support != "unsupported" {
		argv = append(argv, "--thinking", piThinkingLevel(policy.Effort))
	}
	// Oh My Pi exposes OpenAI's service tier as a flag; Pi gets it through the
	// per-run models.json (PiModelsJSON).
	if command == "omp" && RequestsPriorityTier(req.Model, policy.Effort) {
		argv = append(argv, "--service-tier", openAIPriorityServiceTier)
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
			"OPENAI_API_KEY", "OPENROUTER_API_KEY", "OPENCODE_API_KEY", "ANTHROPIC_API_KEY",
			"ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_OAUTH_TOKEN", "ANTHROPIC_BASE_URL",
			"GEMINI_API_KEY", "CODEX_API_KEY", "PI_CODING_AGENT_DIR",
			"PI_CODING_AGENT_SESSION_DIR", "OMP_PROFILE",
		},
	}
	if strings.HasPrefix(req.Model, "openai/") && req.APIKey != "" {
		invocation.SetEnv["OPENAI_API_KEY"] = req.APIKey
	}
	if strings.HasPrefix(req.Model, "opencode-go/") && req.APIKey != "" {
		invocation.SetEnv["OPENCODE_API_KEY"] = req.APIKey
	}
	if strings.HasPrefix(req.Model, "openrouter/") && req.APIKey != "" {
		invocation.SetEnv["OPENROUTER_API_KEY"] = req.APIKey
	}
	if strings.HasPrefix(req.Model, "amazon-bedrock/") {
		invocation.SetEnv["AWS_PROFILE"] = BedrockAWSProfile
		invocation.SetEnv["AWS_REGION"] = BedrockAWSRegion
	}
	if command == "pi" {
		invocation.PiModelsJSON = PiModelsJSON(req.Model, policy.Effort)
	}
	return redactInvocation(invocation, len(invocation.Argv)-1), nil
}

// piThinkingLevel maps Hatch's effort names to Pi's: "none" is Pi's "off".
func piThinkingLevel(effort string) string {
	if effort == "none" {
		return "off"
	}
	return effort
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
