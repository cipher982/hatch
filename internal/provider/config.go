package provider

import (
	"encoding/json"
	"fmt"
	"strings"
)

const boundedRunContract = `Hatch execution contract:
This is a single bounded, non-interactive run. A human is waiting for a useful answer by the behavioral deadline; do not treat the run as an open-ended session. Complete the requested scope and nothing more, using the available context and tool signals to manage the budget. Do not promise or assume an exact wall-clock duration.

Use focused checks by default. Time-box expensive tests, scratch clones or worktrees, broad repository scans, and network or fetch work; run a broad suite or exhaustive investigation only when explicitly requested or clearly required. Check the budget mid-run. Once evidence is sufficient, stop using tools and synthesize; never continue merely to eliminate every uncertainty. At the late budget threshold, stop launching tools and report incomplete evidence instead of racing the hard timeout. Preserve useful partial findings and do not redo completed work. Read each file at most once per run; when a read returns a continuation notice (for example "Use offset=N to continue"), issue exactly that continuation instead of re-reading from the start, and never re-run a search that already returned identical results. Start writing your answer once the core files are read: do not hold output until every file is read, and if the budget runs low, answer with the evidence you have and list what you did not read.

Nested Hatch runs are allowed when the user or task explicitly permits bounded parallel or independent subwork; "single run" describes this invocation, not a ban on child Hatch calls. If you launch children, give each a narrow scope, a small explicit count, and its own deadline. Do not recurse further unless the task explicitly authorizes recursion. Never wait indefinitely for a child: continue with completed results, record missing or timed-out children, and synthesize the best partial answer.

Return a concise status with findings, confidence, unresolved questions, and the exact next action. An incomplete or timed-out run must never be presented as approved or complete. If blocked or running low on budget, return the best current findings and state what is uncertain. If the request explicitly asks for exhaustive or deep work, honor that within the deadline.`

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
	Argv            []string
	RedactedArgv    []string
	SetEnv          map[string]string
	UnsetEnv        []string
	Stdin           []byte
	StreamFormat    string
	Adapter         string
	ProviderVersion string
	ReasoningPolicy ReasoningPolicy
	// OpenCodeConfigJSON, when set, is written into the per-run OpenCode
	// config dir as opencode.json before launch. Carries routing pins (for
	// example an OpenRouter provider order that keeps prefix caching warm);
	// never contains credentials.
	OpenCodeConfigJSON []byte
}

func PreparePrompt(prompt string) string {
	return boundedRunContract + "\n\nUser task:\n" + prompt
}

func Build(req Request) (Invocation, error) {
	if req.Backend == "opencode" && req.Model == "" {
		return Invocation{}, fmt.Errorf("OpenCode backend requires an explicit model")
	}
	policy, err := ResolveReasoning(req.Backend, req.Model, req.ReasoningEffort)
	if err != nil {
		return Invocation{}, err
	}
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
			"--no-session-persistence", "--tools", "default", "--effort", policy.Effort,
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
		return redactInvocation(Invocation{
			Argv: argv, Stdin: []byte(prompt), StreamFormat: streamFormat, Adapter: adapter,
			ReasoningPolicy: policy,
			UnsetEnv: []string{
				"OPENAI_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY",
				"ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "CLAUDE_CODE_USE_BEDROCK",
				"AWS_PROFILE", "AWS_REGION", "AWS_DEFAULT_REGION", "ANTHROPIC_MODEL",
			},
		}), nil
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
		return redactInvocation(Invocation{
			Argv: argv, Stdin: []byte(prompt), StreamFormat: streamFormat, Adapter: adapter,
			ReasoningPolicy: policy,
			SetEnv: map[string]string{
				"CLAUDE_CODE_USE_BEDROCK": "1", "AWS_PROFILE": "zh-ml-mlengineer",
				"AWS_REGION": "us-east-1", "ANTHROPIC_MODEL": model,
			},
			UnsetEnv: []string{"AWS_DEFAULT_REGION", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"},
		}), nil
	case "cursor":
		model := req.Model
		if model == "" {
			model = "cursor-grok-4.6-high"
		}
		invocation := Invocation{
			Argv: []string{
				"cursor-agent", "--print", "--trust", "--model", model,
				"--output-format", "stream-json", "--force", prompt,
			}, StreamFormat: "jsonl", Adapter: "cursor", ReasoningPolicy: policy, SetEnv: map[string]string{},
			UnsetEnv: []string{
				"OPENAI_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN",
				"ANTHROPIC_BASE_URL", "CLAUDE_CODE_USE_BEDROCK",
			},
		}
		invocation = redactInvocation(invocation, len(invocation.Argv)-1)
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
		if policy.Effort != "" && policy.Support != "unsupported" {
			argv = append(argv, "--variant", policy.Effort)
		}
		argv = append(argv, prompt)
		invocation := Invocation{
			Argv:         argv,
			SetEnv:       map[string]string{},
			StreamFormat: "jsonl", Adapter: "opencode", ReasoningPolicy: policy,
			UnsetEnv: []string{
				"AWS_PROFILE", "AWS_REGION", "AWS_DEFAULT_REGION", "OPENAI_API_KEY", "CODEX_API_KEY",
				"OPENCODE_CONFIG", "OPENCODE_CONFIG_CONTENT", "OPENCODE_CONFIG_DIR", "OPENCODE_DISABLE_PROJECT_CONFIG",
				"XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME", "XDG_CACHE_HOME",
			},
		}
		if strings.HasPrefix(req.Model, "openai/") && req.APIKey != "" {
			invocation.SetEnv["OPENAI_API_KEY"] = req.APIKey
		}
		if strings.HasPrefix(req.Model, "openrouter/") && req.APIKey != "" {
			invocation.SetEnv["OPENROUTER_API_KEY"] = req.APIKey
		}
		if strings.HasPrefix(req.Model, "openrouter/deepseek/deepseek-v4-flash") {
			invocation.OpenCodeConfigJSON = openCodeDeepSeekRoutingConfig(strings.TrimPrefix(req.Model, "openrouter/"))
		} else if strings.HasPrefix(req.Model, "openrouter/z-ai/glm-5.3-flash") {
			invocation.OpenCodeConfigJSON = openCodeGLMRoutingConfig(strings.TrimPrefix(req.Model, "openrouter/"))
		}
		if strings.HasPrefix(req.Model, "amazon-bedrock/") {
			invocation.SetEnv["AWS_PROFILE"] = "zh-ml-mlengineer"
			invocation.SetEnv["AWS_REGION"] = "us-east-1"
		}
		return redactInvocation(invocation, len(invocation.Argv)-1), nil
	case "pi", "omp":
		return buildPiLikeInvocation(req, policy)
	case "codex":
		if req.APIKey == "" {
			return Invocation{}, fmt.Errorf("OPENAI_API_KEY not set and no api_key provided")
		}
		argv := []string{"codex", "exec", "--dangerously-bypass-approvals-and-sandbox", "--ignore-user-config", "--ephemeral"}
		if req.Model != "" {
			argv = append(argv, "-m", req.Model)
		}
		argv = append(argv, "-c", "model_reasoning_effort="+policy.Effort)
		if req.SkipGitRepoCheck {
			argv = append(argv, "--skip-git-repo-check")
		}
		return redactInvocation(Invocation{
			Argv: argv, Stdin: []byte(prompt), StreamFormat: "text", Adapter: "raw",
			ReasoningPolicy: policy,
			SetEnv:          map[string]string{"OPENAI_API_KEY": req.APIKey},
			UnsetEnv:        []string{"CODEX_API_KEY", "CODEX_HOME", "CLAUDE_CODE_USE_BEDROCK"},
		}), nil
	case "gemini":
		model := req.Model
		if model == "" {
			model = "gemini-3-pro-preview"
		}
		return redactInvocation(Invocation{
			Argv:         []string{"gemini", "--model", model, "--yolo", "--skip-trust", "-p", "-"},
			Stdin:        []byte(prompt),
			StreamFormat: "text", Adapter: "raw", ReasoningPolicy: policy,
			UnsetEnv: []string{"CLAUDE_CODE_USE_BEDROCK"},
		}), nil
	default:
		return Invocation{}, fmt.Errorf("unsupported backend %q", req.Backend)
	}
}

func redactInvocation(invocation Invocation, promptIndices ...int) Invocation {
	invocation.RedactedArgv = append([]string(nil), invocation.Argv...)
	for _, index := range promptIndices {
		if index >= 0 && index < len(invocation.RedactedArgv) {
			invocation.RedactedArgv[index] = "<prompt>"
		}
	}
	return invocation
}

// openCodeDeepSeekRoutingConfig pins an OpenRouter deepseek-v4-flash model to a
// provider order with working prefix caching. Measured on the hatch account
// (2026-08): DeepSeek 98% cache hit, CoreWeave 91%, Novita 91%, DeepInfra 90%;
// the default price-based load balancer frequently picks non-caching endpoints
// (DigitalOcean, OpenInference) that re-encode the full growing context on
// every agent step, adding tens of seconds of latency per step. Setting
// provider.order disables load balancing and is tried in order; allow_fallbacks
// engages only on provider failure.
func openCodeDeepSeekRoutingConfig(model string) []byte {
	config := map[string]any{
		"provider": map[string]any{
			"openrouter": map[string]any{
				"models": map[string]any{
					model: map[string]any{
						"options": map[string]any{
							"provider": map[string]any{
								"order":           []string{"DeepSeek", "CoreWeave", "Novita", "DeepInfra"},
								"allow_fallbacks": true,
							},
						},
					},
				},
			},
		},
	}
	encoded, err := json.Marshal(config)
	if err != nil {
		panic("static opencode routing config cannot fail to marshal")
	}
	return encoded
}

// openCodeGLMRoutingConfig configures OpenRouter GLM-5.3 Flash with Modal as the
// preferred provider and fallbacks to healthy providers (Z.AI, Novita, Together,
// Parasail, DeepInfra) when Modal lacks specific parameter support (such as tool_choice: auto).
func openCodeGLMRoutingConfig(model string) []byte {
	config := map[string]any{
		"provider": map[string]any{
			"openrouter": map[string]any{
				"models": map[string]any{
					model: map[string]any{
						"options": map[string]any{
							"provider": map[string]any{
								"order":           []string{"Modal", "Z.AI", "Novita", "Together", "Parasail", "DeepInfra"},
								"allow_fallbacks": true,
							},
						},
					},
				},
			},
		},
	}
	encoded, err := json.Marshal(config)
	if err != nil {
		panic("static opencode routing config cannot fail to marshal")
	}
	return encoded
}
