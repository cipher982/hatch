package provider

import (
	"fmt"
)

const boundedRunContract = `Hatch execution contract:
This is a single bounded, non-interactive run. A human is waiting for a useful answer by the behavioral deadline; do not treat the run as an open-ended session. Complete the requested scope and nothing more, using the available context and tool signals to manage the budget. Do not promise or assume an exact wall-clock duration.

Use focused checks by default. Time-box expensive tests, scratch clones or worktrees, broad repository scans, and network or fetch work; run a broad suite or exhaustive investigation only when explicitly requested or clearly required. Check the budget mid-run. Once evidence is sufficient, stop using tools and synthesize; never continue merely to eliminate every uncertainty. At the late budget threshold, stop launching tools and report incomplete evidence instead of racing the hard timeout. Preserve useful partial findings and do not redo completed work. Read each file at most once per run; when a read returns a continuation notice (for example "Use offset=N to continue"), issue exactly that continuation instead of re-reading from the start, and never re-run a search that already returned identical results. Start writing your answer once the core files are read: do not hold output until every file is read, and if the budget runs low, answer with the evidence you have and list what you did not read.

Nested Hatch runs are allowed when the user or task explicitly permits bounded parallel or independent subwork; "single run" describes this invocation, not a ban on child Hatch calls. If you launch children, give each a narrow scope, a small explicit count, and its own deadline. Do not recurse further unless the task explicitly authorizes recursion. Never wait indefinitely for a child: continue with completed results, record missing or timed-out children, and synthesize the best partial answer.

Start with the conclusion or recommendation, then give only the key supporting evidence, material risks or uncertainty, and exact next action.
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
	// PiModelsJSON, when set, is written into the per-run Pi config dir as
	// models.json before launch (see PiModelsJSON). Never contains credentials.
	PiModelsJSON []byte
	// PiDCGRunner, when set, loads the DCG guard extension into the Pi run and
	// names the Agent Home guard runner (~/.config/dcg/hook-runner) it asks
	// before every bash command.
	PiDCGRunner string
	// AntigravityDCGRunner, when set, writes a dcg-guard hook entry in the
	// per-run .gemini/config/hooks.json pointing at Agent Home's hook-runner.
	AntigravityDCGRunner string
}

func PreparePrompt(prompt string) string {
	return boundedRunContract + "\n\nUser task:\n" + prompt
}

func Build(req Request) (Invocation, error) {
	policy, err := ResolveReasoning(req.Backend, req.Model, req.ReasoningEffort)
	if err != nil {
		return Invocation{}, err
	}
	return buildInvocation(req, policy)
}

func buildInvocation(req Request, policy ReasoningPolicy) (Invocation, error) {
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
		argv = append(argv, "--print", "-", "--output-format", outputFormat, "--model", model,
			"--dangerously-skip-permissions",
			"--setting-sources", "local",
			"--no-session-persistence", "--tools", "default", "--effort", policy.Effort,
		)
		if req.IncludePartialMessages || req.OutputFormat == "" || req.OutputFormat == "text" {
			argv = append(argv, "--include-partial-messages")
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
				"OPENAI_API_KEY", "OPENROUTER_API_KEY", "OPENCODE_API_KEY", "ANTHROPIC_API_KEY",
				"ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "CLAUDE_CODE_USE_BEDROCK",
				"AWS_PROFILE", "AWS_REGION", "AWS_DEFAULT_REGION", "ANTHROPIC_MODEL",
			},
		}), nil
	case "cursor":
		model := req.Model
		if model == "" {
			model = CursorSurfaceModels["grok"]
		}
		argv := []string{"cursor-agent", "--print", "--trust", "--model", model, "--output-format", "stream-json", "--force"}
		invocation := Invocation{
			Argv: append(argv, prompt), StreamFormat: "jsonl", Adapter: "cursor", ReasoningPolicy: policy, SetEnv: map[string]string{},
			UnsetEnv: []string{
				"OPENAI_API_KEY", "OPENROUTER_API_KEY", "OPENCODE_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN",
				"ANTHROPIC_BASE_URL", "CLAUDE_CODE_USE_BEDROCK",
			},
		}
		invocation = redactInvocation(invocation, len(invocation.Argv)-1)
		if req.APIKey != "" {
			invocation.SetEnv["CURSOR_API_KEY"] = req.APIKey
		}
		return invocation, nil
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
	case "antigravity":
		// Google's Antigravity CLI. Its print flag takes the prompt as its
		// value, and flags placed after it are read as prompt text, so the
		// prompt must come last.
		model := req.Model
		if model == "" {
			return Invocation{}, invalidModelForBackend(req.Backend)
		}
		argv := []string{"agy", "--dangerously-skip-permissions", "--output-format", "stream-json", "--model", model, "-p", prompt}
		return redactInvocation(Invocation{
			Argv: argv, StreamFormat: "jsonl", Adapter: "antigravity", ReasoningPolicy: policy,
			SetEnv: map[string]string{},
			UnsetEnv: []string{
				"OPENAI_API_KEY", "OPENROUTER_API_KEY", "OPENCODE_API_KEY", "ANTHROPIC_API_KEY",
				"ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_OAUTH_TOKEN", "ANTHROPIC_BASE_URL",
				"GEMINI_API_KEY", "CODEX_API_KEY", "CURSOR_API_KEY", "CLAUDE_CODE_USE_BEDROCK",
				"AWS_PROFILE", "AWS_REGION", "AWS_DEFAULT_REGION",
				"LONGHOUSE_HOME", "LONGHOUSE_HOOK_TOKEN", "LONGHOUSE_HOOK_URL",
				"LONGHOUSE_MANAGED_SESSION_ID", "LONGHOUSE_MANAGED_PROVIDER", "LONGHOUSE_RUN_ID",
				"LONGHOUSE_ANTIGRAVITY_INBOX_DIR", "LONGHOUSE_ANTIGRAVITY_STATE_DIR",
			},
		}, len(argv)-1), nil
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
