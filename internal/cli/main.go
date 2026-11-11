package cli

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"runtime"
	"strings"
	"time"

	"github.com/cipher982/hatch/internal/doctor"
	"github.com/cipher982/hatch/internal/provider"
	runner "github.com/cipher982/hatch/internal/run"
)

var Version = "0.1.0-go-dev"
var Commit = "unknown"
var Dirty = "unknown"
var BuildGoVersion = ""
var BuildTarget = ""

func Main(args []string, stdin io.Reader, stdout, stderr io.Writer, stdoutTTY bool) int {
	return MainContext(context.Background(), args, stdin, stdout, stderr, stdoutTTY)
}

func MainContext(ctx context.Context, args []string, stdin io.Reader, stdout, stderr io.Writer, stdoutTTY bool) int {
	if len(args) > 0 && args[0] == "expert" {
		return runExpert(ctx, args[1:], stdin, stdout, stderr)
	}
	if len(args) > 0 && args[0] == "runs" {
		return runRuns(args[1:], stdout, stderr, stdoutTTY)
	}
	if len(args) > 0 && args[0] == "doctor" {
		return runDoctor(args[1:], stdout, stderr)
	}
	if len(args) > 0 && args[0] == "catalog" {
		return runCatalog(args[1:], stdout, stderr)
	}
	if len(args) > 0 && args[0] == "review" {
		return runReview(args[1:], stderr)
	}
	request, err := Parse(args, stdoutTTY)
	if err != nil {
		return renderConfigError(request.JSON || !stdoutTTY, stdout, stderr, err)
	}
	if request.Help {
		if request.AdvancedHelp {
			fmt.Fprint(stdout, AdvancedHelp)
		} else {
			fmt.Fprint(stdout, Help)
		}
		return 0
	}
	if request.Version {
		goVersion := BuildGoVersion
		if goVersion == "" {
			goVersion = runtime.Version()
		}
		target := BuildTarget
		if target == "" {
			target = runtime.GOOS + "/" + runtime.GOARCH
		}
		fmt.Fprintf(stdout, "hatch %s (commit=%s dirty=%s go=%s target=%s)\n", Version, Commit, Dirty, goVersion, target)
		return 0
	}
	if err := runner.ValidateTitle(request.Title); err != nil {
		return renderConfigError(request.JSON, stdout, stderr, err)
	}
	provenance, err := runner.ResolveProvenance(request.CallerSession, request.CallerRequest)
	if err != nil {
		return renderConfigError(request.JSON, stdout, stderr, err)
	}
	root, err := runner.DefaultRoot()
	if err != nil {
		return renderConfigError(request.JSON, stdout, stderr, err)
	}
	var prior *runner.PriorRun
	if request.Resume != "" {
		loaded, err := runner.LoadPriorRun(root, request.Resume)
		if err != nil {
			return renderConfigError(request.JSON, stdout, stderr, err)
		}
		prior = &loaded
		inheritFromPriorRun(&request, loaded)
	}
	if request.Harness != "" {
		if !oneOf(request.Backend, "pi", "omp") {
			return renderConfigError(request.JSON || !stdoutTTY, stdout, stderr, fmt.Errorf("--harness can only select the coding harness for bedrock, codex, opencode-go, or openrouter surfaces"))
		}
		request.Backend = request.Harness
	}
	if request.Backend == "" {
		return renderConfigError(request.JSON, stdout, stderr, fmt.Errorf("No default model is configured; choose an explicit provider"))
	}
	if !oneOf(request.Backend, "antigravity", "claude", "cursor", "codex", "pi", "omp") {
		return renderConfigError(request.JSON, stdout, stderr, fmt.Errorf("invalid backend %q. Choose one of: antigravity, claude, cursor, codex, pi, omp", request.Backend))
	}
	if request.CWD != "" {
		info, statErr := os.Stat(request.CWD)
		if statErr != nil {
			return renderConfigError(request.JSON, stdout, stderr, fmt.Errorf("cwd does not exist: %s", request.CWD))
		}
		if !info.IsDir() {
			return renderConfigError(request.JSON, stdout, stderr, fmt.Errorf("cwd is not a directory: %s", request.CWD))
		}
	}
	if err := checkFlagSupport(request); err != nil {
		return renderConfigError(request.JSON, stdout, stderr, err)
	}
	if _, err := provider.ResolveReasoning(request.Backend, request.Model, request.ReasoningEffort); err != nil {
		return renderConfigError(request.JSON, stdout, stderr, err)
	}
	prompt, err := readPrompt(request.PromptArgs, stdin)
	if err != nil {
		return renderConfigError(request.JSON, stdout, stderr, err)
	}
	apiKey := request.APIKey
	credentialEnvironment := ""
	if strings.HasPrefix(request.Model, "openrouter/") {
		credentialEnvironment = "OPENROUTER_API_KEY"
	} else if strings.HasPrefix(request.Model, "opencode-go/") {
		credentialEnvironment = "OPENCODE_API_KEY"
	} else if strings.HasPrefix(request.Model, "openai/") || request.Backend == "codex" {
		credentialEnvironment = "OPENAI_API_KEY"
	} else if request.Backend == "cursor" {
		credentialEnvironment = "CURSOR_API_KEY"
	}
	if credentialEnvironment != "" {
		apiKey, err = resolveCredential(apiKey, credentialEnvironment)
		if err != nil {
			return renderConfigError(request.JSON, stdout, stderr, err)
		}
	}
	if strings.HasPrefix(request.Model, "openrouter/") && apiKey == "" {
		return renderConfigError(request.JSON, stdout, stderr, fmt.Errorf("OPENROUTER_API_KEY is not set or available from the configured credential helper"))
	}
	if strings.HasPrefix(request.Model, "opencode-go/") && apiKey == "" {
		return renderConfigError(request.JSON, stdout, stderr, fmt.Errorf("OPENCODE_API_KEY is not set or available from the configured credential helper"))
	}
	if (strings.HasPrefix(request.Model, "openai/") || request.Backend == "codex") && apiKey == "" {
		return renderConfigError(request.JSON, stdout, stderr, fmt.Errorf("OPENAI_API_KEY is not set or available from the configured credential helper"))
	}
	surface, providerName := identity(request.Backend, request.Model)
	providerPrompt := prompt
	if prior != nil {
		if provider.SurfaceAccount(prior.Surface) != provider.SurfaceAccount(surface) {
			return renderConfigError(request.JSON, stdout, stderr, fmt.Errorf(
				"run %s used %s (%s account) and %s is on the %s account; work and personal text never cross. Continue with a %s model, or start a fresh run",
				prior.RunID, prior.Surface, provider.SurfaceAccount(prior.Surface), surface, provider.SurfaceAccount(surface), provider.SurfaceAccount(prior.Surface)))
		}
		providerPrompt = prior.FollowUpPrompt(prompt)
	}
	invocation, err := provider.Build(provider.Request{
		Backend: request.Backend, Model: request.Model, Prompt: providerPrompt, CWD: request.CWD,
		ReasoningEffort: request.ReasoningEffort, OutputFormat: request.OutputFormat, APIKey: apiKey,
		RawStructuredOutput:    request.OutputFormatExplicit && request.OutputFormat == "stream-json",
		SkipGitRepoCheck:       request.SkipGitRepoCheck,
		IncludePartialMessages: request.IncludePartialMessages, ReadOnly: request.ReadOnly,
	})
	if err != nil {
		return renderConfigError(request.JSON, stdout, stderr, err)
	}
	if err := applyHostContext(&invocation, request.Backend, DetectContext()); err != nil {
		return renderConfigError(request.JSON, stdout, stderr, err)
	}
	populateProviderVersion(&invocation)
	if err := preflightBedrock(request.Model, invocation); err != nil {
		return renderConfigError(request.JSON, stdout, stderr, err)
	}
	credentialNames := []string{}
	if strings.HasPrefix(request.Model, "openrouter/") {
		credentialNames = append(credentialNames, "OPENROUTER_API_KEY")
	} else if strings.HasPrefix(request.Model, "opencode-go/") {
		credentialNames = append(credentialNames, "OPENCODE_API_KEY")
	} else if strings.HasPrefix(request.Model, "openai/") || request.Backend == "codex" {
		credentialNames = append(credentialNames, "OPENAI_API_KEY")
	} else if request.Backend == "cursor" && apiKey != "" {
		credentialNames = append(credentialNames, "CURSOR_API_KEY")
	}
	coordinator := runner.NewCoordinator(runner.NewStore(root))
	result := coordinator.Execute(runner.Request{
		Context: ctx, Surface: surface, Backend: request.Backend, Provider: providerName, Model: request.Model, CWD: request.CWD,
		Title: request.Title, Provenance: provenance,
		Prompt: prompt, Timeout: time.Duration(request.TimeoutSeconds) * time.Second,
		Invocation: invocation, CredentialNames: credentialNames, Automation: request.Automation,
		ProgressLabel: progressLabel(surface), Progress: newProgressSink(stderr),
	})
	return renderResult(result, request.MaxOutputBytes, request.JSON, stdout, stderr)
}

func progressLabel(surface string) string {
	switch {
	case strings.HasPrefix(surface, "claude."):
		return "Claude"
	case strings.HasPrefix(surface, "bedrock."):
		return "Bedrock"
	case strings.HasPrefix(surface, "cursor."):
		return "Cursor"
	case strings.HasPrefix(surface, "openrouter."):
		return "OpenRouter"
	case strings.HasPrefix(surface, "opencode-go."):
		return "OpenCode Go"
	case strings.HasPrefix(surface, "codex."):
		return "Codex"
	case strings.HasPrefix(surface, "gemini."):
		return "Gemini"
	case strings.HasPrefix(surface, "antigravity."):
		return "Antigravity"
	default:
		return "Agent"
	}
}

func runDoctor(args []string, stdout, stderr io.Writer) int {
	jsonOutput := false
	for _, arg := range args {
		if arg == "--json" {
			jsonOutput = true
			continue
		}
		return renderConfigError(jsonOutput, stdout, stderr, fmt.Errorf("unrecognized argument: %s", arg))
	}
	cursor, cursorErr := resolveCredential("", "CURSOR_API_KEY")
	checks := doctor.Run(doctor.Options{
		Cursor: doctor.Credential{Value: cursor, ResolutionError: cursorErr},
	})
	ok := true
	for _, check := range checks {
		ok = ok && check.OK
	}
	if jsonOutput {
		_ = json.NewEncoder(stdout).Encode(map[string]any{"ok": ok, "checks": checks})
	} else {
		for _, check := range checks {
			status := "FAIL"
			if check.OK {
				status = "PASS"
			}
			fmt.Fprintf(stdout, "%s %s: %s\n", status, check.Name, check.Detail)
		}
	}
	if ok {
		return 0
	}
	return 4
}

func runCatalog(args []string, stdout, stderr io.Writer) int {
	if len(args) != 1 || args[0] != "--json" {
		return renderConfigError(false, stdout, stderr, fmt.Errorf("usage: hatch catalog --json"))
	}
	if err := json.NewEncoder(stdout).Encode(provider.SurfaceCatalog()); err != nil {
		fmt.Fprintf(stderr, "Error: encode catalog: %v\n", err)
		return 1
	}
	return 0
}

func readPrompt(args []string, input io.Reader) (string, error) {
	if len(args) == 0 || (len(args) == 1 && args[0] == "-") {
		data, err := io.ReadAll(input)
		if err != nil {
			return "", err
		}
		if strings.TrimSpace(string(data)) == "" {
			return "", fmt.Errorf("Empty prompt")
		}
		return string(data), nil
	}
	return strings.Join(args, " "), nil
}

func renderConfigError(jsonOutput bool, stdout, stderr io.Writer, err error) int {
	if jsonOutput {
		_ = json.NewEncoder(stdout).Encode(map[string]any{
			"ok": false, "status": "config_error", "output": "", "exit_code": 4,
			"duration_ms": 0, "error": configErrorPreview(err), "stderr": nil,
		})
	} else {
		fmt.Fprintf(stderr, "Error: %s\n", configErrorPreview(err))
	}
	return 4
}

func identity(backend, model string) (string, string) {
	switch backend {
	case "claude":
		for _, alias := range []string{"fable", "fable-5.2", "opus", "sonnet", "haiku"} {
			if surfaces["claude"].models[alias] == model {
				return "claude." + alias, "anthropic"
			}
		}
		return "claude." + model, "anthropic"
	case "cursor":
		for alias, configured := range surfaces["cursor"].models {
			if configured == model {
				return "cursor." + alias, "cursor"
			}
		}
		return "cursor.raw", "cursor"
	case "antigravity":
		for alias, configured := range surfaces["gemini"].models {
			if configured == model {
				return "gemini." + alias, "google"
			}
		}
		return "antigravity.raw", "google"
	case "pi", "omp":
		for alias, configured := range surfaces["bedrock"].models {
			if configured == model {
				return "bedrock." + alias, "bedrock"
			}
		}
		for alias, configured := range surfaces["openrouter"].models {
			if configured == model {
				return "openrouter." + alias, "openrouter"
			}
		}
		for alias, configured := range surfaces["opencode-go"].models {
			if configured == model {
				return "opencode-go." + alias, "opencode-go"
			}
		}
		for alias, configured := range surfaces["codex"].models {
			if configured == model {
				return "codex." + alias, "openai"
			}
		}
		if strings.HasPrefix(model, "amazon-bedrock/") {
			return "bedrock.raw", "bedrock"
		}
		if strings.HasPrefix(model, "openrouter/") {
			return "openrouter.raw", "openrouter"
		}
		if strings.HasPrefix(model, "opencode-go/") {
			return "opencode-go.raw", "opencode-go"
		}
		if strings.HasPrefix(model, "openai/") {
			return "codex.raw", "openai"
		}
	}
	return backend + ".raw", "unknown"
}

const Help = `usage: hatch <model> [OPTIONS] "prompt"
       hatch claude <haiku|sonnet|opus|fable|fable-5.2> [OPTIONS] "prompt"
       hatch bedrock <opus|sonnet|astra|sol|terra|luna> [OPTIONS] "prompt"
       hatch codex <astra|sol|terra|luna> [OPTIONS] "prompt"
       hatch cursor <grok|kimi-k3|gemini-4-carbon> [OPTIONS] "prompt"
       hatch gemini [carbon] [OPTIONS] "prompt"
       hatch opencode-go <deepseek> [OPTIONS] "prompt"
       hatch openrouter <deepseek|glm-5.3-flash> [OPTIONS] "prompt"
       hatch expert [OPTIONS] "prompt"
       hatch runs <list|inspect|read|audit|gc> [OPTIONS]
       hatch review MODEL... [--reasoning-effort LEVEL] [--base REF] [--pass-timeout SEC]   (explicit model; two-phase, read-only review; no prompt)

One headless CLI for Claude, Bedrock, Codex, Cursor, Gemini, OpenCode Go, OpenRouter, and expert calls

Coding harness selection:
  --harness pi           Use Pi (default for bedrock, codex, opencode-go, and openrouter)
  --harness omp          Use Oh My Pi instead of Pi

Model-first aliases (each surface bills a different account):
  Claude (personal Max subscription):   haiku, sonnet, opus, fable, fable-5.2
  Bedrock (Zeta AWS, Claude + OpenAI):  opus, sonnet (Claude 5.5);
                                        astra, sol, terra, luna (GPT-6 / GPT-6.1)
  Codex (personal OpenAI, explicit):    hatch codex <astra|sol|terra|luna>
  Cursor (Zeta Cursor subscription):    grok, kimi-k3, gemini-4-carbon
  Gemini (personal Google AI Ultra):    carbon   (Google Antigravity CLI: agy)
  OpenCode Go (flat-rate Go plan):      deepseek (the bare "hatch deepseek" alias)
  OpenRouter (OpenRouter balance):      deepseek, glm-5.3-flash (explicit only)

  OpenAI models are served by Bedrock on the Zeta AWS account, so the bare
  astra/sol/terra/luna aliases resolve to "hatch bedrock" (astra = GPT-6 Astra,
  sol = GPT-6.1 Sol, terra = GPT-6.1 Terra, luna = GPT-6 Luna). Use
  "hatch codex <alias>" only for the personal OpenAI account. Bedrock's Claude
  aliases stay explicit-only, so bare opus/sonnet remain the personal Claude route.
Start here:
  hatch opus "Review this diff"
  hatch gemini "Review this branch"
  hatch astra --harness omp "Review this branch"
  hatch terra "Review this branch"
  hatch grok "Review this branch"
  hatch deepseek "Fix the failing tests"
  hatch glm-5.3-flash "Fix the failing tests"
  hatch claude sonnet "Review this diff"
  hatch cursor grok "Review this branch"
  hatch cursor kimi-k3 "Review this branch"
  hatch cursor gemini-4-carbon "Summarize this code"
  hatch expert "Is this refactor direction sound?"

Follow up on an earlier run (any model, any surface):
  hatch --resume <run-id> "Now fix what you found"      same model, same directory
  hatch opus --resume <run-id> "Implement the fix"      a different model of the same account
  Every result prints its run id and a "follow_up" command. The new run receives the
  earlier task and answer as context. Work (Zeta: bedrock, cursor) and personal runs
  never cross; continue within the same account.

Common options:
  -C, --cwd DIR        Working directory for the agent
  --title TEXT         Short single-line task title (up to 160 bytes)
  --caller-session ID  Originating conversation ID (or HATCH_CALLER_SESSION_ID)
  --caller-request ID  Group runs from one request (or HATCH_CALLER_REQUEST_ID)
  --max-output-bytes N  Inline answer preview, 256..32768 bytes (default: 8192)
  -t, --timeout SEC    Hard timeout (default: 1800)
  --harness NAME       Select pi or omp for bedrock/codex/opencode-go/openrouter
  --reasoning-effort LEVEL  none|low|medium|high|xhigh|max
  -r, --resume RUN_ID  Continue an earlier Hatch run (see above)
  --read-only          No file edits and no builds, tests or other programs (every surface)
  --json               Emit exactly one JSON document on stdout
  --advanced-help      Show raw/backend-specific flags

Other commands:
  hatch catalog --json
  hatch doctor [--json]
  hatch runs list [--all|--session ID|--under DIR] [--query TEXT] [--limit N] [--json]
  hatch runs inspect <run-id> [--files] [--json]
  hatch runs read <run-id> [--part result|request|stdout|stderr] [--offset N] [--limit N] [--json]
  hatch runs audit [--minimum-total N] [--minimum-surface N] [--json]
  hatch runs gc [--apply] [--json]
`

const AdvancedHelp = Help + `
Advanced raw/backend options:
  -b, --backend NAME
  --model MODEL
  --skip-git-repo-check
  --output-format <text|json|stream-json>
  --include-partial-messages
  --api-key KEY
  --automation

Credential helper:
  HATCH_CREDENTIAL_HELPER must name an explicitly configured absolute executable.

Provider environment:
  OPENAI_API_KEY, OPENROUTER_API_KEY, OPENCODE_API_KEY, CURSOR_API_KEY, AWS_PROFILE, AWS_REGION
`
