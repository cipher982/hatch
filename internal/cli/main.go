package cli

import (
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
	if len(args) > 0 && args[0] == "expert" {
		return runExpert(args[1:], stdin, stdout, stderr)
	}
	if len(args) > 0 && args[0] == "runs" {
		return runRuns(args[1:], stdout, stderr)
	}
	if len(args) > 0 && args[0] == "doctor" {
		return runDoctor(args[1:], stdout, stderr)
	}
	request, err := Parse(args, stdoutTTY)
	if err != nil {
		return renderConfigError(request.JSON || !stdoutTTY, stdout, stderr, err)
	}
	if request.Help {
		fmt.Fprint(stdout, Help)
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
	if request.Backend == "" {
		return renderConfigError(request.JSON, stdout, stderr, fmt.Errorf("No default model is configured; choose an explicit provider"))
	}
	if !oneOf(request.Backend, "claude", "cursor", "bedrock", "codex", "gemini", "opencode") {
		return renderConfigError(request.JSON, stdout, stderr, fmt.Errorf("invalid backend %q. Choose one of: claude, cursor, bedrock, codex, gemini", request.Backend))
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
	if request.Backend == "opencode" && request.SkipGitRepoCheck {
		return renderConfigError(request.JSON, stdout, stderr, fmt.Errorf("--skip-git-repo-check is not supported for surfaced providers"))
	}
	if request.ReasoningEffort != "" && request.Backend != "codex" && !(request.Backend == "opencode" && strings.HasPrefix(request.Model, "openai/")) {
		return renderConfigError(request.JSON, stdout, stderr, fmt.Errorf("--reasoning-effort only works with Codex models"))
	}
	prompt, err := readPrompt(request.PromptArgs, stdin)
	if err != nil {
		return renderConfigError(request.JSON, stdout, stderr, err)
	}
	apiKey := request.APIKey
	credentialEnvironment := ""
	if strings.HasPrefix(request.Model, "openrouter/") {
		credentialEnvironment = "OPENROUTER_API_KEY"
	} else if strings.HasPrefix(request.Model, "openai/") || request.Backend == "codex" {
		credentialEnvironment = "OPENAI_API_KEY"
	}
	if credentialEnvironment != "" {
		apiKey, err = resolveCredential(apiKey, credentialEnvironment)
		if err != nil {
			return renderConfigError(request.JSON, stdout, stderr, err)
		}
	}
	if strings.HasPrefix(request.Model, "openrouter/") && apiKey == "" {
		return renderConfigError(request.JSON, stdout, stderr, fmt.Errorf("OPENROUTER_API_KEY not set and no credential helper is configured"))
	}
	if (strings.HasPrefix(request.Model, "openai/") || request.Backend == "codex") && apiKey == "" {
		return renderConfigError(request.JSON, stdout, stderr, fmt.Errorf("OPENAI_API_KEY not set and no credential helper is configured"))
	}
	invocation, err := provider.Build(provider.Request{
		Backend: request.Backend, Model: request.Model, Prompt: prompt, CWD: request.CWD,
		ReasoningEffort: request.ReasoningEffort, OutputFormat: request.OutputFormat, APIKey: apiKey,
		Resume: request.Resume, SkipGitRepoCheck: request.SkipGitRepoCheck,
		IncludePartialMessages: request.IncludePartialMessages,
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
	root, err := runner.DefaultRoot()
	if err != nil {
		return renderConfigError(request.JSON, stdout, stderr, err)
	}
	surface, providerName := identity(request.Backend, request.Model)
	credentialNames := []string{}
	if strings.HasPrefix(request.Model, "openrouter/") {
		credentialNames = append(credentialNames, "OPENROUTER_API_KEY")
	} else if strings.HasPrefix(request.Model, "openai/") || request.Backend == "codex" {
		credentialNames = append(credentialNames, "OPENAI_API_KEY")
	} else if request.Backend == "cursor" && apiKey != "" {
		credentialNames = append(credentialNames, "CURSOR_API_KEY")
	}
	coordinator := runner.NewCoordinator(runner.NewStore(root))
	result := coordinator.Execute(runner.Request{
		Surface: surface, Provider: providerName, Model: request.Model, CWD: request.CWD,
		Prompt: prompt, Timeout: time.Duration(request.TimeoutSeconds) * time.Second,
		Invocation: invocation, CredentialNames: credentialNames, Automation: request.Automation,
		ProgressLabel: progressLabel(surface), Progress: func(message string) { fmt.Fprintln(stderr, message) },
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
		fmt.Fprintf(stderr, "Error: %s\n", strings.TrimRight(*result.Error, "\n"))
	}
	return result.CLIExitCode()
}

func progressLabel(surface string) string {
	switch {
	case strings.HasPrefix(surface, "claude."):
		return "Claude"
	case strings.HasPrefix(surface, "bedrock."):
		return "Claude"
	case strings.HasPrefix(surface, "cursor."):
		return "Cursor"
	case strings.HasPrefix(surface, "openrouter."):
		return "OpenRouter"
	case strings.HasPrefix(surface, "codex."):
		return "Codex"
	case strings.HasPrefix(surface, "gemini."):
		return "Gemini"
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
	checks := doctor.Run()
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
			"duration_ms": 0, "error": err.Error(), "stderr": nil,
		})
	} else {
		fmt.Fprintf(stderr, "Error: %v\n", err)
	}
	return 4
}

func identity(backend, model string) (string, string) {
	switch backend {
	case "claude":
		return "claude." + model, "anthropic"
	case "cursor":
		return "cursor.grok", "cursor"
	case "gemini":
		return "gemini.raw", "google"
	case "opencode":
		if strings.HasPrefix(model, "openrouter/") {
			return "openrouter." + strings.TrimPrefix(model[strings.LastIndex(model, "/"):], "/"), "openrouter"
		}
		if strings.HasPrefix(model, "openai/") {
			return "codex." + strings.TrimPrefix(model, "openai/gpt-5.6-"), "openai"
		}
	}
	return backend + ".raw", "unknown"
}

const Help = `usage: hatch claude <haiku|sonnet|opus|fable> [OPTIONS] "prompt"
       hatch codex <sol|terra|luna> [OPTIONS] "prompt"
       hatch cursor grok [OPTIONS] "prompt"
       hatch openrouter <deepseek-v4-pro|kimi-k3> [OPTIONS] "prompt"
       hatch expert [OPTIONS] "prompt"

One headless CLI for Claude, Codex, Cursor, Gemini, OpenRouter, and expert calls
`
