package provider

import (
	"slices"
	"strings"
	"testing"
)

func readOnlyBuild(t *testing.T, backend, model string) Invocation {
	t.Helper()
	invocation, err := Build(Request{Backend: backend, Model: model, Prompt: "prompt", APIKey: "fake", ReadOnly: true})
	if err != nil {
		t.Fatalf("%s: %v", backend, err)
	}
	if invocation.ReadOnlyMechanism == "" {
		t.Fatalf("%s: read-only invocation does not name its mechanism", backend)
	}
	return invocation
}

// argvValue returns the argument that follows flag.
func argvValue(argv []string, flag string) string {
	if i := slices.Index(argv, flag); i >= 0 && i+1 < len(argv) {
		return argv[i+1]
	}
	return ""
}

func TestReadOnlyClaudeHasNoShellOrEditTools(t *testing.T) {
	invocation := readOnlyBuild(t, "claude", "opus")
	if slices.Contains(invocation.Argv, "--dangerously-skip-permissions") {
		t.Fatalf("read-only claude must not bypass permissions: %v", invocation.Argv)
	}
	if got := argvValue(invocation.Argv, "--tools"); got != "Read,Grep,Glob" {
		t.Fatalf("--tools = %q, want the read-only set: %v", got, invocation.Argv)
	}
	if got := argvValue(invocation.Argv, "--permission-mode"); got != "dontAsk" {
		t.Fatalf("--permission-mode = %q", got)
	}
	if !slices.Contains(invocation.Argv, "--strict-mcp-config") {
		t.Fatalf("read-only claude must not load MCP servers: %v", invocation.Argv)
	}
	ordinary, err := Build(Request{Backend: "claude", Model: "opus", Prompt: "prompt"})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Contains(ordinary.Argv, "--dangerously-skip-permissions") || argvValue(ordinary.Argv, "--tools") != "default" {
		t.Fatalf("ordinary claude run changed: %v", ordinary.Argv)
	}
}

func TestReadOnlyPiAndOmpAllowlistTools(t *testing.T) {
	pi := readOnlyBuild(t, "pi", "openrouter/deepseek/deepseek-v4.1-flash-1017")
	if got := argvValue(pi.Argv, "--tools"); got != "read,grep,find,ls" {
		t.Fatalf("pi --tools = %q", got)
	}
	omp := readOnlyBuild(t, "omp", "openrouter/deepseek/deepseek-v4.1-flash-1017")
	if got := argvValue(omp.Argv, "--tools"); got != "read,grep,glob" {
		t.Fatalf("omp --tools = %q", got)
	}
}

func TestReadOnlyCursorUsesAskModeWithoutForce(t *testing.T) {
	invocation := readOnlyBuild(t, "cursor", "grok-4.7-high")
	if argvValue(invocation.Argv, "--mode") != "ask" || slices.Contains(invocation.Argv, "--force") {
		t.Fatalf("cursor read-only argv = %v", invocation.Argv)
	}
	if invocation.RedactedArgv[len(invocation.RedactedArgv)-1] != "<prompt>" {
		t.Fatalf("prompt not redacted last: %v", invocation.RedactedArgv)
	}
	ordinary, err := Build(Request{Backend: "cursor", Model: "grok-4.7-high", Prompt: "prompt"})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Contains(ordinary.Argv, "--force") || slices.Contains(ordinary.Argv, "--mode") {
		t.Fatalf("ordinary cursor run changed: %v", ordinary.Argv)
	}
}

func TestReadOnlyRawCodexUsesReadOnlySandbox(t *testing.T) {
	invocation := readOnlyBuild(t, "codex", "")
	if argvValue(invocation.Argv, "--sandbox") != "read-only" || slices.Contains(invocation.Argv, "--dangerously-bypass-approvals-and-sandbox") {
		t.Fatalf("codex read-only argv = %v", invocation.Argv)
	}
}

func TestReadOnlyRefusedWhereNoMechanismExists(t *testing.T) {
	_, err := Build(Request{Backend: "gemini", Model: "gemini-3-pro-preview", Prompt: "prompt", ReadOnly: true})
	if err == nil || !strings.Contains(err.Error(), "--read-only is not supported") {
		t.Fatalf("raw gemini must refuse read-only rather than run unrestricted: %v", err)
	}
}

func globMatch(pattern, text string) bool {
	if pattern == "" {
		return text == ""
	}
	switch pattern[0] {
	case '*':
		for i := 0; i <= len(text); i++ {
			if globMatch(pattern[1:], text[i:]) {
				return true
			}
		}
		return false
	case '?':
		return text != "" && globMatch(pattern[1:], text[1:])
	default:
		return text != "" && text[0] == pattern[0] && globMatch(pattern[1:], text[1:])
	}
}
