package provider

import (
	"encoding/json"
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

func TestReadOnlyOpenCodePermissionPolicy(t *testing.T) {
	invocation := readOnlyBuild(t, "opencode", "openrouter/deepseek/deepseek-v4.1-flash")
	var config struct {
		Provider   map[string]any             `json:"provider"`
		Permission map[string]json.RawMessage `json:"permission"`
	}
	if err := json.Unmarshal(invocation.OpenCodeConfigJSON, &config); err != nil {
		t.Fatalf("config is not valid JSON: %v\n%s", err, invocation.OpenCodeConfigJSON)
	}
	if config.Provider["openrouter"] == nil {
		t.Fatalf("read-only config dropped the provider routing pin: %s", invocation.OpenCodeConfigJSON)
	}
	text := string(invocation.OpenCodeConfigJSON)
	// OpenCode evaluates rules last-match-wins in key order: each catch-all deny
	// must come before every rule it is meant to be narrowed by.
	permission := text[strings.Index(text, `"permission"`):]
	if !strings.HasPrefix(permission, `"permission":{"*":"deny",`) {
		t.Fatalf("permission policy must open with the catch-all deny: %s", permission)
	}
	bash := permission[strings.Index(permission, `"bash":{`):]
	if !strings.HasPrefix(bash, `"bash":{"*":"deny",`) {
		t.Fatalf("bash policy must open with the catch-all deny: %s", bash)
	}
	for _, allowed := range []string{`"git diff *":"allow"`, `"git log":"allow"`, `"rg *":"allow"`} {
		if !strings.Contains(bash, allowed) {
			t.Fatalf("bash allowlist lacks %s", allowed)
		}
	}
	for _, denied := range []string{`"cargo`, `"make`, `"pytest`, `"uv `, `"npm`, `"bun`, `"python`, `"sed`, `"find`, `"rm`, `"git push`, `"git commit`, `"git checkout`} {
		if strings.Contains(bash, denied) {
			t.Fatalf("bash allowlist must not name %s: %s", denied, bash)
		}
	}
	// The flags that make an inspection command run a program or write a file
	// are denied after the allows that would otherwise match them.
	if strings.LastIndex(bash, `"git *--output*":"deny"`) < strings.LastIndex(bash, `"allow"`) {
		t.Fatalf("flag denies must follow the allowlist: %s", bash)
	}
	// A redirect turns any allowed command into a file write, and OpenCode
	// matches the allow rules without it. Order is the policy: `2>` denied, the
	// two harmless stderr forms allowed back, then every stdout redirect denied
	// after them, so `cat x > y 2>/dev/null` still fails.
	order := []string{`"*2>*":"deny"`, `"*2>/dev/null*":"allow"`, `"*2>&1*":"allow"`, `"* >*":"deny"`, `"*>>*":"deny"`, `"*&>*":"deny"`, `"*1>*":"deny"`}
	last := strings.LastIndex(bash, `"cat *":"allow"`)
	for _, want := range order {
		at := strings.Index(bash, want)
		if at < 0 || at < last {
			t.Fatalf("redirect rule %s missing or before the allowlist: %s", want, bash)
		}
		last = at
	}
	if strings.Contains(bash, `"*>*"`) {
		t.Fatalf("a bare *>* deny would refuse rg patterns containing -> and =>: %s", bash)
	}
	if !strings.Contains(bash, `"git -C * log *":"allow"`) {
		t.Fatalf("git -C form missing: %s", bash)
	}
	for _, want := range []string{`"edit":"deny"`, `"webfetch":"deny"`, `"task":"deny"`, `"*.env":"deny"`} {
		if !strings.Contains(permission, want) {
			t.Fatalf("permission policy lacks %s: %s", want, permission)
		}
	}
	if !slices.Contains(invocation.Argv, "--dangerously-skip-permissions") {
		t.Fatalf("read-only OpenCode still needs the auto-approve flag so nothing asks: %v", invocation.Argv)
	}
}

func TestOpenCodeWithoutReadOnlyCarriesNoPermissionPolicy(t *testing.T) {
	invocation, err := Build(Request{Backend: "opencode", Model: "openrouter/deepseek/deepseek-v4.1-flash", Prompt: "prompt", APIKey: "fake"})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(invocation.OpenCodeConfigJSON), "permission") || invocation.ReadOnlyMechanism != "" {
		t.Fatalf("ordinary run picked up read-only state: %s", invocation.OpenCodeConfigJSON)
	}
	bare, err := Build(Request{Backend: "opencode", Model: "amazon-bedrock/global.anthropic.claude-opus-5-5", Prompt: "prompt", ReadOnly: true})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(string(bare.OpenCodeConfigJSON), `{"permission":{"*":"deny"`) {
		t.Fatalf("read-only run with no provider pin needs the policy on its own: %s", bare.OpenCodeConfigJSON)
	}
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
	pi := readOnlyBuild(t, "pi", "openrouter/deepseek/deepseek-v4.1-flash")
	if got := argvValue(pi.Argv, "--tools"); got != "read,grep,find,ls" {
		t.Fatalf("pi --tools = %q", got)
	}
	omp := readOnlyBuild(t, "omp", "openrouter/deepseek/deepseek-v4.1-flash")
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
