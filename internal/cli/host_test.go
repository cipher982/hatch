package cli

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cipher982/hatch/internal/provider"
)

func TestExecutionContextEffectiveHome(t *testing.T) {
	tests := []struct {
		name string
		ctx  ExecutionContext
		want string
	}{
		{"laptop", ExecutionContext{Home: "/home/person", HomeWritable: true}, "/home/person"},
		{"writable container", ExecutionContext{InContainer: true, HomeWritable: true, Home: "/home/app"}, "/home/app"},
		{"readonly container", ExecutionContext{InContainer: true, HomeWritable: false, Home: "/readonly"}, os.TempDir()},
		{"missing home", ExecutionContext{HomeWritable: true}, os.TempDir()},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := test.ctx.EffectiveHome(); got != test.want {
				t.Fatalf("EffectiveHome() = %q, want %q", got, test.want)
			}
		})
	}
}

func TestDirectoryWritable(t *testing.T) {
	if !directoryWritable(t.TempDir()) {
		t.Fatal("temporary directory should be writable")
	}
	if directoryWritable(filepath.Join(t.TempDir(), "missing")) {
		t.Fatal("missing directory should not be writable")
	}
}

func TestApplyHostContext(t *testing.T) {
	t.Setenv("NODE_EXTRA_CA_CERTS", "")
	t.Setenv("CODEX_CA_CERTIFICATE", "")
	home := t.TempDir()
	ca := filepath.Join(home, ".local", "state", "agent-observatory", "ca", "observatory-ca.pem")
	if err := os.MkdirAll(filepath.Dir(ca), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(ca, []byte("ca"), 0o600); err != nil {
		t.Fatal(err)
	}
	invocation := provider.Invocation{}
	if err := applyHostContext(&invocation, "pi", ExecutionContext{Home: home, HomeWritable: true}); err != nil {
		t.Fatal(err)
	}
	if invocation.SetEnv["NODE_EXTRA_CA_CERTS"] != ca || invocation.SetEnv["CODEX_CA_CERTIFICATE"] != ca {
		t.Fatalf("observatory env = %#v", invocation.SetEnv)
	}

	containerInvocation := provider.Invocation{}
	if err := applyHostContext(&containerInvocation, "gemini", ExecutionContext{InContainer: true, HomeWritable: false, Home: "/readonly"}); err != nil {
		t.Fatal(err)
	}
	if containerInvocation.SetEnv["HOME"] != os.TempDir() {
		t.Fatalf("container env = %#v", containerInvocation.SetEnv)
	}
}

func TestApplyDCGClaudeOverlay(t *testing.T) {
	home := t.TempDir()
	binary := filepath.Join(home, ".local", "bin", "dcg")
	if err := os.MkdirAll(filepath.Dir(binary), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(binary, []byte("#!/bin/sh\n"), 0o700); err != nil {
		t.Fatal(err)
	}
	invocation, err := provider.Build(provider.Request{Backend: "claude", Model: "haiku", Prompt: "p"})
	if err != nil {
		t.Fatal(err)
	}
	if err := applyDCG(&invocation, "claude", ExecutionContext{Home: home, HomeWritable: true}); err != nil {
		t.Fatal(err)
	}
	index := flagValueIndex(invocation.Argv, "--settings")
	if index < 0 || !strings.Contains(invocation.Argv[index+1], binary) {
		t.Fatalf("Claude args = %#v", invocation.Argv)
	}
	if len(invocation.RedactedArgv) != len(invocation.Argv) {
		t.Fatalf("Claude redaction drift: argv=%#v redacted=%#v", invocation.Argv, invocation.RedactedArgv)
	}
}

func TestApplyDCGLoadsGuardExtensionIntoPi(t *testing.T) {
	home := t.TempDir()
	binary := filepath.Join(home, ".local", "bin", "dcg")
	if err := os.MkdirAll(filepath.Dir(binary), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(binary, []byte("#!/bin/sh\n"), 0o700); err != nil {
		t.Fatal(err)
	}
	invocation, err := provider.Build(provider.Request{Backend: "pi", Model: "openai/gpt-6.1-sol", Prompt: "p", APIKey: "key"})
	if err != nil {
		t.Fatal(err)
	}
	if err := applyDCG(&invocation, "pi", ExecutionContext{Home: home, HomeWritable: true}); err != nil {
		t.Fatal(err)
	}
	if invocation.PiDCGBinary != binary {
		t.Fatalf("Pi run does not carry the DCG binary: %#v", invocation)
	}
}

func TestApplyDCGRequiredFailsClosed(t *testing.T) {
	home := t.TempDir()
	declaration := filepath.Join(home, "git", "me", "config", "dcg", "release.json")
	if err := os.MkdirAll(filepath.Dir(declaration), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(declaration, []byte("{\"required\":true}\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	invocation := provider.Invocation{}
	if err := applyDCG(&invocation, "claude", ExecutionContext{Home: home, HomeWritable: true}); err == nil || !strings.Contains(err.Error(), "agents guard install") {
		t.Fatalf("error = %v", err)
	}
}
