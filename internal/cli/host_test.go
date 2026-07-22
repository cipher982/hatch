package cli

import (
	"os"
	"path/filepath"
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
	applyHostContext(&invocation, "opencode", ExecutionContext{Home: home, HomeWritable: true})
	if invocation.SetEnv["NODE_EXTRA_CA_CERTS"] != ca || invocation.SetEnv["CODEX_CA_CERTIFICATE"] != ca {
		t.Fatalf("observatory env = %#v", invocation.SetEnv)
	}

	containerInvocation := provider.Invocation{}
	applyHostContext(&containerInvocation, "gemini", ExecutionContext{InContainer: true, HomeWritable: false, Home: "/readonly"})
	if containerInvocation.SetEnv["HOME"] != os.TempDir() {
		t.Fatalf("container env = %#v", containerInvocation.SetEnv)
	}
}
