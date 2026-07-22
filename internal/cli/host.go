package cli

import (
	"os"
	"path/filepath"
	"strings"

	"github.com/cipher982/hatch/internal/provider"
)

type ExecutionContext struct {
	InContainer  bool
	HomeWritable bool
	Home         string
}

func (c ExecutionContext) EffectiveHome() string {
	if c.InContainer && !c.HomeWritable {
		return os.TempDir()
	}
	if c.Home != "" {
		return c.Home
	}
	return os.TempDir()
}

func DetectContext() ExecutionContext {
	home := os.Getenv("HOME")
	if home == "" {
		home = "/root"
	}
	return ExecutionContext{
		InContainer:  detectContainer(),
		HomeWritable: directoryWritable(home),
		Home:         os.Getenv("HOME"),
	}
}

func detectContainer() bool {
	if pathExists("/.dockerenv") || pathExists("/run/.containerenv") {
		return true
	}
	data, err := os.ReadFile("/proc/1/cgroup")
	if err != nil {
		return false
	}
	value := string(data)
	return strings.Contains(value, "docker") || strings.Contains(value, "containerd") || strings.Contains(value, "kubepods")
}

func directoryWritable(path string) bool {
	file, err := os.CreateTemp(path, ".hatch-write-test-*")
	if err != nil {
		return false
	}
	name := file.Name()
	closeErr := file.Close()
	removeErr := os.Remove(name)
	return closeErr == nil && removeErr == nil
}

func applyHostContext(invocation *provider.Invocation, backend string, ctx ExecutionContext) {
	if invocation.SetEnv == nil {
		invocation.SetEnv = map[string]string{}
	}
	if ctx.InContainer && !ctx.HomeWritable {
		invocation.SetEnv["HOME"] = ctx.EffectiveHome()
	}
	if backend == "opencode" {
		applyObservatoryTrust(invocation.SetEnv, ctx)
	}
}

func applyObservatoryTrust(target map[string]string, ctx ExecutionContext) {
	ca := observatoryCA(target, ctx)
	if ca == "" {
		return
	}
	for _, name := range []string{"NODE_EXTRA_CA_CERTS", "CODEX_CA_CERTIFICATE"} {
		if configured := target[name]; configured != "" && regularFile(configured) {
			continue
		}
		if inherited := os.Getenv(name); inherited != "" && regularFile(inherited) {
			target[name] = inherited
			continue
		}
		target[name] = ca
	}
}

func observatoryCA(target map[string]string, ctx ExecutionContext) string {
	for _, name := range []string{"NODE_EXTRA_CA_CERTS", "CODEX_CA_CERTIFICATE"} {
		for _, value := range []string{target[name], os.Getenv(name)} {
			if value != "" && regularFile(value) {
				return value
			}
		}
	}
	path := filepath.Join(ctx.EffectiveHome(), ".local", "state", "agent-observatory", "ca", "observatory-ca.pem")
	if regularFile(path) {
		return path
	}
	return ""
}

func regularFile(path string) bool {
	info, err := os.Stat(path)
	return err == nil && info.Mode().IsRegular()
}

func pathExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}
