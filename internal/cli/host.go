package cli

import (
	"encoding/json"
	"fmt"
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

func applyHostContext(invocation *provider.Invocation, backend string, ctx ExecutionContext) error {
	if invocation.SetEnv == nil {
		invocation.SetEnv = map[string]string{}
	}
	if ctx.InContainer && !ctx.HomeWritable {
		invocation.SetEnv["HOME"] = ctx.EffectiveHome()
	}
	if backend == "opencode" {
		applyObservatoryTrust(invocation.SetEnv, ctx)
	}
	return applyDCG(invocation, backend, ctx)
}

func applyDCG(invocation *provider.Invocation, backend string, ctx ExecutionContext) error {
	home := ctx.Home
	if home == "" {
		home = ctx.EffectiveHome()
	}
	required, err := dcgRequired(home)
	if err != nil {
		return err
	}
	binary := filepath.Join(home, ".local", "bin", "dcg")
	info, binaryErr := os.Stat(binary)
	if binaryErr != nil || !info.Mode().IsRegular() || info.Mode().Perm()&0o111 == 0 {
		if required {
			return fmt.Errorf("Agent Home requires DCG but %s is missing or not executable; run `agents guard install`", binary)
		}
		return nil
	}
	switch backend {
	case "claude", "bedrock":
		settings, err := json.Marshal(map[string]any{
			"hooks": map[string]any{"PreToolUse": []any{map[string]any{
				"matcher": "Bash", "hooks": []any{map[string]any{"type": "command", "command": binary}},
			}}},
		})
		if err != nil {
			return err
		}
		index := flagValueIndex(invocation.Argv, "--setting-sources")
		if index < 0 {
			return fmt.Errorf("Claude invocation lacks --setting-sources")
		}
		insert := index + 2
		invocation.Argv = append(invocation.Argv[:insert], append([]string{"--settings", string(settings)}, invocation.Argv[insert:]...)...)
	case "opencode":
		root := filepath.Join(home, ".config", "hatch", "dcg")
		source := filepath.Join(home, "git", "me", "config", "dcg", "opencode-plugin.js")
		plugin := filepath.Join(root, "opencode", "plugins", "dcg-guard.js")
		isolatedBinary := filepath.Join(root, "bin", "dcg")
		if !exactSymlink(plugin, source) || !regularNonSymlink(source) || !exactSymlink(isolatedBinary, binary) {
			if required {
				return fmt.Errorf("Agent Home requires the reviewed OpenCode DCG plugin at %s; run `agents guard install`", plugin)
			}
			return nil
		}
		invocation.SetEnv["XDG_CONFIG_HOME"] = filepath.Join(root, "xdg")
		invocation.SetEnv["XDG_DATA_HOME"] = filepath.Join(root, "data")
		invocation.SetEnv["XDG_CACHE_HOME"] = filepath.Join(root, "cache")
		invocation.SetEnv["XDG_STATE_HOME"] = filepath.Join(root, "state")
		invocation.SetEnv["OPENCODE_CONFIG_DIR"] = filepath.Join(root, "opencode")
		invocation.SetEnv["OPENCODE_DISABLE_PROJECT_CONFIG"] = "1"
		invocation.Argv = removeArg(invocation.Argv, "--pure")
	}
	return nil
}

func dcgRequired(home string) (bool, error) {
	path := filepath.Join(home, "git", "me", "config", "dcg", "release.json")
	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return false, nil
	}
	if err != nil {
		return false, fmt.Errorf("invalid Agent Home DCG declaration: %s", path)
	}
	var declaration map[string]json.RawMessage
	if json.Unmarshal(data, &declaration) != nil {
		return false, fmt.Errorf("invalid Agent Home DCG declaration: %s must contain boolean `required`", path)
	}
	raw, ok := declaration["required"]
	if !ok {
		return false, fmt.Errorf("invalid Agent Home DCG declaration: %s must contain boolean `required`", path)
	}
	var required bool
	if json.Unmarshal(raw, &required) != nil {
		return false, fmt.Errorf("invalid Agent Home DCG declaration: %s must contain boolean `required`", path)
	}
	return required, nil
}

func exactSymlink(path, target string) bool {
	info, err := os.Lstat(path)
	if err != nil || info.Mode()&os.ModeSymlink == 0 {
		return false
	}
	actual, err := os.Readlink(path)
	return err == nil && actual == target
}

func regularNonSymlink(path string) bool {
	info, err := os.Lstat(path)
	return err == nil && info.Mode().IsRegular()
}

func flagValueIndex(args []string, flag string) int {
	for index, arg := range args {
		if arg == flag {
			return index
		}
	}
	return -1
}

func removeArg(args []string, remove string) []string {
	result := make([]string, 0, len(args))
	for _, arg := range args {
		if arg != remove {
			result = append(result, arg)
		}
	}
	return result
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
