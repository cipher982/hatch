package cli

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

const storageAdmissionTimeout = 5 * time.Second

// applyStorageAdmission lets an explicitly configured host reserve local
// capacity before Hatch allocates a run or starts a provider. Hatch owns no
// disk policy itself; the host hook is the authority.
func applyStorageAdmission(ctx ExecutionContext) (bool, error) {
	home := ctx.Home
	if home == "" {
		home = ctx.EffectiveHome()
	}
	required, err := storageAdmissionRequired(home)
	if err != nil {
		return false, err
	}
	hook := filepath.Join(home, ".local", "bin", "agent-storage-admit")
	info, statErr := os.Stat(hook)
	if statErr != nil || !info.Mode().IsRegular() || info.Mode().Perm()&0o111 == 0 {
		if required {
			return false, fmt.Errorf("Agent Home requires storage admission but %s is missing or not executable; reinstall the cinder storage guard", hook)
		}
		return false, nil
	}

	commandContext, cancel := context.WithTimeout(context.Background(), storageAdmissionTimeout)
	defer cancel()
	command := exec.CommandContext(commandContext, hook)
	command.Env = append(os.Environ(), "AGENT_STORAGE_SOURCE=hatch", fmt.Sprintf("AGENT_STORAGE_PARENT_PID=%d", os.Getpid()))
	output, runErr := command.CombinedOutput()
	if commandContext.Err() == context.DeadlineExceeded {
		return false, fmt.Errorf("storage admission check timed out after %s", storageAdmissionTimeout)
	}
	if runErr != nil {
		detail := strings.TrimSpace(string(output))
		if detail == "" {
			detail = runErr.Error()
		}
		return false, fmt.Errorf("storage admission denied: %s", detail)
	}
	return true, nil
}

func storageAdmissionRequired(home string) (bool, error) {
	path := filepath.Join(home, "git", "me", "config", "storage-admission", "release.json")
	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return false, nil
	}
	if err != nil {
		return false, fmt.Errorf("invalid Agent Home storage admission declaration: %s", path)
	}
	var declaration map[string]json.RawMessage
	if json.Unmarshal(data, &declaration) != nil {
		return false, fmt.Errorf("invalid Agent Home storage admission declaration: %s must contain boolean `required`", path)
	}
	raw, ok := declaration["required"]
	if !ok {
		return false, fmt.Errorf("invalid Agent Home storage admission declaration: %s must contain boolean `required`", path)
	}
	var required bool
	if json.Unmarshal(raw, &required) != nil {
		return false, fmt.Errorf("invalid Agent Home storage admission declaration: %s must contain boolean `required`", path)
	}
	return required, nil
}
