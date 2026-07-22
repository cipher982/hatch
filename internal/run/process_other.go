//go:build !darwin && !linux

package run

import "os/exec"

func configureProcess(cmd *exec.Cmd) {}

func killProcessGroup(cmd *exec.Cmd) (string, error) {
	if cmd.Process == nil {
		return "", nil
	}
	return "KILL", cmd.Process.Kill()
}

func processStartIdentity(int) string { return "" }
func processAlive(int) (bool, bool)   { return false, false }
