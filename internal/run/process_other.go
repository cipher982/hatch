//go:build !darwin && !linux

package run

import "os/exec"

func configureProcess(cmd *exec.Cmd) {}

func killRunProcesses(cmd *exec.Cmd, runID string) (string, int, error) {
	if cmd.Process == nil {
		return "", 0, nil
	}
	return "KILL", 0, cmd.Process.Kill()
}

func sweepRunProcesses(string) int { return 0 }

func processStartIdentity(int) string { return "" }
func processAlive(int) (bool, bool)   { return false, false }
