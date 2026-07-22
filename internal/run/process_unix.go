//go:build darwin || linux

package run

import (
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"syscall"
)

func configureProcess(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
}

func killProcessGroup(cmd *exec.Cmd) error {
	if cmd.Process == nil {
		return nil
	}
	return syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
}

func processStartIdentity(pid int) string {
	if runtime.GOOS == "linux" {
		data, err := os.ReadFile(fmt.Sprintf("/proc/%d/stat", pid))
		if err != nil {
			return ""
		}
		end := strings.LastIndexByte(string(data), ')')
		if end < 0 {
			return ""
		}
		fields := strings.Fields(string(data)[end+1:])
		if len(fields) <= 19 {
			return ""
		}
		return "linux-proc-startticks:" + fields[19]
	}
	output, err := exec.Command("/bin/ps", "-o", "lstart=", "-p", fmt.Sprint(pid)).Output()
	if err != nil || strings.TrimSpace(string(output)) == "" {
		return ""
	}
	return "darwin-ps-lstart:" + strings.Join(strings.Fields(string(output)), " ")
}
