//go:build darwin || linux

package run

import (
	"bufio"
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"
)

func configureProcess(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
}

// killRunProcesses ends everything a provider run started. Killing the process
// group is not enough: a provider that launches each shell command detached
// (OpenCode does) puts it in its own session, so a hung `cargo test` outlives
// its parent. The run's descendants are read from the process table before the
// group dies, and any survivor that still carries the run's HATCH_RUN_ID (every
// child inherits it) is swept afterwards, which also catches orphans already
// reparented to init. It returns the signal used and how many processes
// outside the group were killed.
func killRunProcesses(cmd *exec.Cmd, runID string) (string, int, error) {
	if cmd.Process == nil {
		return "", 0, nil
	}
	tree := descendantPIDs(cmd.Process.Pid)
	err := syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
	killed := killPIDs(tree)
	killed += sweepRunProcesses(runID)
	return "SIGKILL", killed, err
}

// sweepRunProcesses kills processes still carrying the run's HATCH_RUN_ID after
// the provider has exited or been killed, and returns how many it killed.
func sweepRunProcesses(runID string) int {
	return killPIDs(processesWithRunID(runID))
}

func killPIDs(pids []int) int {
	self, killed := os.Getpid(), 0
	for _, pid := range pids {
		if pid <= 1 || pid == self {
			continue
		}
		if syscall.Kill(pid, syscall.SIGKILL) == nil {
			killed++
		}
	}
	return killed
}

// descendantPIDs returns every process below root in the process tree.
func descendantPIDs(root int) []int {
	output, err := exec.Command("/bin/ps", "-A", "-o", "pid=,ppid=").Output()
	if err != nil {
		return nil
	}
	children := map[int][]int{}
	scanner := bufio.NewScanner(bytes.NewReader(output))
	for scanner.Scan() {
		fields := strings.Fields(scanner.Text())
		if len(fields) != 2 {
			continue
		}
		pid, pidErr := strconv.Atoi(fields[0])
		ppid, ppidErr := strconv.Atoi(fields[1])
		if pidErr == nil && ppidErr == nil {
			children[ppid] = append(children[ppid], pid)
		}
	}
	var found []int
	queue := []int{root}
	for len(queue) > 0 {
		next := children[queue[0]]
		queue = queue[1:]
		found = append(found, next...)
		queue = append(queue, next...)
	}
	return found
}

// processesWithRunID lists this user's processes whose environment carries
// HATCH_RUN_ID=<runID>.
func processesWithRunID(runID string) []int {
	if runID == "" {
		return nil
	}
	tag := "HATCH_RUN_ID=" + runID
	var pids []int
	if runtime.GOOS == "linux" {
		environs, _ := filepath.Glob("/proc/[0-9]*/environ")
		for _, path := range environs {
			data, err := os.ReadFile(path)
			if err != nil {
				continue
			}
			for _, entry := range bytes.Split(data, []byte{0}) {
				if string(entry) == tag {
					if pid, err := strconv.Atoi(strings.Split(path, "/")[2]); err == nil {
						pids = append(pids, pid)
					}
					break
				}
			}
		}
		return pids
	}
	// macOS: `ps -E` appends each process's environment to its command line.
	output, err := exec.Command("/bin/ps", "-axEww", "-o", "pid=,command=").Output()
	if err != nil {
		return nil
	}
	scanner := bufio.NewScanner(bytes.NewReader(output))
	scanner.Buffer(make([]byte, 0, 1<<20), 16<<20)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		space := strings.IndexByte(line, ' ')
		if space < 0 || !strings.Contains(line[space:]+" ", " "+tag+" ") {
			continue
		}
		if pid, err := strconv.Atoi(line[:space]); err == nil {
			pids = append(pids, pid)
		}
	}
	return pids
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

func processAlive(pid int) (bool, bool) {
	err := syscall.Kill(pid, 0)
	if err == nil || err == syscall.EPERM {
		if processIsZombie(pid) {
			return false, true
		}
		return true, true
	}
	if err == syscall.ESRCH {
		return false, true
	}
	return false, false
}

func processIsZombie(pid int) bool {
	if runtime.GOOS == "linux" {
		data, err := os.ReadFile(fmt.Sprintf("/proc/%d/stat", pid))
		if err != nil {
			return false
		}
		end := strings.LastIndexByte(string(data), ')')
		fields := strings.Fields(string(data)[end+1:])
		return end >= 0 && len(fields) > 0 && fields[0] == "Z"
	}
	output, err := exec.Command("/bin/ps", "-o", "stat=", "-p", fmt.Sprint(pid)).Output()
	return err == nil && strings.HasPrefix(strings.TrimSpace(string(output)), "Z")
}
