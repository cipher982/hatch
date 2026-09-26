package cli

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"syscall"
)

// runReview hands `hatch review ...` to the external hatch-review launcher.
// Review policy (git ranges, requester intent, CI) lives outside Hatch; Hatch
// stays a runner. hatch-review calls back into `hatch <surface>` per phase.
func runReview(args []string, stderr io.Writer) int {
	path, err := exec.LookPath("hatch-review")
	if err != nil {
		fmt.Fprintln(stderr, "Error: hatch review needs hatch-review on PATH (review-hub in ~/git/me; run `agents tools sync`)")
		return 2
	}
	err = syscall.Exec(path, append([]string{"hatch-review"}, args...), os.Environ())
	fmt.Fprintf(stderr, "Error: exec hatch-review: %v\n", err)
	return 1
}
