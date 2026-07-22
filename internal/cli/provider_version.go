package cli

import (
	"context"
	"os/exec"
	"strings"
	"time"

	"github.com/cipher982/hatch/internal/provider"
)

func populateProviderVersion(invocation *provider.Invocation) {
	if invocation.Adapter != "opencode" || len(invocation.Argv) == 0 || invocation.ProviderVersion != "" {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	output, err := exec.CommandContext(ctx, invocation.Argv[0], "--version").Output()
	if err != nil {
		return
	}
	line := strings.TrimSpace(strings.SplitN(string(output), "\n", 2)[0])
	if line == "" || len(line) > 200 || strings.HasPrefix(line, "{") {
		return
	}
	invocation.ProviderVersion = line
}
