package cli

import (
	"fmt"
	"os"

	runner "github.com/cipher982/hatch/internal/run"
)

// inheritFromPriorRun fills what a follow-up left unsaid from the run it
// continues, so `hatch --resume <run-id> "task"` needs nothing else: the same
// model and harness, reasoning effort and working directory. Naming a model,
// harness, effort or --cwd overrides the inherited value; the earlier run's
// read-only restriction is deliberately not inherited, since the usual
// follow-up to an investigation is to make the change.
func inheritFromPriorRun(request *Request, prior runner.PriorRun) {
	if request.Backend == "" && request.Model == "" {
		request.Backend, request.Model = prior.Backend, prior.Model
		if request.ReasoningEffort == "" {
			request.ReasoningEffort = prior.Effort
		}
	}
	if request.CWD == "" && prior.CWD != "" {
		if info, err := os.Stat(prior.CWD); err == nil && info.IsDir() {
			request.CWD = prior.CWD
		}
	}
}

// checkFlagSupport rejects a flag the chosen backend would silently ignore.
// Every surfaced provider supports the same documented options; only these two
// raw escape-hatch flags are tied to one backend each.
func checkFlagSupport(request Request) error {
	if request.SkipGitRepoCheck && request.Backend != "codex" {
		return fmt.Errorf("--skip-git-repo-check only applies to the raw codex backend (-b codex)")
	}
	if request.IncludePartialMessages && request.Backend != "claude" {
		return fmt.Errorf("--include-partial-messages only applies to the claude backend")
	}
	return nil
}
