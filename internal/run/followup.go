package run

import (
	"fmt"
	"strings"
)

// followUpPartLimit bounds each part (request, result) carried into a
// follow-up prompt. ReadContent already caps a single page at this size.
const followUpPartLimit = 32768

// PriorRun is what a follow-up needs from an earlier Hatch run: where it ran,
// which model ran it, what it was asked and what it answered.
type PriorRun struct {
	RunID          string
	Surface        string
	Backend        string
	Model          string
	Effort         string
	CWD            string
	Outcome        string
	Request        string
	RequestClipped bool
	Result         string
	ResultClipped  bool
	ResultCommand  string
	RequestCommand string
}

// LoadPriorRun reads a finished Hatch run from the run store. Provider-native
// session state is never needed: the request and the durable answer are the
// whole context, so a follow-up works the same on every provider.
func LoadPriorRun(root, id string) (PriorRun, error) {
	record, err := InspectRecord(root, "", id)
	if err != nil {
		return PriorRun{}, err
	}
	m := record.Manifest
	if record.Kind != "hatch_run" || m == nil {
		return PriorRun{}, fmt.Errorf("run %q is not a Hatch agent run and cannot be continued", id)
	}
	if m.Lifecycle != LifecycleTerminal {
		return PriorRun{}, fmt.Errorf("run %q has not finished (%s); wait for it or read it with: hatch runs inspect %s", id, m.Lifecycle, id)
	}
	if m.Backend == "" || m.Execution == "http" || strings.HasPrefix(m.Surface, "expert") {
		return PriorRun{}, fmt.Errorf("run %q is an expert consultation; continue it by asking a new question with that answer pasted in", id)
	}
	prior := PriorRun{
		RunID: m.RunID, Surface: m.Surface, Backend: m.Backend, Model: m.Model,
		Effort: m.ReasoningPolicy.Effort, CWD: m.CWD,
		ResultCommand:  "hatch runs read " + m.RunID + " --part result",
		RequestCommand: "hatch runs read " + m.RunID + " --part request",
	}
	if m.Outcome != nil {
		prior.Outcome = string(*m.Outcome)
	}
	if page, err := ReadContent(root, "", id, ContentOptions{Part: "request", Limit: followUpPartLimit}); err == nil {
		prior.Request, prior.RequestClipped = page.Content, page.Truncated
	}
	if page, err := ReadContent(root, "", id, ContentOptions{Part: "result", Limit: followUpPartLimit}); err == nil {
		prior.Result, prior.ResultClipped = page.Content, page.Truncated
	}
	return prior, nil
}

// FollowUpPrompt wraps a new task with the earlier run's request and answer.
func (p PriorRun) FollowUpPrompt(task string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "You are continuing earlier Hatch run %s (%s). That run was a separate agent session whose memory you do not have; its task and final answer are reproduced below. Treat its findings as prior evidence from the same working tree: reuse them, and re-check only what the new task depends on.\n\n", p.RunID, p.Surface)
	if p.Outcome != "" && p.Outcome != string(OutcomeSucceeded) && p.Outcome != string(OutcomeSucceededWarnings) {
		fmt.Fprintf(&b, "Note: that run ended as %s, so its answer may be partial.\n\n", p.Outcome)
	}
	b.WriteString("Earlier task:\n")
	b.WriteString(strings.TrimSpace(p.Request))
	if p.RequestClipped {
		fmt.Fprintf(&b, "\n[clipped; full text: %s]", p.RequestCommand)
	}
	b.WriteString("\n\nEarlier answer:\n")
	if strings.TrimSpace(p.Result) == "" {
		b.WriteString("(no answer was captured)")
	} else {
		b.WriteString(strings.TrimSpace(p.Result))
	}
	if p.ResultClipped {
		fmt.Fprintf(&b, "\n[clipped; full text: %s]", p.ResultCommand)
	}
	b.WriteString("\n\nNew task:\n")
	b.WriteString(task)
	return b.String()
}
