package provider

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestInterpretDetectsStalledOpenCodeRun(t *testing.T) {
	var stdout strings.Builder
	for i := 0; i < 10; i++ {
		stdout.WriteString(`{"type":"step_start","sessionID":"ses_stall","part":{"type":"step-start"}}` + "\n")
	}
	for i := 0; i < 12; i++ {
		stdout.WriteString(`{"type":"tool_use","part":{"tool":"glob","state":{"status":"completed","input":{"pattern":"docs/specs/*qualification*.md"},"output":"same"}}}` + "\n")
	}
	stdout.WriteString(`{"type":"text","part":{"type":"text","text":"Let me read the core docs first."}}` + "\n")
	got := Interpret("opencode", []byte(stdout.String()), nil)
	var stall *Warning
	for i := range got.Warnings {
		if got.Warnings[i].Code == "stall_detected" {
			stall = &got.Warnings[i]
		}
	}
	if stall == nil {
		t.Fatalf("expected stall_detected warning, got %#v", got.Warnings)
	}
	if !strings.Contains(stall.Message, "10 steps") || !strings.Contains(stall.Message, "12 tool calls") ||
		!strings.Contains(stall.Message, "repeated identical tool calls") ||
		!strings.Contains(stall.Message, "glob:docs/specs/*qualification*.md x12") {
		t.Fatalf("stall message = %q", stall.Message)
	}
}

func TestInterpretHealthyOpenCodeRunHasNoStallWarning(t *testing.T) {
	var stdout strings.Builder
	stdout.WriteString(`{"type":"step_start","sessionID":"ses_ok","part":{"type":"step-start"}}` + "\n")
	stdout.WriteString(`{"type":"tool_use","part":{"tool":"bash","state":{"status":"completed","input":{"command":"ls -la"},"output":"x"}}}` + "\n")
	stdout.WriteString(`{"type":"tool_use","part":{"tool":"read","state":{"status":"completed","input":{"filePath":"a.md"},"output":"y"}}}` + "\n")
	stdout.WriteString(`{"type":"text","part":{"type":"text","text":"Here is the synthesis with enough substance to satisfy a review."}}` + "\n")
	stdout.WriteString(`{"type":"step_finish","part":{"type":"step-finish","reason":"stop"}}` + "\n")
	got := Interpret("opencode", []byte(stdout.String()), nil)
	for _, warning := range got.Warnings {
		if warning.Code == "stall_detected" {
			t.Fatalf("healthy run flagged as stalled: %#v", got.Warnings)
		}
	}
	if got.TerminalMarker != "observed" {
		t.Fatalf("terminal marker = %q", got.TerminalMarker)
	}
}

func TestInterpretMissingTerminalPreservesOutputButDoesNotClaimSuccess(t *testing.T) {
	stdout := []byte(`{"type":"assistant","message":{"content":[{"type":"text","text":"useful evidence"}]}}` + "\n")
	got := Interpret("claude", stdout, nil)
	if string(got.Output) != "useful evidence" || got.TerminalMarker != "not_observed" {
		t.Fatalf("unexpected interpretation: %#v", got)
	}
}

func TestInterpretCursorError(t *testing.T) {
	stdout := []byte(`{"type":"result","subtype":"error","is_error":true,"result":"request rejected"}` + "\n")
	got := Interpret("cursor", stdout, nil)
	if got.Error != "request rejected" || got.TerminalMarker != "observed" {
		t.Fatalf("unexpected interpretation: %#v", got)
	}
}

func TestInterpretClaudeStructuredError(t *testing.T) {
	stdout := []byte(`{"type":"result","subtype":"error","is_error":true,"result":"permission denied"}` + "\n")
	got := Interpret("claude", stdout, nil)
	if got.Error != "permission denied" || got.TerminalMarker != "observed" || got.Retention != "provider_owned" {
		t.Fatalf("unexpected interpretation: %#v", got)
	}
}

func TestInterpretReorderedAndUnknownEvents(t *testing.T) {
	stdout := []byte(
		`{"type":"future_event","payload":{"kept":"in raw evidence"}}` + "\n" +
			`{"type":"step_finish","part":{"reason":"stop"}}` + "\n" +
			`{"type":"text","part":{"text":"late output"}}` + "\n",
	)
	got := Interpret("opencode", stdout, nil)
	if string(got.Output) != "late output" || got.TerminalMarker != "observed" {
		t.Fatalf("unexpected interpretation: %#v", got)
	}
}

func TestInterpretOpenCodeJoinsChunksAndRecoversWithWarning(t *testing.T) {
	stdout := []byte(
		`{"type":"error","error":{"data":{"message":"transient"}}}` + "\n" +
			`{"type":"text","part":{"text":"one ","metadata":{"openai":{"phase":"final_answer"}}}}` + "\n" +
			`{"type":"text","part":{"text":"two","metadata":{"openai":{"phase":"final_answer"}}}}` + "\n" +
			`{"type":"step_finish","part":{"reason":"stop"}}` + "\n",
	)
	got := Interpret("opencode", stdout, nil)
	if string(got.Output) != "one two" || got.Error != "" || len(got.Warnings) != 1 || got.Warnings[0].Code != "transient_provider_error" || got.Warnings[0].Message != "transient" || got.TerminalMarker != "observed" {
		t.Fatalf("unexpected interpretation: %#v", got)
	}
}

func TestInterpretEmitsAdapterDriftTripwire(t *testing.T) {
	got := Interpret("claude", []byte(`{"type":"future_terminal","result":"answer"}`+"\n"), nil)
	if got.TerminalMarker != "not_observed" || len(got.Warnings) != 1 || got.Warnings[0].Code != "adapter_recognition_empty" {
		t.Fatalf("drift interpretation: %#v", got)
	}
}

func TestInterpretRecordsRecoveredStderrError(t *testing.T) {
	stdout := []byte(`{"type":"text","part":{"text":"answer","metadata":{"openai":{"phase":"final_answer"}}}}` + "\n" +
		`{"type":"step_finish","part":{"reason":"stop"}}` + "\n")
	stderr := []byte(`ERROR error={"message":"earlier transport error"}`)
	got := Interpret("opencode", stdout, stderr)
	if got.Error != "" || len(got.Warnings) != 1 || got.Warnings[0].Code != "stderr_error_recovered" {
		t.Fatalf("recovered stderr interpretation: %#v", got)
	}
}

func TestInterpretLongLogicalLine(t *testing.T) {
	text := strings.Repeat("x", 20*1024*1024)
	event, err := json.Marshal(map[string]any{"type": "result", "subtype": "success", "result": text})
	if err != nil {
		t.Fatal(err)
	}
	got := Interpret("cursor", append(event, '\n'), nil)
	if len(got.Output) != len(text) || got.TerminalMarker != "observed" {
		t.Fatalf("long line output=%d marker=%s error=%q", len(got.Output), got.TerminalMarker, got.Error)
	}
}

func TestInterpretOpenCodeLogError(t *testing.T) {
	stderr := []byte(`2026 ERROR service error={"statusCode":503,"data":{"message":"Service unavailable"}}`)
	got := Interpret("opencode", nil, stderr)
	if got.Error != "Bedrock error 503: Service unavailable" {
		t.Fatalf("error = %q", got.Error)
	}
}

func TestInterpretPiLikeJSONEvents(t *testing.T) {
	stdout := []byte(
		`{"type":"session","id":"pi-session"}` + "\n" +
			`{"type":"message_update","assistantMessageEvent":{"type":"text_delta","delta":"ignored "}}` + "\n" +
			`{"type":"message_end","message":{"role":"assistant","content":[{"type":"text","text":"final answer"}]}}` + "\n" +
			`{"type":"agent_end","isTerminal":true}` + "\n",
	)
	for _, adapter := range []string{"pi", "omp"} {
		got := Interpret(adapter, stdout, nil)
		if string(got.Output) != "final answer" || got.NativeID != "pi-session" || got.TerminalMarker != "observed" || got.Error != "" {
			t.Fatalf("%s interpretation = %#v", adapter, got)
		}
	}
}

func TestInterpretPiLikeNonterminalAgentEndDoesNotComplete(t *testing.T) {
	stdout := []byte(`{"type":"session","id":"omp-session"}` + "\n" + `{"type":"message_update","delta":"partial"}` + "\n" + `{"type":"agent_end","isTerminal":false}` + "\n")
	got := Interpret("omp", stdout, nil)
	if string(got.Output) != "partial" || got.TerminalMarker != "not_observed" {
		t.Fatalf("nonterminal interpretation = %#v", got)
	}
}

func FuzzInterpret(f *testing.F) {
	f.Add("raw", []byte("plain output\n"), []byte{})
	f.Add("claude", []byte(`{"type":"result","result":"done"}`+"\n"), []byte{})
	f.Add("cursor", []byte(`{"type":"result","subtype":"success","result":"done"}`+"\n"), []byte{})
	f.Add("opencode", []byte(`{"type":"step_start","sessionID":"ses_1"}`+"\n"), []byte{})
	f.Add("pi", []byte(`{"type":"session","id":"pi_1"}`+"\n"), []byte{})
	f.Add("omp", []byte(`{"type":"session","id":"omp_1"}`+"\n"), []byte{})
	f.Fuzz(func(t *testing.T, adapter string, stdout, stderr []byte) {
		switch adapter {
		case "raw", "claude", "cursor", "opencode", "pi", "omp":
			_ = Interpret(adapter, stdout, stderr)
		default:
			t.Skip()
		}
	})
}
