package provider

import (
	"encoding/json"
	"strings"
	"testing"
)

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
			`{"type":"agent_end","isTerminal":true}` + "\n" +
			`{"type":"message_end","message":{"role":"assistant","content":[{"type":"text","text":"late output"}]}}` + "\n",
	)
	got := Interpret("pi", stdout, nil)
	if string(got.Output) != "late output" || got.TerminalMarker != "observed" {
		t.Fatalf("unexpected interpretation: %#v", got)
	}
}

func TestInterpretEmitsAdapterDriftTripwire(t *testing.T) {
	got := Interpret("claude", []byte(`{"type":"future_terminal","result":"answer"}`+"\n"), nil)
	if got.TerminalMarker != "not_observed" || len(got.Warnings) != 1 || got.Warnings[0].Code != "adapter_recognition_empty" {
		t.Fatalf("drift interpretation: %#v", got)
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

func TestInterpretPiSurfacesProviderErrorCarriedOnTheAssistantMessage(t *testing.T) {
	stdout := []byte(`{"type":"session","id":"pi-session"}` + "\n" +
		`{"type":"message_end","message":{"role":"assistant","content":[],"stopReason":"error","errorMessage":"OpenAI API error (401): invalid_project"}}` + "\n" +
		`{"type":"agent_end","isTerminal":true}` + "\n")
	got := Interpret("pi", stdout, nil)
	if got.Error != "OpenAI API error (401): invalid_project" || len(got.Output) != 0 {
		t.Fatalf("provider failure was not surfaced: %#v", got)
	}
}

func FuzzInterpret(f *testing.F) {
	f.Add("raw", []byte("plain output\n"), []byte{})
	f.Add("claude", []byte(`{"type":"result","result":"done"}`+"\n"), []byte{})
	f.Add("cursor", []byte(`{"type":"result","subtype":"success","result":"done"}`+"\n"), []byte{})
	f.Add("pi", []byte(`{"type":"session","id":"pi_1"}`+"\n"), []byte{})
	f.Add("omp", []byte(`{"type":"session","id":"omp_1"}`+"\n"), []byte{})
	f.Fuzz(func(t *testing.T, adapter string, stdout, stderr []byte) {
		switch adapter {
		case "raw", "claude", "cursor", "pi", "omp":
			_ = Interpret(adapter, stdout, stderr)
		default:
			t.Skip()
		}
	})
}
