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

func TestInterpretOpenCodeJoinsChunksAndRecoversWithWarning(t *testing.T) {
	stdout := []byte(
		`{"type":"error","error":{"data":{"message":"transient"}}}` + "\n" +
			`{"type":"text","part":{"text":"one ","metadata":{"openai":{"phase":"final_answer"}}}}` + "\n" +
			`{"type":"text","part":{"text":"two","metadata":{"openai":{"phase":"final_answer"}}}}` + "\n" +
			`{"type":"step_finish","part":{"reason":"stop"}}` + "\n",
	)
	got := Interpret("opencode", stdout, nil)
	if string(got.Output) != "one two" || got.Error != "" || len(got.Warnings) != 1 || got.Warnings[0] != "transient" || got.TerminalMarker != "observed" {
		t.Fatalf("unexpected interpretation: %#v", got)
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

func FuzzInterpret(f *testing.F) {
	f.Add("raw", []byte("plain output\n"), []byte{})
	f.Add("claude", []byte(`{"type":"result","result":"done"}`+"\n"), []byte{})
	f.Add("cursor", []byte(`{"type":"result","subtype":"success","result":"done"}`+"\n"), []byte{})
	f.Add("opencode", []byte(`{"type":"step_start","sessionID":"ses_1"}`+"\n"), []byte{})
	f.Fuzz(func(t *testing.T, adapter string, stdout, stderr []byte) {
		switch adapter {
		case "raw", "claude", "cursor", "opencode":
			_ = Interpret(adapter, stdout, stderr)
		default:
			t.Skip()
		}
	})
}
