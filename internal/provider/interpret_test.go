package provider

import "testing"

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
	if got.Error != "request rejected" || got.TerminalMarker != "not_observed" {
		t.Fatalf("unexpected interpretation: %#v", got)
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
