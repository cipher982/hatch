package provider

import "testing"

func TestProgressParserDeduplicatesToolsAndReportsLifecycle(t *testing.T) {
	tests := []struct {
		name, adapter, label string
		lines                []string
		want                 []string
	}{
		{
			name: "claude", adapter: "claude", label: "Claude",
			lines: []string{
				`{"type":"system","subtype":"init","model":"haiku","session_id":"claude-session"}`,
				`{"type":"assistant","message":{"content":[{"type":"tool_use","id":"tool-1","name":"Read","input":{"file_path":"/tmp/a"}}]}}`,
				`{"type":"assistant","message":{"content":[{"type":"tool_use","id":"tool-1","name":"Read","input":{"file_path":"/tmp/a"}}]}}`,
				`{"type":"result","duration_ms":1200}`,
			},
			want: []string{"[hatch] Claude started (haiku, session claude-s)", "[hatch] Read: /tmp/a", "[hatch] Claude completed in 1.2s"},
		},
		{
			name: "cursor", adapter: "cursor", label: "Cursor",
			lines: []string{
				`{"type":"tool_call","subtype":"started","call_id":"call-1","tool_call":{"shell":{"args":{"command":"go test ./..."}}}}`,
				`{"type":"tool_call","subtype":"started","call_id":"call-1","tool_call":{"shell":{"args":{"command":"go test ./..."}}}}`,
			},
			want: []string{"[hatch] Cursor shell: go test ./..."},
		},
		{
			name: "omp", adapter: "omp", label: "OpenRouter",
			lines: []string{
				`{"type":"session","id":"omp_123456789"}`,
				`{"type":"tool_execution_start","toolCallId":"call-1","toolName":"bash","args":{"command":"go test ./..."}}`,
				`{"type":"agent_end","isTerminal":true}`,
			},
			want: []string{"[hatch] OpenRouter started (session omp_1234)", "[hatch] bash: go test ./...", "[hatch] OpenRouter completed"},
		},
		{
			name: "antigravity", adapter: "antigravity", label: "Gemini",
			lines: []string{
				`{"event":"init","conversation_id":"conv-123456789","init":{"model":"gemini-4-carbon-medium"}}`,
				`{"event":"step_update","step_update":{"conversation_id":"conv-123456789","step_index":2,"state":"ACTIVE","step_type":"tool","tool_name":"run_command","tool_info":{"parameters":{"CommandLine":"go test ./..."}}}}`,
				`{"event":"step_update","step_update":{"conversation_id":"conv-123456789","step_index":2,"state":"ACTIVE","step_type":"tool","tool_name":"run_command","tool_info":{"parameters":{"CommandLine":"go test ./..."}}}}`,
				`{"event":"result","result":{"status":"SUCCESS","duration_seconds":2.5}}`,
			},
			want: []string{"[hatch] Antigravity started (gemini-4-carbon-medium, session conv-123)", "[hatch] Antigravity run_command: go test ./...", "[hatch] Antigravity completed in 2.5s"},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			parser := NewProgressParser(test.adapter, test.label)
			var got []string
			for _, line := range test.lines {
				got = append(got, parser.Observe([]byte(line))...)
			}
			if len(got) != len(test.want) {
				t.Fatalf("progress = %#v, want %#v", got, test.want)
			}
			for index := range got {
				if got[index] != test.want[index] {
					t.Fatalf("progress = %#v, want %#v", got, test.want)
				}
			}
		})
	}
}
