package provider

import (
	"bufio"
	"bytes"
	"encoding/json"
	"strings"
)

type Interpretation struct {
	Output         []byte
	Error          string
	TerminalMarker string
	NativeID       string
	NativeIDState  string
	Retention      string
	Capabilities   map[string]string
}

func Interpret(adapter string, stdout, stderr []byte) Interpretation {
	result := Interpretation{
		TerminalMarker: "not_applicable", NativeIDState: "not_exposed",
		Retention: "unknown", Capabilities: map[string]string{},
	}
	if adapter == "" || adapter == "raw" {
		result.Output = append([]byte(nil), stdout...)
		return result
	}
	result.TerminalMarker = "not_observed"
	result.NativeIDState = "unavailable"
	var lastText string
	scanner := bufio.NewScanner(bytes.NewReader(stdout))
	buffer := make([]byte, 64*1024)
	scanner.Buffer(buffer, 16*1024*1024)
	for scanner.Scan() {
		var event map[string]any
		if json.Unmarshal(scanner.Bytes(), &event) != nil {
			continue
		}
		typeName, _ := event["type"].(string)
		switch adapter {
		case "claude", "cursor":
			if typeName == "system" && event["subtype"] == "init" {
				if id, ok := event["session_id"].(string); ok && id != "" {
					result.NativeID, result.NativeIDState = id, "observed"
				}
			}
			if typeName == "assistant" {
				lastText = assistantText(event)
			}
			if typeName == "result" {
				if value, ok := event["result"].(string); ok {
					if adapter == "cursor" && (event["is_error"] == true || event["subtype"] == "error") {
						result.Error = value
					} else {
						result.Output = []byte(value)
						result.TerminalMarker = "observed"
					}
				}
			}
		case "opencode":
			if typeName == "step_start" {
				if id, ok := event["sessionID"].(string); ok && id != "" {
					result.NativeID, result.NativeIDState = id, "observed"
					result.Retention = "hatch_preserved"
					result.Capabilities["snapshot"] = "supported"
				}
			}
			if typeName == "text" {
				if part, ok := event["part"].(map[string]any); ok {
					text, _ := part["text"].(string)
					lastText = text
					if isFinalOpenCodeText(part) {
						result.Output = []byte(text)
					}
				}
			}
			if typeName == "step_finish" && len(result.Output) > 0 {
				result.TerminalMarker = "observed"
			}
			if typeName == "error" {
				result.Error = nestedErrorMessage(event)
			}
		}
	}
	if len(result.Output) == 0 && lastText != "" {
		result.Output = []byte(lastText)
	}
	if scanner.Err() != nil && result.Error == "" {
		result.Error = scanner.Err().Error()
	}
	return result
}

func assistantText(event map[string]any) string {
	message, _ := event["message"].(map[string]any)
	content, _ := message["content"].([]any)
	var texts []string
	for _, item := range content {
		block, _ := item.(map[string]any)
		if block["type"] == "text" {
			if text, ok := block["text"].(string); ok && text != "" {
				texts = append(texts, text)
			}
		}
	}
	return strings.Join(texts, "")
}

func isFinalOpenCodeText(part map[string]any) bool {
	metadata, _ := part["metadata"].(map[string]any)
	openAI, _ := metadata["openai"].(map[string]any)
	return openAI["phase"] == "final_answer"
}

func nestedErrorMessage(event map[string]any) string {
	errorValue, _ := event["error"].(map[string]any)
	data, _ := errorValue["data"].(map[string]any)
	message, _ := data["message"].(string)
	return message
}
