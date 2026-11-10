package provider

import (
	"bytes"
	"encoding/json"
	"strings"
)

type Interpretation struct {
	Output         []byte
	Error          string
	Warnings       []Warning
	TerminalMarker string
	NativeID       string
	NativeIDState  string
	Retention      string
	Capabilities   map[string]string
}

type Warning struct {
	Code    string
	Message string
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
	switch adapter {
	case "claude":
		result.Retention = "provider_owned"
		result.Capabilities["identify"] = "supported"
	case "cursor":
		result.Retention = "unknown"
		result.Capabilities["identify"] = "supported"
	case "antigravity":
		result.Retention = "provider_owned"
		result.Capabilities["identify"] = "supported"
	case "pi", "omp":
		result.Capabilities["identify"] = "supported"
	}
	var textChunks, finalChunks []string
	validStructuredEvent := false
	for _, line := range bytes.Split(stdout, []byte{'\n'}) {
		if len(bytes.TrimSpace(line)) == 0 {
			continue
		}
		var event map[string]any
		if json.Unmarshal(line, &event) != nil {
			continue
		}
		validStructuredEvent = true
		typeName, _ := event["type"].(string)
		switch adapter {
		case "claude":
			if typeName == "system" && event["subtype"] == "init" {
				observeSession(&result, event, "session_id")
			}
			if typeName == "assistant" {
				if text := assistantText(event); text != "" {
					textChunks = append(textChunks, text)
				}
			}
			if typeName == "result" {
				result.TerminalMarker = "observed"
				value, _ := event["result"].(string)
				if event["is_error"] == true || event["subtype"] == "error" {
					if strings.TrimSpace(value) == "" {
						value = "Claude returned an error result"
					}
					result.Error = value
				} else if strings.TrimSpace(value) != "" {
					finalChunks = []string{value}
				}
			}
			if typeName == "error" {
				result.Error = nestedErrorMessage(event)
			}
		case "cursor":
			if typeName == "system" && event["subtype"] == "init" {
				observeSession(&result, event, "session_id")
			}
			if typeName == "assistant" {
				if text := assistantText(event); text != "" {
					textChunks = append(textChunks, text)
				}
			}
			if typeName == "result" {
				result.TerminalMarker = "observed"
				value, _ := event["result"].(string)
				if event["subtype"] != "success" || event["is_error"] == true {
					if strings.TrimSpace(value) == "" {
						value = "Cursor returned an error result"
					}
					result.Error = value
				} else if strings.TrimSpace(value) == "" {
					result.Error = "Cursor result event did not contain output"
				} else {
					finalChunks = []string{value}
				}
			}
		case "pi", "omp":
			if typeName == "session" {
				observeSession(&result, event, "id")
			}
			if typeName == "message_update" {
				if delta := piLikeTextDelta(event); delta != "" {
					textChunks = append(textChunks, delta)
				}
			}
			if typeName == "message_end" {
				if text := piLikeAssistantText(event["message"]); text != "" {
					finalChunks = append(finalChunks, text)
				}
				// A provider failure (401, quota, unknown model) arrives as an
				// assistant message with stopReason "error", not as an error event.
				if message, ok := event["message"].(map[string]any); ok && message["stopReason"] == "error" {
					if text, _ := message["errorMessage"].(string); text != "" {
						result.Error = text
					}
				}
			}
			if typeName == "agent_end" {
				terminal, hasTerminal := event["isTerminal"].(bool)
				if !hasTerminal || terminal {
					result.TerminalMarker = "observed"
					if len(finalChunks) == 0 {
						if text := piLikeAssistantText(event["messages"]); text != "" {
							finalChunks = append(finalChunks, text)
						}
					}
				}
			}
			if typeName == "error" {
				if message := piLikeErrorMessage(event); message != "" {
					result.Error = message
				}
			}
		case "antigravity":
			// Antigravity names its envelope by the event itself
			// ({"event":"init",...}) and puts the payload under that key.
			name, _ := event["event"].(string)
			payload, _ := event[name].(map[string]any)
			switch name {
			case "init":
				observeSession(&result, event, "conversation_id")
			case "step_update":
				if payload["step_type"] == "agent_response" && payload["state"] == "ACTIVE" {
					if delta, ok := payload["text_delta"].(string); ok && delta != "" {
						textChunks = append(textChunks, delta)
					}
				}
			case "result":
				result.TerminalMarker = "observed"
				status, _ := payload["status"].(string)
				response, _ := payload["response"].(string)
				if status != "SUCCESS" {
					// Antigravity exits 0 even when the run fails, and reports
					// the failure only in the result event.
					message, _ := payload["error"].(string)
					if strings.TrimSpace(message) == "" {
						message = "Antigravity returned an unsuccessful result (" + defaultName(status, "no status") + ")"
					}
					result.Error = message
				} else if strings.TrimSpace(response) == "" {
					result.Error = "Antigravity result event did not contain output"
				} else {
					finalChunks = []string{response}
				}
			}
		}
	}
	chunks := finalChunks
	if len(chunks) == 0 {
		chunks = textChunks
	}
	result.Output = []byte(strings.Join(chunks, ""))
	if result.Error != "" && result.TerminalMarker == "observed" && len(result.Output) > 0 {
		result.Warnings = append(result.Warnings, Warning{Code: "transient_provider_error", Message: result.Error})
		result.Error = ""
	}
	if validStructuredEvent && result.TerminalMarker == "not_observed" && result.Error == "" {
		message := "structured events contained no terminal result recognized by the adapter"
		result.Warnings = append(result.Warnings, Warning{Code: "adapter_recognition_empty", Message: message})
	}
	return result
}

func piLikeTextDelta(event map[string]any) string {
	if delta, ok := event["delta"].(string); ok {
		return delta
	}
	if nested, ok := event["assistantMessageEvent"].(map[string]any); ok {
		if delta, ok := nested["delta"].(string); ok {
			return delta
		}
		if text, ok := nested["text"].(string); ok {
			return text
		}
	}
	return ""
}

func piLikeAssistantText(value any) string {
	switch current := value.(type) {
	case map[string]any:
		if role, ok := current["role"].(string); ok && role != "" && role != "assistant" {
			return ""
		}
		if text, ok := current["text"].(string); ok {
			return text
		}
		if content, ok := current["content"]; ok {
			return piLikeAssistantText(content)
		}
	case []any:
		var texts []string
		for _, item := range current {
			if text := piLikeAssistantText(item); text != "" {
				texts = append(texts, text)
			}
		}
		return strings.Join(texts, "")
	}
	return ""
}

func piLikeErrorMessage(event map[string]any) string {
	if message, ok := event["message"].(string); ok && message != "" {
		return message
	}
	if errorValue, ok := event["error"].(string); ok && errorValue != "" {
		return errorValue
	}
	if nested, ok := event["error"].(map[string]any); ok {
		if message, ok := nested["message"].(string); ok && message != "" {
			return message
		}
	}
	return nestedErrorMessage(event)
}

func observeSession(result *Interpretation, event map[string]any, name string) {
	if id, ok := event[name].(string); ok && strings.TrimSpace(id) != "" {
		result.NativeID, result.NativeIDState = id, "observed"
	}
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

func nestedErrorMessage(event map[string]any) string {
	errorValue, _ := event["error"].(map[string]any)
	data, _ := errorValue["data"].(map[string]any)
	for _, value := range []any{data["message"], errorValue["message"], errorValue["name"]} {
		if message, ok := value.(string); ok && message != "" {
			return message
		}
	}
	return ""
}
