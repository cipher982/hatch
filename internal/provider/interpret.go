package provider

import (
	"bytes"
	"encoding/json"
	"regexp"
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
	case "opencode":
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
		case "opencode":
			if typeName == "step_start" {
				observeSession(&result, event, "sessionID")
				if result.NativeID != "" {
					result.Retention = "hatch_preserved"
					result.Capabilities["snapshot"] = "supported"
				}
			}
			if typeName == "text" {
				if part, ok := event["part"].(map[string]any); ok {
					text, _ := part["text"].(string)
					if text != "" {
						textChunks = append(textChunks, text)
						if isFinalOpenCodeText(part) {
							finalChunks = append(finalChunks, text)
						}
					}
				}
			}
			if typeName == "step_finish" {
				part, _ := event["part"].(map[string]any)
				if part["reason"] == "stop" {
					result.TerminalMarker = "observed"
				}
			}
			if typeName == "error" {
				if message := nestedErrorMessage(event); message != "" {
					result.Error = message
				}
			}
		}
	}
	chunks := finalChunks
	if len(chunks) == 0 {
		chunks = textChunks
	}
	result.Output = []byte(strings.Join(chunks, ""))
	if adapter == "opencode" && result.Error == "" {
		if stderrError := extractOpenCodeLogError(string(stderr)); stderrError != "" {
			if result.TerminalMarker == "observed" && len(result.Output) > 0 {
				result.Warnings = append(result.Warnings, Warning{Code: "stderr_error_recovered", Message: stderrError})
			} else if len(result.Output) == 0 {
				result.Error = stderrError
			}
		}
	}
	if result.Error != "" && result.TerminalMarker == "observed" && len(result.Output) > 0 {
		result.Warnings = append(result.Warnings, Warning{Code: "transient_provider_error", Message: result.Error})
		result.Error = ""
	}
	if validStructuredEvent && result.TerminalMarker == "not_observed" && result.Error == "" {
		result.Warnings = append(result.Warnings, Warning{Code: "adapter_recognition_empty", Message: "structured events contained no terminal result recognized by the adapter"})
	}
	return result
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

func isFinalOpenCodeText(part map[string]any) bool {
	metadata, _ := part["metadata"].(map[string]any)
	openAI, _ := metadata["openai"].(map[string]any)
	return openAI["phase"] == "final_answer"
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

var statusCodePattern = regexp.MustCompile(`"statusCode":\s*(\d{3})`)
var dataMessagePattern = regexp.MustCompile(`"data":\{"message":"((?:[^"\\]|\\.)*)"`)
var messagePattern = regexp.MustCompile(`"message":"((?:[^"\\]|\\.)*)"`)

func extractOpenCodeLogError(stderr string) string {
	best, status := "", ""
	for _, line := range strings.Split(stderr, "\n") {
		if !strings.Contains(line, "ERROR") || !strings.Contains(line, "error=") {
			continue
		}
		if match := statusCodePattern.FindStringSubmatch(line); len(match) > 1 {
			status = match[1]
		}
		if match := dataMessagePattern.FindStringSubmatch(line); len(match) > 1 {
			best = unescapeJSONString(match[1])
		} else if match := messagePattern.FindStringSubmatch(line); len(match) > 1 {
			best = unescapeJSONString(match[1])
		}
	}
	if best != "" && status != "" {
		return "Bedrock error " + status + ": " + best
	}
	return best
}

func unescapeJSONString(value string) string {
	var decoded string
	if json.Unmarshal([]byte(`"`+value+`"`), &decoded) == nil {
		return decoded
	}
	return value
}
