package provider

import (
	"bytes"
	"encoding/json"
	"fmt"
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
	case "pi", "omp":
		result.Capabilities["identify"] = "supported"
	}
	var textChunks, finalChunks []string
	var openCodeSteps, openCodeTools, openCodeTextBytes int
	openCodeToolInputs := map[string]int{}
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
				openCodeSteps++
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
						openCodeTextBytes += len(text)
						if isFinalOpenCodeText(part) {
							finalChunks = append(finalChunks, text)
						}
					}
				}
			}
			if typeName == "tool_use" {
				openCodeTools++
				if part, ok := event["part"].(map[string]any); ok {
					if input := openCodeToolInputID(part); input != "" {
						openCodeToolInputs[input]++
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
	if adapter == "opencode" && validStructuredEvent {
		if warning := stalledRunWarning(openCodeSteps, openCodeTools, openCodeTextBytes, openCodeToolInputs); warning != "" {
			result.Warnings = append(result.Warnings, Warning{Code: "stall_detected", Message: warning})
		}
	}
	if validStructuredEvent && result.TerminalMarker == "not_observed" && result.Error == "" {
		message := "structured events contained no terminal result recognized by the adapter"
		if adapter == "opencode" {
			message += fmt.Sprintf(" (%d steps, %d tool calls, %d text bytes)", openCodeSteps, openCodeTools, openCodeTextBytes)
		}
		result.Warnings = append(result.Warnings, Warning{Code: "adapter_recognition_empty", Message: message})
	}
	return result
}

const (
	stallMinSteps     = 8
	stallMinRepeats   = 3
	stallMaxTextBytes = 500
)

// stalledRunWarning classifies an OpenCode run that produced no meaningful
// assistant text across many tool-only steps while repeating identical tool
// calls (for example re-reading the same file or re-running the same glob).
// Returns "" for healthy runs.
func stalledRunWarning(steps, toolCalls, textBytes int, inputs map[string]int) string {
	if steps < stallMinSteps || textBytes > stallMaxTextBytes {
		return ""
	}
	var repeated []string
	totalRepeats := 0
	for input, count := range inputs {
		if count >= stallMinRepeats {
			totalRepeats += count
			if len(repeated) < 3 {
				repeated = append(repeated, fmt.Sprintf("%s x%d", input, count))
			}
		}
	}
	if totalRepeats == 0 {
		return ""
	}
	return fmt.Sprintf("%d steps, %d tool calls, %d text bytes; repeated identical tool calls: %s", steps, toolCalls, textBytes, strings.Join(repeated, ", "))
}

// openCodeToolInputID derives a stable identity for a tool call so repeated
// identical invocations can be counted. Empty for calls without a useful input.
func openCodeToolInputID(part map[string]any) string {
	tool, _ := part["tool"].(string)
	state, _ := part["state"].(map[string]any)
	input, _ := state["input"].(map[string]any)
	switch tool {
	case "read":
		if path := firstString(input, "filePath"); path != "" {
			return "read:" + path
		}
	case "bash":
		if command := firstString(input, "command"); command != "" {
			return "bash:" + truncateCommand(command)
		}
	case "glob":
		if pattern := firstString(input, "pattern"); pattern != "" {
			return "glob:" + pattern
		}
	case "grep":
		if pattern := firstString(input, "pattern"); pattern != "" {
			return "grep:" + pattern
		}
	default:
		if tool != "" {
			if raw, err := json.Marshal(input); err == nil && len(raw) > 0 {
				return tool + ":" + string(raw)
			}
		}
	}
	return ""
}

func truncateCommand(command string) string {
	const maxCommandLength = 80
	if len(command) > maxCommandLength {
		return command[:maxCommandLength] + "..."
	}
	return command
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
