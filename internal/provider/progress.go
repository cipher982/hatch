package provider

import (
	"encoding/json"
	"fmt"
	"strings"
)

type ProgressParser struct {
	Adapter string
	Label   string
	seen    map[string]bool
}

func NewProgressParser(adapter, label string) *ProgressParser {
	return &ProgressParser{Adapter: adapter, Label: label, seen: map[string]bool{}}
}

func (p *ProgressParser) Observe(line []byte) []string {
	var event map[string]any
	if json.Unmarshal(line, &event) != nil {
		return nil
	}
	typeName, _ := event["type"].(string)
	switch p.Adapter {
	case "claude":
		if typeName == "system" && event["subtype"] == "init" {
			return []string{startedMessage("Claude", event, "session_id")}
		}
		if typeName == "assistant" {
			return p.claudeTools(event)
		}
		if typeName == "result" {
			return []string{completedMessage("Claude", event["duration_ms"])}
		}
	case "cursor":
		if typeName == "system" && event["subtype"] == "init" {
			return []string{startedMessage("Cursor", event, "session_id")}
		}
		if typeName == "tool_call" && event["subtype"] == "started" {
			id, _ := event["call_id"].(string)
			if id != "" && p.seen[id] {
				return nil
			}
			p.seen[id] = true
			call, _ := event["tool_call"].(map[string]any)
			return []string{summarizeCursorTool(call)}
		}
		if typeName == "result" && event["subtype"] == "success" && event["is_error"] != true {
			return []string{completedMessage("Cursor", event["duration_ms"])}
		}
	case "opencode":
		label := p.Label
		if label == "" {
			label = "Agent"
		}
		if typeName == "step_start" {
			return []string{startedMessage(label, event, "sessionID")}
		}
		if typeName == "tool_use" {
			part, _ := event["part"].(map[string]any)
			id, _ := part["callID"].(string)
			if id != "" && p.seen[id] {
				return nil
			}
			p.seen[id] = true
			return []string{summarizeOpenCodeTool(part)}
		}
		if typeName == "step_finish" {
			part, _ := event["part"].(map[string]any)
			if part["reason"] == "stop" {
				return []string{"[hatch] " + label + " completed"}
			}
		}
	}
	return nil
}

func (p *ProgressParser) claudeTools(event map[string]any) []string {
	message, _ := event["message"].(map[string]any)
	var result []string
	for _, raw := range sliceValue(message["content"]) {
		block, _ := raw.(map[string]any)
		if block["type"] != "tool_use" {
			continue
		}
		id, _ := block["id"].(string)
		if id != "" && p.seen[id] {
			continue
		}
		p.seen[id] = true
		name, _ := block["name"].(string)
		if strings.HasPrefix(name, "mcp__claude_ai_Slack__") {
			continue
		}
		input, _ := block["input"].(map[string]any)
		detail := firstString(input, "description", "command", "file_path", "path")
		if detail != "" {
			result = append(result, fmt.Sprintf("[hatch] %s: %s", defaultName(name, "tool"), compact(detail)))
		} else {
			result = append(result, "[hatch] "+defaultName(name, "tool"))
		}
	}
	return result
}

func startedMessage(label string, event map[string]any, idName string) string {
	id, _ := event[idName].(string)
	if id != "" {
		if len(id) > 8 {
			id = id[:8]
		}
		return fmt.Sprintf("[hatch] %s started (session %s)", label, id)
	}
	return "[hatch] " + label + " started"
}

func completedMessage(label string, raw any) string {
	switch value := raw.(type) {
	case float64:
		return fmt.Sprintf("[hatch] %s completed in %.1fs", label, value/1000)
	case int:
		return fmt.Sprintf("[hatch] %s completed in %.1fs", label, float64(value)/1000)
	default:
		return "[hatch] " + label + " completed"
	}
}

func summarizeCursorTool(call map[string]any) string {
	for name, raw := range call {
		details, _ := raw.(map[string]any)
		args, _ := details["args"].(map[string]any)
		if value := firstString(args, "path", "filePath", "command", "query"); value != "" {
			return fmt.Sprintf("[hatch] Cursor %s: %s", name, compact(value))
		}
		return "[hatch] Cursor " + name
	}
	return "[hatch] Cursor tool"
}

func summarizeOpenCodeTool(part map[string]any) string {
	name, _ := part["tool"].(string)
	name = defaultName(name, "tool")
	if title, _ := part["title"].(string); title != "" {
		return fmt.Sprintf("[hatch] %s: %s", name, compact(title))
	}
	state, _ := part["state"].(map[string]any)
	input, _ := state["input"].(map[string]any)
	if value := firstString(input, "description", "command", "filePath", "path", "pattern", "query"); value != "" {
		return fmt.Sprintf("[hatch] %s: %s", name, compact(value))
	}
	return "[hatch] " + name
}

func sliceValue(value any) []any { result, _ := value.([]any); return result }
func firstString(values map[string]any, names ...string) string {
	for _, name := range names {
		if value, ok := values[name].(string); ok && value != "" {
			return value
		}
	}
	return ""
}
func defaultName(value, fallback string) string {
	if value == "" {
		return fallback
	}
	return value
}
func compact(value string) string {
	value = strings.Join(strings.Fields(value), " ")
	if len(value) > 180 {
		return value[:179] + "..."
	}
	return value
}
