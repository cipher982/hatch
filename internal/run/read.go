package run

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"
)

const maxMetadataBytes = 4 << 20

type Record struct {
	Kind        string                 `json:"kind"`
	Path        string                 `json:"path"`
	Files       []string               `json:"files,omitempty"`
	Observation *InspectionObservation `json:"observation,omitempty"`
	Manifest    *Manifest              `json:"manifest,omitempty"`
	Legacy      map[string]any         `json:"legacy,omitempty"`
	Raw         map[string]any         `json:"-"`
}

type InspectionObservation struct {
	ObservedAt         time.Time `json:"observed_at"`
	PID                int       `json:"pid"`
	ProcessAlive       *bool     `json:"process_alive"`
	StartIdentityMatch *bool     `json:"start_identity_match"`
	SuspectedOrphan    bool      `json:"suspected_orphan"`
}

type Summary struct {
	RunID            string                 `json:"run_id"`
	Kind             string                 `json:"kind"`
	Path             string                 `json:"path"`
	Lifecycle        string                 `json:"lifecycle"`
	Outcome          string                 `json:"outcome"`
	Surface          string                 `json:"surface"`
	Model            string                 `json:"model,omitempty"`
	Title            string                 `json:"title,omitempty"`
	CWD              string                 `json:"cwd,omitempty"`
	Provenance       *Provenance            `json:"provenance,omitempty"`
	CreatedAt        string                 `json:"created_at,omitempty"`
	Capture          string                 `json:"capture,omitempty"`
	NativeID         *string                `json:"native_id,omitempty"`
	Preview          string                 `json:"preview,omitempty"`
	PreviewTruncated bool                   `json:"preview_truncated,omitempty"`
	OutputBytes      int64                  `json:"output_bytes,omitempty"`
	Answer           string                 `json:"answer,omitempty"`
	TimeSource       string                 `json:"time_source,omitempty"`
	MatchedIn        []string               `json:"matched_in,omitempty"`
	Observation      *InspectionObservation `json:"observation,omitempty"`
	SortTime         time.Time              `json:"-"`
	rawPath          string
	rawCWD           string
	rawProvenance    *Provenance
}

type discoveryDiagnostics struct {
	Skipped  int      `json:"skipped,omitempty"`
	Warnings []string `json:"warnings,omitempty"`
}

func (d *discoveryDiagnostics) addSkip(err error) {
	d.Skipped++
	if err != nil && len(d.Warnings) < 8 {
		d.Warnings = append(d.Warnings, boundString(err.Error(), 256))
	}
}
func (d *discoveryDiagnostics) addWarning(err error) {
	if err != nil && len(d.Warnings) < 8 {
		d.Warnings = append(d.Warnings, boundString(err.Error(), 256))
	}
}

func ListRecords(root, expertCache string) ([]Summary, error) {
	result, _, err := discoverRecords(root, expertCache)
	return result, err
}

func discoverRecords(root, expertCache string) ([]Summary, discoveryDiagnostics, error) {
	result := []Summary{}
	diagnostics := discoveryDiagnostics{}
	entries, err := os.ReadDir(root)
	if err != nil && !os.IsNotExist(err) {
		return nil, diagnostics, err
	}
	for _, entry := range entries {
		if !entry.IsDir() || entry.Type()&os.ModeSymlink != 0 {
			continue
		}
		path := filepath.Join(root, entry.Name())
		if regularNonSymlinkFile(filepath.Join(path, "manifest.json")) {
			record, err := ReadMetadata(path)
			if err != nil || record.Manifest == nil {
				diagnostics.addSkip(err)
				continue
			}
			if !metadataIdentityValid(record.Manifest.RunID, record.Manifest.Provenance) {
				diagnostics.addSkip(fmt.Errorf("pathological run identity in %s", boundString(path, 256)))
				continue
			}
			result = append(result, summaryFromRecord(record))
		} else if regularNonSymlinkFile(filepath.Join(path, "metadata.json")) {
			legacy, err := readJSONObject(filepath.Join(path, "metadata.json"))
			if err != nil || legacy["artifact_kind"] != "hatch_opencode_run" {
				diagnostics.addSkip(err)
				continue
			}
			id := readString(legacy["run_id"])
			if id == "" {
				id = entry.Name()
			}
			if !metadataIdentityValid(id, nil) || !legacyIdentityValid(legacy) {
				diagnostics.addSkip(fmt.Errorf("pathological legacy identity in %s", boundString(path, 256)))
				continue
			}
			result = append(result, summaryFromLegacy("legacy_opencode", path, id, legacy))
		}
	}
	cacheEntries, cacheErr := os.ReadDir(expertCache)
	if cacheErr == nil {
		for _, entry := range cacheEntries {
			if entry.IsDir() || entry.Type()&os.ModeSymlink != 0 || filepath.Ext(entry.Name()) != ".json" {
				continue
			}
			path := filepath.Join(expertCache, entry.Name())
			legacy, err := readJSONObject(path)
			id := readString(legacy["response_id"])
			if err != nil || id == "" {
				diagnostics.addSkip(err)
				continue
			}
			if !metadataIdentityValid(id, nil) || !legacyIdentityValid(legacy) {
				diagnostics.addSkip(fmt.Errorf("pathological legacy identity in %s", boundString(path, 256)))
				continue
			}
			result = append(result, summaryFromLegacy("legacy_expert", path, id, legacy))
		}
	} else if !os.IsNotExist(cacheErr) {
		diagnostics.addWarning(cacheErr)
	}
	sort.SliceStable(result, func(i, j int) bool {
		if !result[i].SortTime.IsZero() && !result[j].SortTime.IsZero() && !result[i].SortTime.Equal(result[j].SortTime) {
			return result[i].SortTime.After(result[j].SortTime)
		}
		if result[i].SortTime.IsZero() != result[j].SortTime.IsZero() {
			return !result[i].SortTime.IsZero()
		}
		return result[i].RunID > result[j].RunID
	})
	return result, diagnostics, nil
}

func InspectRecord(root, expertCache, id string) (Record, error) {
	if id == "" || len(id) > 512 || filepath.Base(id) != id || id == "." || id == ".." {
		return Record{}, fmt.Errorf("invalid run id %q", boundString(id, 512))
	}
	path := filepath.Join(root, id)
	if info, err := os.Lstat(path); err == nil && info.IsDir() && info.Mode()&os.ModeSymlink == 0 {
		return ReadMetadata(path)
	}
	if path, ok := directExpertRecord(expertCache, id); ok {
		legacy, err := readJSONObject(path)
		return Record{Kind: "legacy_expert", Path: path, Legacy: legacy}, err
	}
	// Resolve legacy cache records without recursively walking run artifacts.
	summaries, err := ListRecords(root, expertCache)
	if err != nil {
		return Record{}, err
	}
	for _, summary := range summaries {
		if summary.RunID != id {
			continue
		}
		switch summary.Kind {
		case "legacy_opencode":
			legacy, err := readJSONObject(filepath.Join(summary.Path, "metadata.json"))
			return Record{Kind: summary.Kind, Path: summary.Path, Legacy: legacy}, err
		case "legacy_expert":
			legacy, err := readJSONObject(summary.Path)
			return Record{Kind: summary.Kind, Path: summary.Path, Legacy: legacy}, err
		default:
			return ReadMetadata(summary.Path)
		}
	}
	return Record{}, fmt.Errorf("run %q not found", id)
}

// ReadMetadata reads only the manifest (or legacy metadata) and never inventories
// the artifact directory. ReadRecord retains Raw for integrity callers.
func ReadMetadata(path string) (Record, error) { return readRecord(path, false) }

func ReadRecord(path string) (Record, error) { return readRecord(path, true) }

func readRecord(path string, retainRaw bool) (Record, error) {
	info, err := os.Lstat(path)
	if err != nil || !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return Record{}, fmt.Errorf("unsafe or missing run directory: %s", boundString(path, 1024))
	}
	manifestPath := filepath.Join(path, "manifest.json")
	data, err := readJSONBytes(manifestPath)
	if err != nil {
		if legacy, legacyErr := readJSONObject(filepath.Join(path, "metadata.json")); legacyErr == nil && legacy["artifact_kind"] == "hatch_opencode_run" {
			record := Record{Kind: "legacy_opencode", Path: path, Legacy: legacy}
			if retainRaw {
				record.Raw = legacy
			}
			return record, nil
		}
		return Record{}, err
	}
	var manifest Manifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		return Record{}, err
	}
	if manifest.SchemaVersion != 1 {
		return Record{}, fmt.Errorf("unsupported manifest schema version %d", manifest.SchemaVersion)
	}
	normalizeUnknownEnums(&manifest)
	record := Record{Kind: "hatch_run", Path: path,
		Observation: inspectNonterminalProcess(&manifest), Manifest: &manifest}
	if retainRaw {
		var raw map[string]any
		if err := json.Unmarshal(data, &raw); err != nil {
			return Record{}, err
		}
		record.Raw = raw
	}
	return record, nil
}

func summaryFromRecord(record Record) Summary {
	m := record.Manifest
	s := Summary{RunID: boundString(m.RunID, 512), Kind: record.Kind, Path: boundString(record.Path, 1024),
		Lifecycle: boundString(string(m.Lifecycle), 64), Surface: boundString(m.Surface, 256), Model: boundString(m.Model, 256),
		Title: boundString(m.Title, 160), CWD: boundString(m.CWD, 1024), Capture: boundString(m.Capture.State, 64),
		NativeID: boundedPtr(m.ProviderState.NativeID, 512), Observation: record.Observation, OutputBytes: m.Result.OutputBytes,
		SortTime: m.CreatedAt, TimeSource: "created_at", rawPath: record.Path, rawCWD: m.CWD, rawProvenance: m.Provenance}
	if m.Outcome != nil {
		s.Outcome = boundString(string(*m.Outcome), 64)
	}
	if m.CreatedAt.IsZero() {
		s.CreatedAt = ""
		s.SortTime = time.Time{}
		s.TimeSource = "unknown"
	} else {
		s.CreatedAt = m.CreatedAt.UTC().Format(time.RFC3339Nano)
	}
	s.Answer = answerState(s.Lifecycle, s.Outcome, m.Result.Output)
	s.Provenance = m.Provenance
	if s.Provenance != nil {
		s.Provenance = boundedProvenance(s.Provenance)
	}
	return s
}
func summaryFromLegacy(kind, path, id string, legacy map[string]any) Summary {
	s := Summary{RunID: boundString(id, 512), Kind: kind, Path: boundString(path, 1024),
		Lifecycle: "terminal", Surface: "legacy.opencode", CWD: boundString(legacyCWD(legacy), 1024),
		TimeSource: "unknown", Answer: "unknown", rawPath: path, rawCWD: legacyCWD(legacy)}
	if kind == "legacy_expert" {
		s.Surface = "expert"
		s.NativeID = boundedPtr(optionalString(legacy["response_id"]), 512)
		s.Outcome = boundString(readString(legacy["status"]), 64)
		text, present := legacyExpertResult(legacy)
		if present && text != "" {
			s.OutputBytes = int64(len(text))
			if s.Outcome == "completed" || s.Outcome == string(OutcomeSucceeded) || s.Outcome == string(OutcomeSucceededWarnings) {
				s.Answer = "complete"
			} else {
				s.Answer = "partial"
			}
		}
	} else {
		s.Outcome = boundString(readString(legacy["outcome"]), 64)
		s.NativeID = boundedPtr(optionalString(legacy["session_id"]), 512)
	}
	if t, source := legacyTime(legacy); !t.IsZero() {
		s.SortTime, s.TimeSource = t, source
		if source == "created_at" {
			s.CreatedAt = t.UTC().Format(time.RFC3339Nano)
		}
	}
	if p := legacyProvenanceExact(legacy); p != nil {
		s.rawProvenance = p
		s.Provenance = boundedProvenance(p)
	}
	if s.Provenance != nil && s.Provenance.CallerCWD != "" {
		s.CWD = boundString(s.Provenance.CallerCWD, 1024)
	}
	return s
}

func answerState(lifecycle, outcome, output string) string {
	if lifecycle != string(LifecycleTerminal) {
		if output == "present" {
			return "partial"
		}
		return "absent"
	}
	if output != "present" {
		return "absent"
	}
	if outcome == string(OutcomeSucceeded) || outcome == string(OutcomeSucceededWarnings) {
		return "complete"
	}
	return "partial"
}

func boundedProvenance(p *Provenance) *Provenance {
	if p == nil {
		return nil
	}
	q := *p
	q.CallerCWD, q.CallerKind = boundString(q.CallerCWD, 1024), boundString(q.CallerKind, maxCallerKindBytes)
	q.CallerSessionID, q.CallerRequestID, q.ParentRunID = boundString(q.CallerSessionID, 512), boundString(q.CallerRequestID, 512), boundString(q.ParentRunID, 512)
	return &q
}

func legacyCWD(m map[string]any) string {
	for _, key := range []string{"cwd", "working_directory", "target_cwd"} {
		if v := strings.TrimSpace(readString(m[key])); v != "" {
			return v
		}
	}
	return ""
}

func boundedPtr(value *string, max int) *string {
	if value == nil {
		return nil
	}
	bounded := boundString(*value, max)
	return &bounded
}

func legacyTime(m map[string]any) (time.Time, string) {
	for _, item := range []struct{ key, source string }{{"updated_at", "updated_at"}, {"created_at", "created_at"}} {
		switch value := m[item.key].(type) {
		case json.Number:
			seconds, err := strconv.ParseInt(value.String(), 10, 64)
			if err == nil {
				parsed := time.Unix(seconds, 0).UTC()
				if year := parsed.Year(); year >= 1 && year <= 9999 {
					return parsed, item.source
				}
			}
		case string:
			if value = strings.TrimSpace(value); value != "" {
				if parsed, err := time.Parse(time.RFC3339Nano, value); err == nil {
					return parsed, item.source
				}
			}
		}
	}
	return time.Time{}, "unknown"
}
func metadataIdentityValid(runID string, p *Provenance) bool {
	if runID == "" || len(runID) > 512 {
		return false
	}
	if p == nil {
		return true
	}
	return len(p.CallerSessionID) <= 512 && len(p.CallerRequestID) <= 512 && len(p.ParentRunID) <= 512
}
func legacyIdentityValid(m map[string]any) bool {
	value, ok := m["provenance"].(map[string]any)
	if !ok {
		return true
	}
	for _, key := range []string{"caller_session_id", "caller_request_id", "parent_run_id"} {
		if len(readString(value[key])) > 512 {
			return false
		}
	}
	return true
}

func legacyProvenanceExact(m map[string]any) *Provenance {
	value, ok := m["provenance"].(map[string]any)
	if !ok {
		return nil
	}
	data, err := json.Marshal(value)
	if err != nil {
		return nil
	}
	var p Provenance
	if json.Unmarshal(data, &p) != nil {
		return nil
	}
	return &p
}

func boundString(value string, max int) string {
	if max < 0 {
		return ""
	}
	for len(value) > max {
		_, size := utf8.DecodeLastRuneInString(value)
		if size <= 0 {
			size = 1
		}
		value = value[:len(value)-size]
	}
	return strings.ToValidUTF8(value, "\uFFFD")
}

func inspectNonterminalProcess(manifest *Manifest) *InspectionObservation {
	if manifest.Lifecycle == LifecycleTerminal || manifest.Process == nil || manifest.Process.PID <= 0 {
		return nil
	}
	alive, known := processAlive(manifest.Process.PID)
	observation := &InspectionObservation{ObservedAt: time.Now().UTC(), PID: manifest.Process.PID}
	if known {
		observation.ProcessAlive = &alive
		observation.SuspectedOrphan = !alive
	}
	if manifest.Process.StartIdentity != nil {
		current := processStartIdentity(manifest.Process.PID)
		if current != "" {
			matches := current == *manifest.Process.StartIdentity
			observation.StartIdentityMatch = &matches
			if !matches {
				observation.SuspectedOrphan = true
			}
		}
	}
	return observation
}

func normalizeUnknownEnums(manifest *Manifest) {
	if !oneOfString(string(manifest.Lifecycle), "prepared", "running", "terminal") {
		manifest.Lifecycle = Lifecycle("unknown")
	}
	if manifest.Outcome != nil && !oneOfString(string(*manifest.Outcome), "succeeded", "succeeded_with_warnings", "failed", "timed_out", "cancelled", "launch_failed", "abandoned") {
		unknown := Outcome("unknown")
		manifest.Outcome = &unknown
	}
	if !oneOfString(manifest.Capture.State, "durable", "degraded", "disabled") {
		manifest.Capture.State = "unknown"
	}
	if !oneOfString(manifest.ProviderState.Retention, "hatch_preserved", "provider_owned", "remote_provider", "unavailable", "unknown") {
		manifest.ProviderState.Retention = "unknown"
	}
}

func readJSONBytes(path string) ([]byte, error) {
	if !regularNonSymlinkFile(path) {
		return nil, fmt.Errorf("unsafe or missing metadata file: %s", boundString(path, 1024))
	}
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	data, err := io.ReadAll(io.LimitReader(file, maxMetadataBytes+1))
	if err != nil {
		return nil, err
	}
	if len(data) > maxMetadataBytes {
		return nil, fmt.Errorf("metadata exceeds %d bytes", maxMetadataBytes)
	}
	return data, nil
}

func readJSONObject(path string) (map[string]any, error) {
	data, err := readJSONBytes(path)
	if err != nil {
		return nil, err
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	var result map[string]any
	if err := decoder.Decode(&result); err != nil {
		return nil, err
	}
	return result, nil
}

func regularNonSymlinkFile(path string) bool {
	info, err := os.Lstat(path)
	return err == nil && info.Mode().IsRegular()
}

func readString(value any) string { result, _ := value.(string); return result }
func optionalString(value any) *string {
	result := strings.TrimSpace(readString(value))
	if result == "" {
		return nil
	}
	return &result
}
func oneOfString(value string, choices ...string) bool {
	for _, choice := range choices {
		if value == choice {
			return true
		}
	}
	return false
}
