package run

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
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
	RunID     string  `json:"run_id"`
	Kind      string  `json:"kind"`
	Path      string  `json:"path"`
	Lifecycle string  `json:"lifecycle"`
	Outcome   string  `json:"outcome"`
	Surface   string  `json:"surface"`
	CreatedAt string  `json:"created_at,omitempty"`
	Capture   string  `json:"capture,omitempty"`
	NativeID  *string `json:"native_id,omitempty"`
}

func ListRecords(root, expertCache string) ([]Summary, error) {
	result := []Summary{}
	entries, err := os.ReadDir(root)
	if err != nil && !os.IsNotExist(err) {
		return nil, err
	}
	for _, entry := range entries {
		if !entry.IsDir() || entry.Type()&os.ModeSymlink != 0 {
			continue
		}
		path := filepath.Join(root, entry.Name())
		if regularNonSymlinkFile(filepath.Join(path, "manifest.json")) {
			record, err := ReadRecord(path)
			if err != nil {
				continue
			}
			manifest := record.Manifest
			outcome := ""
			if manifest.Outcome != nil {
				outcome = string(*manifest.Outcome)
			}
			result = append(result, Summary{RunID: manifest.RunID, Kind: record.Kind, Path: path, Lifecycle: string(manifest.Lifecycle), Outcome: outcome, Surface: manifest.Surface, CreatedAt: manifest.CreatedAt.Format("2006-01-02T15:04:05.999999999Z07:00"), Capture: manifest.Capture.State, NativeID: manifest.ProviderState.NativeID})
		} else if regularNonSymlinkFile(filepath.Join(path, "metadata.json")) {
			legacy, err := readJSONObject(filepath.Join(path, "metadata.json"))
			if err != nil || legacy["artifact_kind"] != "hatch_opencode_run" {
				continue
			}
			id := readString(legacy["run_id"])
			if id == "" {
				id = entry.Name()
			}
			native := optionalString(legacy["session_id"])
			result = append(result, Summary{RunID: id, Kind: "legacy_opencode", Path: path, Lifecycle: "terminal", Outcome: readString(legacy["outcome"]), Surface: "legacy.opencode", NativeID: native})
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
			if err != nil || readString(legacy["response_id"]) == "" {
				continue
			}
			id := readString(legacy["response_id"])
			result = append(result, Summary{RunID: id, Kind: "legacy_expert", Path: path, Lifecycle: "terminal", Outcome: readString(legacy["status"]), Surface: "expert", NativeID: &id})
		}
	}
	sort.Slice(result, func(i, j int) bool { return result[i].RunID > result[j].RunID })
	return result, nil
}

func InspectRecord(root, expertCache, id string) (Record, error) {
	if id == "" || filepath.Base(id) != id || id == "." || id == ".." {
		return Record{}, fmt.Errorf("invalid run id %q", id)
	}
	path := filepath.Join(root, id)
	if info, err := os.Lstat(path); err == nil && info.IsDir() && info.Mode()&os.ModeSymlink == 0 {
		return ReadRecord(path)
	}
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
			return Record{Kind: summary.Kind, Path: summary.Path, Legacy: legacy, Raw: legacy}, err
		case "legacy_expert":
			legacy, err := readJSONObject(summary.Path)
			return Record{Kind: summary.Kind, Path: summary.Path, Legacy: legacy, Raw: legacy}, err
		default:
			return ReadRecord(summary.Path)
		}
	}
	return Record{}, fmt.Errorf("run %q not found", id)
}

func ReadRecord(path string) (Record, error) {
	info, err := os.Lstat(path)
	if err != nil || !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return Record{}, fmt.Errorf("unsafe or missing run directory: %s", path)
	}
	manifestPath := filepath.Join(path, "manifest.json")
	raw, err := readJSONObject(manifestPath)
	if err != nil {
		if legacy, legacyErr := readJSONObject(filepath.Join(path, "metadata.json")); legacyErr == nil && legacy["artifact_kind"] == "hatch_opencode_run" {
			return Record{Kind: "legacy_opencode", Path: path, Legacy: legacy, Raw: legacy}, nil
		}
		return Record{}, err
	}
	encoded, _ := json.Marshal(raw)
	var manifest Manifest
	if err := json.Unmarshal(encoded, &manifest); err != nil {
		return Record{}, err
	}
	if manifest.SchemaVersion != 1 {
		return Record{}, fmt.Errorf("unsupported manifest schema version %d", manifest.SchemaVersion)
	}
	normalizeUnknownEnums(&manifest)
	files, err := artifactFiles(path)
	if err != nil {
		return Record{}, err
	}
	return Record{
		Kind: "hatch_run", Path: path, Files: files,
		Observation: inspectNonterminalProcess(&manifest), Manifest: &manifest, Raw: raw,
	}, nil
}

func artifactFiles(root string) ([]string, error) {
	files := []string{}
	err := filepath.WalkDir(root, func(path string, entry os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if path == root {
			return nil
		}
		if entry.Type()&os.ModeSymlink != 0 {
			if entry.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		if entry.IsDir() {
			return nil
		}
		info, err := entry.Info()
		if err != nil {
			return err
		}
		if !info.Mode().IsRegular() {
			return nil
		}
		relative, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		files = append(files, filepath.ToSlash(relative))
		return nil
	})
	sort.Strings(files)
	return files, err
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

func readJSONObject(path string) (map[string]any, error) {
	if !regularNonSymlinkFile(path) {
		return nil, fmt.Errorf("unsafe or missing metadata file: %s", path)
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
