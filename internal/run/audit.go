package run

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"
)

type FieldAudit struct {
	Eligible            int               `json:"eligible"`
	MinimumTotal        int               `json:"minimum_total"`
	Observed            int               `json:"observed"`
	ExcludedPreContract int               `json:"excluded_pre_contract"`
	Incomplete          int               `json:"incomplete"`
	NonSuccess          int               `json:"non_success"`
	NonSurfaced         int               `json:"non_surfaced"`
	Unsafe              int               `json:"unsafe"`
	MinimumSurface      int               `json:"minimum_surface"`
	Surfaces            map[string]int    `json:"surfaces"`
	UnsafeRuns          []FieldAuditIssue `json:"unsafe_runs"`
}

type FieldAuditIssue struct {
	RunID  string `json:"run_id"`
	Reason string `json:"reason"`
}

func (audit FieldAudit) Passed() bool {
	if audit.Eligible < audit.MinimumTotal || audit.Unsafe != 0 {
		return false
	}
	for _, surface := range []string{"claude", "codex", "cursor", "openrouter", "expert"} {
		if audit.Surfaces[surface] < audit.MinimumSurface {
			return false
		}
	}
	return true
}

func AuditFieldEvidence(root string, minimumTotal, minimumSurface int) (FieldAudit, error) {
	audit := FieldAudit{MinimumTotal: minimumTotal, MinimumSurface: minimumSurface, Surfaces: map[string]int{}}
	if minimumTotal < 0 || minimumSurface < 0 {
		return audit, fmt.Errorf("field evidence minimums must be nonnegative")
	}
	if info, err := os.Lstat(root); err == nil && (!info.IsDir() || info.Mode()&os.ModeSymlink != 0 || !privateMode(info.Mode())) {
		return audit, fmt.Errorf("unsafe field evidence root: %s", root)
	}
	entries, err := os.ReadDir(root)
	if err != nil {
		if os.IsNotExist(err) {
			return audit, nil
		}
		return audit, err
	}
	for _, entry := range entries {
		if !entry.IsDir() || entry.Type()&os.ModeSymlink != 0 {
			continue
		}
		runDir := filepath.Join(root, entry.Name())
		manifestPath := filepath.Join(runDir, "manifest.json")
		manifestInfo, manifestInfoErr := os.Lstat(manifestPath)
		if os.IsNotExist(manifestInfoErr) {
			continue
		}
		if manifestInfoErr != nil || !manifestInfo.Mode().IsRegular() {
			audit.Observed++
			audit.addUnsafe(entry.Name(), "manifest is missing or unsafe")
			continue
		}
		manifest, err := readAuditManifest(manifestPath)
		if err != nil {
			audit.Observed++
			audit.addUnsafe(entry.Name(), "manifest is unreadable: "+err.Error())
			continue
		}
		if manifest.SchemaVersion != 1 {
			continue
		}
		audit.Observed++
		if manifest.Writer.Implementation != "go" || manifest.Writer.ContractRevision != 1 {
			audit.ExcludedPreContract++
			continue
		}
		if manifest.Lifecycle != LifecycleTerminal {
			audit.Incomplete++
			continue
		}
		runInfo, infoErr := entry.Info()
		if infoErr != nil || !privateMode(runInfo.Mode()) || !privateMode(manifestInfo.Mode()) {
			audit.addUnsafe(entry.Name(), "run directory or manifest permissions are unsafe")
			continue
		}
		if err := ValidateManifest(manifest); err != nil {
			audit.addUnsafe(entry.Name(), "manifest contract violation: "+err.Error())
			continue
		}
		if manifest.Capture.State != "durable" || capturePersistenceFailed(manifest) {
			audit.addUnsafe(entry.Name(), "terminal capture is degraded")
			continue
		}
		if manifest.Backend == "unknown" || manifest.Capture.ArtifactPath != runDir || manifest.RunID != entry.Name() {
			audit.addUnsafe(entry.Name(), "run identity, backend, or artifact path is not canonical")
			continue
		}
		if err := verifyTerminalSemantics(runDir, manifest); err != nil {
			audit.addUnsafe(entry.Name(), err.Error())
			continue
		}
		if err := verifyClosedEvidence(runDir, manifest); err != nil {
			audit.addUnsafe(entry.Name(), err.Error())
			continue
		}
		if manifest.Outcome == nil || (*manifest.Outcome != OutcomeSucceeded && *manifest.Outcome != OutcomeSucceededWarnings) {
			audit.NonSuccess++
			continue
		}
		surface, ok := fieldSurface(manifest.Surface)
		if !ok {
			audit.NonSurfaced++
			continue
		}
		audit.Eligible++
		audit.Surfaces[surface]++
	}
	return audit, nil
}

func (audit *FieldAudit) addUnsafe(runID, reason string) {
	audit.Unsafe++
	audit.UnsafeRuns = append(audit.UnsafeRuns, FieldAuditIssue{RunID: runID, Reason: reason})
}

func verifyTerminalSemantics(runDir string, manifest Manifest) error {
	if !regularPrivateFile(filepath.Join(runDir, "result.json")) {
		return fmt.Errorf("public result projection is missing or unsafe")
	}
	requestPath, err := evidencePath(runDir, manifest.Invocation.RequestFile)
	if err != nil {
		return err
	}
	requestDigest, err := hashFile(requestPath)
	if err != nil || requestDigest != manifest.Invocation.RequestSHA256 {
		return fmt.Errorf("request digest mismatch")
	}
	if manifest.Result.OutputFile != nil {
		resultPath, err := evidencePath(runDir, *manifest.Result.OutputFile)
		if err != nil {
			return err
		}
		info, err := os.Lstat(resultPath)
		if err != nil || !info.Mode().IsRegular() || info.Size() != manifest.Result.OutputBytes {
			return fmt.Errorf("result size mismatch")
		}
	}
	if manifest.Outcome != nil && (*manifest.Outcome == OutcomeSucceeded || *manifest.Outcome == OutcomeSucceededWarnings) {
		switch manifest.Execution {
		case "subprocess":
			if manifest.Process == nil || manifest.Process.ExitedAt == nil || manifest.Process.ExitCode == nil {
				return fmt.Errorf("successful subprocess lacks terminal process evidence")
			}
		case "http":
			if manifest.HTTP == nil || manifest.HTTP.CompletedAt == nil {
				return fmt.Errorf("successful HTTP run lacks terminal attempt evidence")
			}
		}
	}
	if manifest.Outcome != nil && *manifest.Outcome == OutcomeTimedOut && manifest.Execution == "subprocess" {
		if manifest.Process == nil || manifest.Process.TimeoutCleanup == nil || !manifest.Process.TimeoutCleanup.WaitBounded {
			return fmt.Errorf("timed-out subprocess lacks bounded cleanup evidence")
		}
	}
	if manifest.Outcome != nil && *manifest.Outcome == OutcomeCancelled && manifest.Execution == "subprocess" {
		if manifest.Process == nil || manifest.Process.CancelCleanup == nil || !manifest.Process.CancelCleanup.WaitBounded {
			return fmt.Errorf("cancelled subprocess lacks bounded cleanup evidence")
		}
	}
	return nil
}

func readAuditManifest(path string) (Manifest, error) {
	file, err := os.Open(path)
	if err != nil {
		return Manifest{}, err
	}
	defer file.Close()
	data, err := io.ReadAll(io.LimitReader(file, maxMetadataBytes+1))
	if err != nil {
		return Manifest{}, err
	}
	if len(data) > maxMetadataBytes {
		return Manifest{}, fmt.Errorf("manifest exceeds %d bytes", maxMetadataBytes)
	}
	var manifest Manifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		return Manifest{}, err
	}
	return manifest, nil
}

func capturePersistenceFailed(manifest Manifest) bool {
	for _, warning := range manifest.Warnings {
		if warning.Code == "capture_persistence_failed" {
			return true
		}
	}
	return false
}

func verifyClosedEvidence(runDir string, manifest Manifest) error {
	evidenceManifestPath, err := evidencePath(runDir, manifest.Capture.EvidenceManifestFile)
	if err != nil || !regularPrivateFile(evidenceManifestPath) || manifest.Capture.EvidenceSHA256 == nil {
		return fmt.Errorf("invalid evidence manifest")
	}
	contents, err := readBoundedAuditFile(evidenceManifestPath)
	if err != nil {
		return err
	}
	digest := sha256.Sum256(contents)
	if hex.EncodeToString(digest[:]) != *manifest.Capture.EvidenceSHA256 {
		return fmt.Errorf("evidence manifest digest mismatch")
	}

	required := map[string]bool{
		filepath.ToSlash(manifest.Invocation.RequestFile): true,
		filepath.ToSlash(manifest.Capture.StdoutFile):     true,
		filepath.ToSlash(manifest.Capture.StderrFile):     true,
	}
	if manifest.Result.OutputFile != nil {
		required[filepath.ToSlash(*manifest.Result.OutputFile)] = true
	}
	snapshotPrefixes := []string{}
	if manifest.ProviderState.SnapshotPath != nil {
		base := strings.TrimSuffix(filepath.ToSlash(*manifest.ProviderState.SnapshotPath), "/")
		snapshotPrefixes = []string{base + "/data/", base + "/state/"}
	}

	listed := []string{}
	scanner := bufio.NewScanner(strings.NewReader(string(contents)))
	for scanner.Scan() {
		line := scanner.Text()
		if len(line) < 67 || line[64:66] != "  " || !validSHA256(line[:64]) {
			return fmt.Errorf("malformed evidence entry")
		}
		name := line[66:]
		path, err := evidencePath(runDir, filepath.FromSlash(name))
		if err != nil || filepath.ToSlash(filepath.FromSlash(name)) != name || !regularNonSymlinkFile(path) {
			return fmt.Errorf("unsafe evidence entry %q", name)
		}
		allowed := required[name]
		for _, prefix := range snapshotPrefixes {
			allowed = allowed || strings.HasPrefix(name, prefix)
		}
		if !allowed {
			return fmt.Errorf("undeclared evidence entry %q", name)
		}
		fileDigest, err := hashFile(path)
		if err != nil || fileDigest != line[:64] {
			return fmt.Errorf("evidence hash mismatch for %q", name)
		}
		if len(listed) > 0 && listed[len(listed)-1] >= name {
			return fmt.Errorf("evidence manifest is unsorted or contains duplicates")
		}
		listed = append(listed, name)
		delete(required, name)
	}
	if err := scanner.Err(); err != nil {
		return err
	}
	if len(required) != 0 {
		return fmt.Errorf("evidence manifest omits required files")
	}

	actual := []string{}
	if manifest.Archive.ReceiptFile != nil && !regularPrivateFile(filepath.Join(runDir, *manifest.Archive.ReceiptFile)) {
		return fmt.Errorf("archive receipt is missing or unsafe")
	}
	err = filepath.WalkDir(runDir, func(path string, entry os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if path == runDir {
			return nil
		}
		if entry.Type()&os.ModeSymlink != 0 {
			return fmt.Errorf("artifact contains symlink %q", path)
		}
		if entry.IsDir() {
			return nil
		}
		info, err := entry.Info()
		if err != nil || !info.Mode().IsRegular() || !privateMode(info.Mode()) {
			return fmt.Errorf("artifact contains non-regular file %q", path)
		}
		relative, err := filepath.Rel(runDir, path)
		if err != nil {
			return err
		}
		name := filepath.ToSlash(relative)
		if name == "manifest.json" || name == "result.json" || name == manifest.Capture.EvidenceManifestFile || (manifest.Archive.ReceiptFile != nil && name == *manifest.Archive.ReceiptFile) {
			return nil
		}
		actual = append(actual, name)
		return nil
	})
	if err != nil {
		return err
	}
	sort.Strings(actual)
	if !slices.Equal(listed, actual) {
		return fmt.Errorf("artifact contains undeclared evidence")
	}
	return nil
}

func regularPrivateFile(path string) bool {
	info, err := os.Lstat(path)
	return err == nil && info.Mode().IsRegular() && privateMode(info.Mode())
}

func privateMode(mode os.FileMode) bool {
	return mode.Perm()&0o077 == 0
}

func readBoundedAuditFile(path string) ([]byte, error) {
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
		return nil, fmt.Errorf("evidence manifest exceeds %d bytes", maxMetadataBytes)
	}
	return data, nil
}

func hashFile(path string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer file.Close()
	hash := sha256.New()
	if _, err := io.Copy(hash, file); err != nil {
		return "", err
	}
	return hex.EncodeToString(hash.Sum(nil)), nil
}

func fieldSurface(surface string) (string, bool) {
	switch surface {
	case "claude.haiku", "claude.sonnet", "claude.opus", "claude.fable":
		return "claude", true
	case "codex.sol", "codex.terra", "codex.luna", "codex.nano", "codex.mini", "codex.max":
		return "codex", true
	case "cursor.grok":
		return "cursor", true
	case "openrouter.deepseek-v4-pro", "openrouter.kimi-k3":
		return "openrouter", true
	case "expert":
		return "expert", true
	default:
		return "", false
	}
}
