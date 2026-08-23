package run

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

const (
	GarbageOpenCodeConfig = "opencode_config_runtime"
	GarbageOpenCodeCache  = "opencode_cache"
	GarbageProviderState  = "ephemeral_provider_state"
)

type GarbageClassReport struct {
	Paths        int   `json:"paths"`
	Files        int64 `json:"files"`
	LogicalBytes int64 `json:"logical_bytes"`
}

type GarbageReport struct {
	Root                   string                        `json:"root"`
	Applied                bool                          `json:"applied"`
	RunsScanned            int                           `json:"runs_scanned"`
	RunsSkippedNonterminal int                           `json:"runs_skipped_nonterminal"`
	RunsSkippedPinned      int                           `json:"runs_skipped_pinned"`
	Classes                map[string]GarbageClassReport `json:"classes"`
	TotalPaths             int                           `json:"total_paths"`
	TotalFiles             int64                         `json:"total_files"`
	TotalLogicalBytes      int64                         `json:"total_logical_bytes"`
	RemovedLogicalBytes    int64                         `json:"removed_logical_bytes"`
	Errors                 []string                      `json:"errors"`
}

// CollectGarbage removes only provider-created runtime material that is never
// included in the evidence manifest. Canonical run evidence and native state
// snapshots are deliberately outside this command's scope.
func CollectGarbage(root string, apply bool) (GarbageReport, error) {
	report := GarbageReport{Root: root, Applied: apply, Classes: map[string]GarbageClassReport{}, Errors: []string{}}
	rootInfo, err := os.Lstat(root)
	if os.IsNotExist(err) {
		return report, nil
	}
	if err != nil {
		return report, err
	}
	if !rootInfo.IsDir() || rootInfo.Mode()&os.ModeSymlink != 0 {
		return report, fmt.Errorf("unsafe run root: %s", root)
	}
	entries, err := os.ReadDir(root)
	if err != nil {
		return report, err
	}
	for _, entry := range entries {
		if !entry.IsDir() || entry.Type()&os.ModeSymlink != 0 {
			continue
		}
		runPath := filepath.Join(root, entry.Name())
		manifest, err := garbageManifest(runPath)
		if err != nil {
			continue
		}
		report.RunsScanned++
		if manifest.Lifecycle != LifecycleTerminal {
			report.RunsSkippedNonterminal++
			continue
		}
		if regularNonSymlinkFile(filepath.Join(runPath, ".hatch-pin")) {
			report.RunsSkippedPinned++
			continue
		}
		targets := []struct{ class, path string }{
			{GarbageOpenCodeConfig, filepath.Join(runPath, "provider", "opencode-config")},
			{GarbageOpenCodeCache, filepath.Join(runPath, "provider", "opencode-cache")},
			{GarbageProviderState, filepath.Join(runPath, "provider", "opencode")},
			{GarbageProviderState, filepath.Join(runPath, "provider", "codex")},
			{GarbageProviderState, filepath.Join(runPath, "provider", "pi")},
			{GarbageProviderState, filepath.Join(runPath, "provider", "omp")},
		}
		for _, target := range targets {
			files, bytes, exists, err := measureGarbageTree(target.path)
			if err != nil {
				report.Errors = append(report.Errors, fmt.Sprintf("%s: %v", target.path, err))
				continue
			}
			if !exists {
				continue
			}
			class := report.Classes[target.class]
			class.Paths++
			class.Files += files
			class.LogicalBytes += bytes
			report.Classes[target.class] = class
			report.TotalPaths++
			report.TotalFiles += files
			report.TotalLogicalBytes += bytes
			if apply {
				if err := os.RemoveAll(target.path); err != nil {
					report.Errors = append(report.Errors, fmt.Sprintf("%s: %v", target.path, err))
					continue
				}
				report.RemovedLogicalBytes += bytes
			}
		}
		if apply {
			_ = os.Remove(filepath.Join(runPath, "provider"))
		}
	}
	return report, nil
}

func garbageManifest(runPath string) (Manifest, error) {
	path := filepath.Join(runPath, "manifest.json")
	if !regularNonSymlinkFile(path) {
		return Manifest{}, fmt.Errorf("missing manifest")
	}
	data, err := readBoundedAuditFile(path)
	if err != nil {
		return Manifest{}, err
	}
	var manifest Manifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		return Manifest{}, err
	}
	return manifest, nil
}

func measureGarbageTree(root string) (files, bytes int64, exists bool, resultErr error) {
	info, err := os.Lstat(root)
	if os.IsNotExist(err) {
		return 0, 0, false, nil
	}
	if err != nil {
		return 0, 0, false, err
	}
	if !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return 0, 0, false, fmt.Errorf("garbage target is not a real directory")
	}
	err = filepath.WalkDir(root, func(path string, entry os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if entry.Type()&os.ModeSymlink != 0 || entry.IsDir() {
			return nil
		}
		info, err := entry.Info()
		if err != nil {
			return err
		}
		if info.Mode().IsRegular() {
			files++
			bytes += info.Size()
		}
		return nil
	})
	return files, bytes, true, err
}
