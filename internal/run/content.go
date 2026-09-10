package run

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"unicode/utf8"
)

const (
	defaultContentLimit   = 8192
	maxContentLimit       = 32768
	defaultInventoryLimit = 20
	maxInventoryLimit     = 100
	// maxInventoryWalkEntries bounds directory enumeration and metadata stats
	// for an automatically discovered inventory. A capped result is a lower
	// bound, not a complete inventory.
	maxInventoryWalkEntries = 10000
	inventoryReadDirBatch   = 256
)

type ContentOptions struct {
	Part   string
	Offset int64
	Limit  int
}

type ContentPage struct {
	RunID               string `json:"run_id"`
	Part                string `json:"part"`
	Lifecycle           string `json:"lifecycle,omitempty"`
	Outcome             string `json:"outcome,omitempty"`
	Answer              string `json:"answer,omitempty"`
	File                string `json:"file"`
	Path                string `json:"path"`
	TotalBytes          int64  `json:"total_bytes"`
	Offset              int64  `json:"offset"`
	Length              int64  `json:"length"`
	NextOffset          *int64 `json:"next_offset,omitempty"`
	Truncated           bool   `json:"truncated"`
	OffsetAdjusted      bool   `json:"offset_adjusted"`
	ReplacedInvalidUTF8 bool   `json:"replaced_invalid_utf8"`
	Content             string `json:"content"`
}

type FilePage struct {
	Files         []FileInfo `json:"files"`
	Offset        int        `json:"offset"`
	NextOffset    *int       `json:"next_offset,omitempty"`
	Truncated     bool       `json:"truncated"`
	WalkTruncated bool       `json:"walk_truncated,omitempty"`
}

type FileInfo struct {
	Name  string `json:"name"`
	Bytes int64  `json:"bytes"`
}

// ReadContent reads one bounded, manifest-selected part of a run. It never
// reads a complete large artifact into memory and does not modify evidence.
func ReadContent(root, expertCache, id string, opts ContentOptions) (ContentPage, error) {
	part := opts.Part
	if part == "" {
		part = "result"
	}
	if !knownContentPart(part) {
		return ContentPage{}, fmt.Errorf("unsupported content part %q", part)
	}
	if opts.Offset < 0 {
		return ContentPage{}, errors.New("content offset must be nonnegative")
	}
	limit, err := contentLimit(opts.Limit)
	if err != nil {
		return ContentPage{}, err
	}
	if !validRunID(id) {
		return ContentPage{}, fmt.Errorf("invalid run id %q", id)
	}

	// Keep the raw manifest available even when another metadata field is
	// absent or malformed. This is also deliberately independent of Files.
	if part == "manifest" || part == "result-json" {
		if runDir, ok := directRunDirectory(root, id); ok {
			name := partFileName(part)
			if path, ok := safeArtifactFile(runDir, name); ok {
				return readContentFile(runDir, id, part, name, path, opts.Offset, limit)
			}
		}
	}
	// Legacy expert records have a stable direct path. Resolve it before
	// falling back to registry discovery, which may contain unrelated or
	// malformed records.
	if path, ok := directExpertRecord(expertCache, id); ok {
		if part == "manifest" {
			return readContentFile(expertCache, id, part, filepath.Base(path), path, opts.Offset, limit)
		}
		if part == "result" {
			legacy, err := readJSONObject(path)
			if err != nil {
				return ContentPage{}, err
			}
			text, ok := legacyExpertResult(legacy)
			if !ok {
				return ContentPage{}, errors.New("legacy expert result text is unavailable")
			}
			page := readContentBytes(id, part, "response.output", path, []byte(text), opts.Offset, limit)
			page.Lifecycle, page.Outcome = "terminal", readString(legacy["status"])
			page.Answer = "partial"
			if page.Outcome == "completed" || page.Outcome == string(OutcomeSucceeded) || page.Outcome == string(OutcomeSucceededWarnings) {
				page.Answer = "complete"
			}
			return page, nil
		}
	}

	record, err := InspectRecord(root, expertCache, id)

	if err != nil {
		return ContentPage{}, err
	}
	if record.Kind == "legacy_expert" {
		if part != "result" {
			return ContentPage{}, fmt.Errorf("content part %q is unavailable for legacy expert run", part)
		}
		if !safeStandaloneFile(record.Path, expertCache) {
			return ContentPage{}, fmt.Errorf("unsafe legacy expert record: %s", record.Path)
		}
		text, ok := legacyExpertResult(record.Legacy)
		if !ok {
			return ContentPage{}, errors.New("legacy expert result text is unavailable")
		}
		page := readContentBytes(id, part, "response.output", record.Path, []byte(text), opts.Offset, limit)
		page.Lifecycle, page.Outcome = "terminal", readString(record.Legacy["status"])
		page.Answer = "partial"
		if page.Outcome == "completed" || page.Outcome == string(OutcomeSucceeded) || page.Outcome == string(OutcomeSucceededWarnings) {
			page.Answer = "complete"
		}
		return page, nil
	}
	if record.Kind == "legacy_opencode" && part == "manifest" {
		path, ok := safeArtifactFile(record.Path, "metadata.json")
		if !ok {
			return ContentPage{}, errors.New("unsafe or unavailable legacy metadata")
		}
		return readContentFile(record.Path, id, part, "metadata.json", path, opts.Offset, limit)
	}
	if record.Manifest == nil || record.Kind != "hatch_run" {
		return ContentPage{}, fmt.Errorf("run %q has no readable manifest", id)
	}
	name, err := manifestPartFile(record.Manifest, part)
	if err != nil {
		return ContentPage{}, err
	}
	path, ok := safeArtifactFile(record.Path, name)
	if !ok {
		return ContentPage{}, fmt.Errorf("unsafe or unavailable content file %q", name)
	}
	page, err := readContentFile(record.Path, id, part, name, path, opts.Offset, limit)
	if err != nil {
		return ContentPage{}, err
	}
	page.Lifecycle = string(record.Manifest.Lifecycle)
	if record.Manifest.Outcome != nil {
		page.Outcome = string(*record.Manifest.Outcome)
	}
	page.Answer = answerState(page.Lifecycle, page.Outcome, record.Manifest.Result.Output)
	return page, nil
}

func knownContentPart(part string) bool {
	switch part {
	case "result", "request", "stdout", "stderr", "manifest", "result-json", "evidence":
		return true
	default:
		return false
	}
}

func contentLimit(limit int) (int, error) {
	if limit == 0 {
		return defaultContentLimit, nil
	}
	if limit < 4 || limit > maxContentLimit {
		return 0, fmt.Errorf("content limit must be between 4 and %d bytes", maxContentLimit)
	}
	return limit, nil
}

func validRunID(id string) bool {
	return id != "" && filepath.Base(id) == id && id != "." && id != ".." && !strings.ContainsRune(id, filepath.Separator)
}

func directRunDirectory(root, id string) (string, bool) {
	if !validRunID(id) || root == "" {
		return "", false
	}
	info, err := os.Lstat(root)
	if err != nil || !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return "", false
	}
	dir := filepath.Join(root, id)
	info, err = os.Lstat(dir)
	if err != nil || !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return "", false
	}
	return dir, true
}
func directExpertRecord(cache, id string) (string, bool) {
	if !validRunID(id) || cache == "" {
		return "", false
	}
	info, err := os.Lstat(cache)
	if err != nil || !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return "", false
	}
	path := filepath.Join(cache, id+".json")
	if !safeStandaloneFile(path, cache) {
		return "", false
	}
	return path, true
}

func partFileName(part string) string {
	switch part {
	case "manifest":
		return "manifest.json"
	case "result-json":
		return "result.json"
	default:
		return ""
	}
}

func manifestPartFile(manifest *Manifest, part string) (string, error) {
	var name string
	switch part {
	case "result":
		if manifest.Result.OutputFile != nil {
			name = *manifest.Result.OutputFile
		}
	case "request":
		name = manifest.Invocation.RequestFile
	case "stdout":
		name = manifest.Capture.StdoutFile
	case "stderr":
		name = manifest.Capture.StderrFile
	case "manifest":
		name = "manifest.json"
	case "result-json":
		name = "result.json"
	case "evidence":
		name = manifest.Capture.EvidenceManifestFile
	}
	if name == "" {
		return "", fmt.Errorf("content part %q is unavailable", part)
	}
	if !safeRelativeFile(name) {
		return "", fmt.Errorf("unsafe content file %q", name)
	}
	return name, nil
}

func safeArtifactFile(root, name string) (string, bool) {
	if root == "" || !safeRelativeFile(name) {
		return "", false
	}
	base, err := os.Lstat(root)
	if err != nil || !base.IsDir() || base.Mode()&os.ModeSymlink != 0 {
		return "", false
	}
	current := root
	for _, component := range strings.Split(filepath.Clean(name), string(filepath.Separator)) {
		current = filepath.Join(current, component)
		info, err := os.Lstat(current)
		if err != nil || info.Mode()&os.ModeSymlink != 0 {
			return "", false
		}
		if component != filepath.Base(name) && !info.IsDir() {
			return "", false
		}
	}
	info, err := os.Lstat(current)
	if err != nil || !info.Mode().IsRegular() || info.Mode()&os.ModeSymlink != 0 {
		return "", false
	}
	return current, true
}

func safeStandaloneFile(path, root string) bool {
	info, err := os.Lstat(path)
	if err != nil || !info.Mode().IsRegular() || info.Mode()&os.ModeSymlink != 0 {
		return false
	}
	if root == "" {
		return true
	}
	base, err := os.Lstat(root)
	if err != nil || !base.IsDir() || base.Mode()&os.ModeSymlink != 0 {
		return false
	}
	absPath, err := filepath.Abs(path)
	if err != nil {
		return false
	}
	absRoot, err := filepath.Abs(root)
	if err != nil {
		return false
	}
	rel, err := filepath.Rel(absRoot, absPath)
	return err == nil && safeRelativeFile(rel) && filepath.Base(rel) == rel
}

func readContentFile(root, id, part, name, path string, offset int64, limit int) (ContentPage, error) {
	info, err := os.Lstat(path)
	if err != nil || !info.Mode().IsRegular() || info.Mode()&os.ModeSymlink != 0 {
		return ContentPage{}, fmt.Errorf("unsafe or unavailable content file %q", name)
	}
	absRoot, err := filepath.Abs(root)
	if err != nil {
		return ContentPage{}, err
	}
	absPath, err := filepath.Abs(path)
	if err != nil {
		return ContentPage{}, err
	}
	relative, err := filepath.Rel(absRoot, absPath)
	if err != nil || !safeRelativeFile(relative) {
		return ContentPage{}, fmt.Errorf("unsafe content path %q", name)
	}
	confined, err := os.OpenRoot(absRoot)
	if err != nil {
		return ContentPage{}, err
	}
	defer confined.Close()
	file, err := confined.Open(filepath.ToSlash(relative))
	if err != nil {
		return ContentPage{}, err
	}
	defer file.Close()
	openedInfo, err := file.Stat()
	if err != nil || !openedInfo.Mode().IsRegular() || !os.SameFile(info, openedInfo) {
		return ContentPage{}, fmt.Errorf("unsafe or unavailable content file %q", name)
	}
	return readContentReader(id, part, name, absPath, file, openedInfo.Size(), offset, limit)
}
func readContentReader(id, part, name, path string, file *os.File, total, offset int64, limit int) (ContentPage, error) {
	adjusted := false
	if offset > total {
		offset = total
		adjusted = true
	}
	readSize := int64(limit + utf8.UTFMax)
	if remaining := total - offset; remaining < readSize {
		readSize = remaining
	}
	if readSize < 0 {
		readSize = 0
	}
	buf := make([]byte, int(readSize))
	n, err := file.ReadAt(buf, offset)
	if err != nil && !errors.Is(err, io.EOF) {
		return ContentPage{}, err
	}
	buf = buf[:n]
	prefixSize := offset
	if prefixSize > 3 {
		prefixSize = 3
	}
	prefix := make([]byte, int(prefixSize))
	if prefixSize > 0 {
		_, _ = file.ReadAt(prefix, offset-prefixSize)
	}
	skipped := midRuneSkip(prefix, buf)
	if skipped > 0 {
		offset += int64(skipped)
		adjusted = true
		buf = buf[skipped:]
	}
	page, consumed, replaced := decodePage(buf, total-offset, limit)
	next := offset + consumed
	if next >= total {
		return ContentPage{RunID: id, Part: part, File: name, Path: path, TotalBytes: total, Offset: offset, Length: consumed, Truncated: false, OffsetAdjusted: adjusted, ReplacedInvalidUTF8: replaced, Content: page}, nil
	}
	return ContentPage{RunID: id, Part: part, File: name, Path: path, TotalBytes: total, Offset: offset, Length: consumed, NextOffset: int64Ptr(next), Truncated: true, OffsetAdjusted: adjusted, ReplacedInvalidUTF8: replaced, Content: page}, nil
}

func readContentBytes(id, part, name, path string, data []byte, offset int64, limit int) ContentPage {
	total := int64(len(data))
	adjusted := false
	if offset > total {
		offset = total
		adjusted = true
	}
	remaining := total - offset
	readSize := int64(limit + utf8.UTFMax)
	if remaining < readSize {
		readSize = remaining
	}
	prefixSize := offset
	if prefixSize > 3 {
		prefixSize = 3
	}
	skip := midRuneSkip(data[offset-prefixSize:offset], data[offset:offset+readSize])
	if skip > 0 {
		offset += int64(skip)
		adjusted = true
	}
	page, consumed, replaced := decodePage(data[offset:offset+readSize-int64(skip)], total-offset, limit)
	next := offset + consumed
	var nextPtr *int64
	if next < total {
		nextPtr = int64Ptr(next)
	}
	return ContentPage{RunID: id, Part: part, File: name, Path: path, TotalBytes: total, Offset: offset, Length: consumed, NextOffset: nextPtr, Truncated: nextPtr != nil, OffsetAdjusted: adjusted, ReplacedInvalidUTF8: replaced, Content: page}
}

func midRuneSkip(prefix, data []byte) int {
	if len(prefix) == 0 || len(data) == 0 {
		return 0
	}
	combined := append(append([]byte(nil), prefix...), data...)
	boundary := len(prefix)
	for start := boundary - 1; start >= 0 && boundary-start <= utf8.UTFMax; start-- {
		r, size := utf8.DecodeRune(combined[start:])
		if r == utf8.RuneError && size == 1 {
			continue
		}
		if size > boundary-start && start+size <= len(combined) {
			return start + size - boundary
		}
	}
	return 0
}

func int64Ptr(value int64) *int64 { return &value }

func isContinuation(value byte) bool { return value&0xc0 == 0x80 }

// decodePage consumes at most limit source bytes, backing off a complete rune
// that would cross the page boundary. Invalid source bytes become U+FFFD.
func decodePage(data []byte, totalRemaining int64, limit int) (string, int64, bool) {
	var out strings.Builder
	consumed := 0
	replaced := false
	for consumed < len(data) && consumed < limit {
		width := 1
		if data[consumed] >= 0xc2 && data[consumed] <= 0xdf {
			width = 2
		} else if data[consumed] >= 0xe0 && data[consumed] <= 0xef {
			width = 3
		} else if data[consumed] >= 0xf0 && data[consumed] <= 0xf4 {
			width = 4
		}
		if width > 1 && consumed+width > len(data) && int64(consumed+width) > totalRemaining {
			// At EOF an incomplete sequence consists of invalid bytes and each
			// byte remains reachable. Only a page read boundary backs off.
			width = 1
		} else if width > 1 && consumed+width > len(data) {
			break
		}
		r, size := utf8.DecodeRune(data[consumed:])
		if size == 0 {
			break
		}
		if r == utf8.RuneError && size == 1 {
			out.WriteRune(utf8.RuneError)
			replaced = true
			consumed++
			continue
		}
		if consumed+size > limit {
			break
		}
		out.Write(data[consumed : consumed+size])
		consumed += size
	}
	return out.String(), int64(consumed), replaced
}

// Inventory returns a bounded page of file metadata. It does not read file
// contents. A nil Files list means the caller explicitly requested inventory,
// so the safe run tree is walked; a populated list is treated as an already
// scoped discovery result.
func Inventory(record Record, offset, limit int) (FilePage, error) {
	if offset < 0 {
		return FilePage{}, errors.New("inventory offset must be nonnegative")
	}
	if limit == 0 {
		limit = defaultInventoryLimit
	}
	if limit < 0 || limit > maxInventoryLimit {
		return FilePage{}, fmt.Errorf("inventory limit must be between 1 and %d files", maxInventoryLimit)
	}
	if record.Path == "" {
		return FilePage{}, errors.New("record path is empty")
	}
	rootInfo, err := os.Lstat(record.Path)
	if err != nil || rootInfo.Mode()&os.ModeSymlink != 0 {
		return FilePage{}, fmt.Errorf("unsafe inventory root: %s", record.Path)
	}
	files := append([]string(nil), record.Files...)
	walkTruncated := false
	if files == nil {
		files, walkTruncated, err = inventoryWalk(record.Path)
		if err != nil {
			return FilePage{}, err
		}
	}
	sort.Strings(files)
	all := make([]FileInfo, 0, len(files))
	for _, name := range files {
		if !safeRelativeFile(name) {
			return FilePage{}, fmt.Errorf("unsafe inventory path %q", name)
		}
		var path string
		if rootInfo.Mode().IsRegular() {
			if name != filepath.Base(record.Path) {
				return FilePage{}, fmt.Errorf("unsafe inventory path %q", name)
			}
			path = record.Path
		} else {
			var ok bool
			path, ok = safeArtifactFile(record.Path, name)
			if !ok {
				return FilePage{}, fmt.Errorf("unsafe or unavailable inventory path %q", name)
			}
		}
		info, err := os.Stat(path)
		if err != nil {
			return FilePage{}, err
		}
		all = append(all, FileInfo{Name: name, Bytes: info.Size()})
	}
	if offset >= len(all) {
		return FilePage{Files: []FileInfo{}, Offset: offset, WalkTruncated: walkTruncated}, nil
	}
	end := offset + limit
	if end > len(all) {
		end = len(all)
	}
	page := FilePage{Files: all[offset:end], Offset: offset, WalkTruncated: walkTruncated}
	if end < len(all) {
		next := end
		page.NextOffset = &next
		page.Truncated = true
	}
	return page, nil
}

type inventoryDirectory struct {
	path string
	rel  string
}

// inventoryWalk uses bounded Readdirnames batches and one extra-entry
// sentinel. It returns at most maxInventoryWalkEntries files and reports
// whether any directory entries remained after the cap.
func inventoryWalk(root string) ([]string, bool, error) {
	info, err := os.Lstat(root)
	if err != nil {
		return nil, false, err
	}
	if info.Mode()&os.ModeSymlink != 0 {
		return nil, false, fmt.Errorf("unsafe inventory root: %s", root)
	}
	if info.Mode().IsRegular() {
		return []string{filepath.Base(root)}, false, nil
	}
	if !info.IsDir() {
		return nil, false, fmt.Errorf("unsafe inventory root: %s", root)
	}
	files := make([]string, 0)
	stack := []inventoryDirectory{{path: root}}
	seen := 0
	for len(stack) > 0 {
		last := len(stack) - 1
		dir := stack[last]
		stack = stack[:last]
		file, err := os.Open(dir.path)
		if err != nil {
			return nil, false, err
		}
		for {
			batchSize := inventoryReadDirBatch
			if remaining := maxInventoryWalkEntries + 1 - seen; remaining < batchSize {
				batchSize = remaining
			}
			if batchSize <= 0 {
				_ = file.Close()
				return files, true, nil
			}
			names, readErr := file.Readdirnames(batchSize)
			for _, name := range names {
				seen++
				if seen > maxInventoryWalkEntries {
					_ = file.Close()
					return files, true, nil
				}
				path := filepath.Join(dir.path, name)
				info, err := os.Lstat(path)
				if err != nil {
					_ = file.Close()
					return nil, false, err
				}
				if info.Mode()&os.ModeSymlink != 0 {
					continue
				}
				rel := filepath.Join(dir.rel, name)
				if info.IsDir() {
					stack = append(stack, inventoryDirectory{path: path, rel: rel})
					continue
				}
				if !info.Mode().IsRegular() {
					continue
				}
				rel = filepath.ToSlash(rel)
				if !safeRelativeFile(rel) {
					_ = file.Close()
					return nil, false, fmt.Errorf("unsafe inventory path %q", rel)
				}
				files = append(files, rel)
			}
			if errors.Is(readErr, io.EOF) {
				break
			}
			if readErr != nil {
				_ = file.Close()
				return nil, false, readErr
			}
		}
		if err := file.Close(); err != nil {
			return nil, false, err
		}
	}
	return files, false, nil
}

// ClipText bounds a display string without splitting UTF-8. The returned bool
// reports whether any source bytes were omitted. Invalid bytes are replaced
// with U+FFFD; controls are intentionally left untouched.
func ClipText(text string, limit int, fromTail bool) (string, bool) {
	if limit <= 0 {
		return "", len(text) > 0
	}
	if len(text) <= limit && utf8.ValidString(text) {
		return text, false
	}
	if !fromTail {
		data := []byte(text[:min(len(text), limit+utf8.UTFMax)])
		out, used, _ := normalizeBounded(data, limit)
		return out, used < len(text)
	}
	start, encodedBytes := len(text), 0
	for start > 0 {
		r, size := utf8.DecodeLastRuneInString(text[:start])
		width := size
		if r == utf8.RuneError && size == 1 {
			width = utf8.RuneLen(utf8.RuneError)
		}
		if encodedBytes+width > limit {
			break
		}
		start -= size
		encodedBytes += width
	}
	return strings.ToValidUTF8(text[start:], "\uFFFD"), start > 0
}

func normalizeBounded(data []byte, limit int) (string, int, bool) {
	var out strings.Builder
	used := 0
	replaced := false
	for used < len(data) {
		r, size := utf8.DecodeRune(data[used:])
		if size < 1 {
			size = 1
		}
		encoded := data[used : used+size]
		if r == utf8.RuneError && size == 1 {
			if out.Len()+utf8.RuneLen(utf8.RuneError) > limit {
				break
			}
			out.WriteRune(utf8.RuneError)
			replaced = true
			used++
			continue
		}
		if out.Len()+len(encoded) > limit {
			break
		}
		out.Write(encoded)
		used += size
	}
	return out.String(), used, replaced
}

func legacyExpertResult(raw map[string]any) (string, bool) {
	if raw == nil {
		return "", false
	}
	if response, ok := raw["response"].(map[string]any); ok {
		if text, ok := responseOutputText(response); ok {
			return text, true
		}
	}
	if text, ok := raw["output"].(string); ok {
		return text, true
	}
	if text, ok := responseOutputText(raw); ok {
		return text, true
	}
	return "", false
}

func responseOutputText(response map[string]any) (string, bool) {
	if text, ok := response["output"].(string); ok {
		return text, true
	}
	items, ok := response["output"].([]any)
	if !ok {
		return "", false
	}
	var chunks []string
	for _, raw := range items {
		item, ok := raw.(map[string]any)
		if !ok || item["type"] != "message" {
			continue
		}
		contents, _ := item["content"].([]any)
		for _, rawContent := range contents {
			content, ok := rawContent.(map[string]any)
			if !ok || content["type"] != "output_text" {
				continue
			}
			if text, ok := content["text"].(string); ok && text != "" {
				chunks = append(chunks, text)
			}
		}
	}
	if len(chunks) == 0 {
		return "", false
	}
	return strings.TrimSpace(strings.Join(chunks, "\n")), true
}
