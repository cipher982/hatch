package run

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const (
	defaultQueryLimit = 20
	maxQueryLimit     = 100
	maxRequestHead    = 4 << 10
	maxPageJSON       = 64 << 10
)

type QueryOptions struct {
	All        bool
	Session    string
	CallerKind string
	CallerCWD  string
	Under      string
	Request    string
	Parent     string
	Query      string
	Status     string
	Since      time.Time
	Until      time.Time
	Before     string
	Limit      int
}

type RunScope struct {
	Kind       string `json:"kind"`
	Value      string `json:"value"`
	CallerKind string `json:"caller_kind,omitempty"`
}

type QueryDiagnostics struct {
	Skipped  int      `json:"skipped,omitempty"`
	Warnings []string `json:"warnings,omitempty"`
}

type QueryCoverage struct {
	RequestHeadBytes int      `json:"request_head_bytes"`
	PreviewBytes     int      `json:"preview_bytes"`
	Fields           []string `json:"fields"`
}

type RunPage struct {
	Runs        []Summary        `json:"runs"`
	Scope       RunScope         `json:"scope"`
	QueryScope  QueryCoverage    `json:"query_scope"`
	NextCursor  string           `json:"next_cursor,omitempty"`
	Truncated   bool             `json:"truncated"`
	Diagnostics QueryDiagnostics `json:"diagnostics,omitempty"`
}

// QueryRecords discovers and filters only metadata. Request previews are read
// from request.txt only for query matching and cards that are actually emitted;
func QueryRecords(root, expertCache string, opts QueryOptions) (RunPage, error) {
	if err := validateQueryOptions(opts); err != nil {
		return RunPage{}, err
	}
	if opts.Session != "" && opts.Under != "" {
		return RunPage{}, fmt.Errorf("session and under scopes are mutually exclusive")
	}
	if opts.Limit <= 0 {
		opts.Limit = defaultQueryLimit
	}
	if opts.Limit > maxQueryLimit {
		opts.Limit = maxQueryLimit
	}
	all, diagnostics, err := discoverRecords(root, expertCache)
	if err != nil {
		return RunPage{}, err
	}
	page := RunPage{
		Runs:  make([]Summary, 0),
		Scope: queryScope(opts),
		QueryScope: QueryCoverage{RequestHeadBytes: maxRequestHead, PreviewBytes: 160,
			Fields: []string{"title", "run_id", "cwd", "surface", "model", "request_preview"}},
		Diagnostics: QueryDiagnostics{Skipped: diagnostics.Skipped, Warnings: append([]string(nil), diagnostics.Warnings...)},
	}
	cursorStart := 0
	if opts.Before != "" {
		cursorStart = -1
		for i := range all {
			if all[i].RunID == opts.Before {
				cursorStart = i + 1
				break
			}
		}
		if cursorStart < 0 {
			return RunPage{}, fmt.Errorf("unknown before cursor %q", boundString(opts.Before, 512))
		}
	}
	filtered := make([]Summary, 0, len(all))
	for _, summary := range all[cursorStart:] {
		if !matchesScope(summary, opts) || !matchesMetadata(summary, opts) {
			continue
		}
		if !matchesQuery(&summary, opts.Query) {
			continue
		}
		filtered = append(filtered, summary)
	}
	end := opts.Limit
	if end > len(filtered) {
		end = len(filtered)
	}
	page.Truncated = end < len(filtered)
	for _, summary := range filtered[:end] {
		prepareCard(&summary)
		page.Runs = append(page.Runs, summary)
	}
	if page.Truncated && len(page.Runs) > 0 {
		page.NextCursor = page.Runs[len(page.Runs)-1].RunID
	}
	if err := boundRunPage(&page); err != nil {
		return RunPage{}, err
	}
	return page, nil
}
func validateQueryOptions(opts QueryOptions) error {
	if len(opts.Session) > 512 || len(opts.Request) > 512 || len(opts.Parent) > 512 || len(opts.Before) > 512 {
		return fmt.Errorf("query identity exceeds 512 bytes")
	}
	if len(opts.CallerKind) > maxCallerKindBytes {
		return fmt.Errorf("caller kind exceeds %d bytes", maxCallerKindBytes)
	}
	if len(opts.CallerCWD) > 1024 || len(opts.Under) > 1024 {
		return fmt.Errorf("query path exceeds 1024 bytes")
	}
	if (opts.CallerCWD != "" && len(normalizePath(opts.CallerCWD)) > 1024) || (opts.Under != "" && len(normalizePath(opts.Under)) > 1024) {
		return fmt.Errorf("normalized query path exceeds 1024 bytes")
	}
	return nil
}

func queryScope(opts QueryOptions) RunScope {
	callerKind := opts.CallerKind
	if opts.All {
		return RunScope{Kind: "all", Value: "*", CallerKind: callerKind}
	}
	if opts.Session != "" {
		return RunScope{Kind: "session", Value: opts.Session, CallerKind: callerKind}
	}
	if opts.Under != "" {
		return RunScope{Kind: "under", Value: normalizePath(opts.Under), CallerKind: callerKind}
	}
	cwd := opts.CallerCWD
	if cwd == "" {
		cwd, _ = os.Getwd()
	}
	return RunScope{Kind: "cwd", Value: normalizePath(cwd), CallerKind: callerKind}
}

func matchesScope(s Summary, opts QueryOptions) bool {
	p := s.rawProvenance
	if p == nil {
		p = s.Provenance
	}
	cwdValue := s.rawCWD
	if cwdValue == "" {
		cwdValue = s.CWD
	}
	if opts.CallerKind != "" && (p == nil || p.CallerKind != opts.CallerKind) {
		return false
	}
	if opts.All {
		return true
	}
	if opts.Session != "" {
		if p == nil || p.CallerSessionID != opts.Session {
			return false
		}
		return true
	}
	if opts.Under != "" {
		under := normalizePath(opts.Under)
		if cwdValue != "" && pathWithin(under, normalizePath(cwdValue)) {
			return true
		}
		return p != nil && p.CallerCWD != "" && pathWithin(under, normalizePath(p.CallerCWD))
	}
	cwd := opts.CallerCWD
	if cwd == "" {
		cwd, _ = os.Getwd()
	}
	cwd = normalizePath(cwd)
	if p != nil && p.CallerCWD != "" {
		return normalizePath(p.CallerCWD) == cwd
	}
	return cwdValue != "" && normalizePath(cwdValue) == cwd
}

func matchesMetadata(s Summary, opts QueryOptions) bool {
	p := s.rawProvenance
	if p == nil {
		p = s.Provenance
	}
	if opts.Request != "" && (p == nil || p.CallerRequestID != opts.Request) {
		return false
	}
	if opts.Parent != "" && (p == nil || p.ParentRunID != opts.Parent) {
		return false
	}
	if opts.Status != "" && s.Lifecycle != opts.Status && s.Outcome != opts.Status && s.Capture != opts.Status {
		return false
	}
	if !opts.Since.IsZero() && (s.SortTime.IsZero() || s.SortTime.Before(opts.Since)) {
		return false
	}
	if !opts.Until.IsZero() && (s.SortTime.IsZero() || s.SortTime.After(opts.Until)) {
		return false
	}
	return true
}

func matchesQuery(s *Summary, query string) bool {
	if query == "" {
		return true
	}
	needle := strings.ToLower(boundString(query, maxRequestHead))
	cwd := s.rawCWD
	if cwd == "" {
		cwd = s.CWD
	}
	fields := []struct{ name, value string }{
		{"title", s.Title}, {"run_id", s.RunID}, {"cwd", cwd}, {"surface", s.Surface}, {"model", s.Model},
	}
	for _, field := range fields {
		if strings.Contains(strings.ToLower(field.value), needle) {
			s.MatchedIn = append(s.MatchedIn, field.name)
		}
	}
	if s.Kind != "legacy_expert" {
		path := s.rawPath
		if path == "" {
			path = s.Path
		}
		if request, _, ok := requestHead(path, maxRequestHead); ok && strings.Contains(strings.ToLower(request), needle) {
			s.MatchedIn = append(s.MatchedIn, "request_preview")
		}
	}
	return len(s.MatchedIn) != 0
}

func prepareCard(s *Summary) {
	if s.Preview != "" || s.Kind == "legacy_expert" {
		return
	}
	path := s.rawPath
	if path == "" {
		path = s.Path
	}
	if request, requestTruncated, ok := requestHead(path, 161); ok {
		s.Preview, s.PreviewTruncated = oneLinePreview(request, 160)
		s.PreviewTruncated = requestTruncated || s.PreviewTruncated
	}
}
func requestHead(path string, limit int) (string, bool, bool) {
	if limit <= 0 {
		return "", false, false
	}
	filePath := path
	if info, err := os.Lstat(path); err == nil && info.IsDir() && info.Mode()&os.ModeSymlink == 0 {
		filePath = filepath.Join(path, "request.txt")
	}
	if !regularNonSymlinkFile(filePath) {
		return "", false, false
	}
	file, err := os.Open(filePath)
	if err != nil {
		return "", false, false
	}
	defer file.Close()
	data, err := io.ReadAll(io.LimitReader(file, int64(limit)+1))
	if err != nil {
		return "", false, false
	}
	truncated := len(data) > limit
	if truncated {
		data = data[:limit]
	}
	return string(data), truncated, true
}

func oneLinePreview(value string, limit int) (string, bool) {
	value = strings.Map(func(r rune) rune {
		if r == '\n' || r == '\r' || r == '\t' {
			return ' '
		}
		if r < 0x20 {
			return ' '
		}
		return r
	}, value)
	value = strings.TrimSpace(value)
	if len(value) <= limit {
		return value, false
	}
	return boundString(value, limit), true
}

func pathWithin(parent, child string) bool {
	rel, err := filepath.Rel(parent, child)
	return err == nil && rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator)) && !filepath.IsAbs(rel)
}
func normalizePath(path string) string {
	if path == "" {
		return ""
	}
	absolute, err := filepath.Abs(path)
	if err != nil {
		return filepath.Clean(path)
	}
	if resolved, err := filepath.EvalSymlinks(absolute); err == nil {
		absolute = resolved
	}
	return filepath.Clean(absolute)
}

func boundRunPage(page *RunPage) error {
	for i := range page.Runs {
		page.Runs[i].RunID = boundString(page.Runs[i].RunID, 512)
		page.Runs[i].Path = boundString(page.Runs[i].Path, 1024)
		page.Runs[i].Title = boundString(page.Runs[i].Title, 160)
		page.Runs[i].CWD = boundString(page.Runs[i].CWD, 1024)
		page.Runs[i].Preview = boundString(page.Runs[i].Preview, 160)
	}
	hadRuns := len(page.Runs) > 0
	for len(marshalRunPage(page)) > maxPageJSON && len(page.Runs) > 0 {
		page.Runs = page.Runs[:len(page.Runs)-1]
		page.Truncated = true
	}
	if len(page.Runs) > 0 && page.Truncated {
		page.NextCursor = page.Runs[len(page.Runs)-1].RunID
	}
	if len(page.Runs) == 0 {
		page.NextCursor = ""
		page.Truncated = false
		if hadRuns {
			return fmt.Errorf("run page card exceeds %d bytes", maxPageJSON)
		}
	}
	if len(marshalRunPage(page)) > maxPageJSON {
		return fmt.Errorf("run page metadata exceeds %d bytes", maxPageJSON)
	}
	return nil
}
func marshalRunPage(page *RunPage) []byte { data, _ := json.Marshal(page); return data }
