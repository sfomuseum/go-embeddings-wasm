//go:build js && wasm && wasmjs

package main

import (
	"errors"
	"fmt"
	"io"
	"math"
	"slices"
	"sort"
	"strings"

	pq "github.com/parquet-go/parquet-go"
)

// resultRow is the subset of an embeddings Record fetched for the top hits. Embeddings are not
// read: they are already in memory.
type resultRow struct {
	Provider    string            `parquet:"provider" json:"provider"`
	DepictionId string            `parquet:"depiction_id" json:"depiction_id"`
	SubjectId   string            `parquet:"subject_id" json:"subject_id"`
	Attributes  map[string]string `parquet:"attributes" json:"attributes"`
}

// hit is a search result: a row's position in the file and its cosine distance to the query.
type hit struct {
	Row      int64
	Distance float32
}

// searchResult is a record returned to the page: the fetched columns plus its
// cosine distance from the query, as "similarity" (0 = identical).
type searchResult struct {
	resultRow
	Similarity float32 `json:"similarity"`
}

// index is an open Parquet file plus, once built, the normalized embeddings of the rows for one model.
type index struct {
	reader *blockReader
	file   *pq.File

	modelCol, embCol int // leaf column indexes
	// models[rowGroup][row] is an index into modelNames; filled lazily by scanModels.
	models     [][]uint16
	modelNames []string
	modelRows  []int

	matrix []float32 // count*dim, each row L2-normalized
	rows   []int64   // file row number of each matrix row
	dim    int
}

// openIndex opens the Parquet file at uri and reads only its footer. The file must
// have "model" and "embeddings" columns.
func openIndex(uri string) (*index, error) {
	r, err := newBlockReader(uri)
	if err != nil {
		return nil, err
	}
	f, err := pq.OpenFile(r, r.Size())
	if err != nil {
		return nil, fmt.Errorf("opening parquet: %w", err)
	}
	ix := &index{reader: r, file: f, modelCol: -1, embCol: -1}
	for i, path := range f.Schema().Columns() {
		switch path[0] {
		case "model":
			ix.modelCol = i
		case "embeddings":
			ix.embCol = i
		}
	}
	if ix.modelCol < 0 || ix.embCol < 0 {
		return nil, errors.New("parquet file needs 'model' and 'embeddings' columns")
	}
	return ix, nil
}

// scanModels reads the model column once, recording which model each row has.
func (ix *index) scanModels(progress func(float64)) error {
	if ix.models != nil {
		return nil
	}
	ix.reader.SetBlockSize(4<<20, 4)
	ids := map[string]uint16{}
	models := make([][]uint16, 0, len(ix.file.RowGroups()))
	buf := make([]pq.Value, 4096)

	for g, rg := range ix.file.RowGroups() {
		col := make([]uint16, 0, rg.NumRows())
		pages := rg.ColumnChunks()[ix.modelCol].Pages()
		for {
			p, err := pages.ReadPage()
			if errors.Is(err, io.EOF) {
				break
			}
			if err != nil {
				pages.Close()
				return fmt.Errorf("reading model column: %w", err)
			}
			vr := p.Values()
			for {
				n, err := vr.ReadValues(buf)
				for _, v := range buf[:n] {
					name := string(v.ByteArray())
					id, ok := ids[name]
					if !ok {
						id = uint16(len(ix.modelNames))
						ids[name] = id
						ix.modelNames = append(ix.modelNames, name)
						ix.modelRows = append(ix.modelRows, 0)
					}
					ix.modelRows[id]++
					col = append(col, id)
				}
				if err != nil {
					if errors.Is(err, io.EOF) {
						break
					}
					pq.Release(p)
					pages.Close()
					return fmt.Errorf("reading model values: %w", err)
				}
			}
			pq.Release(p)
		}
		pages.Close()
		models = append(models, col)
		progress(float64(g+1) / float64(len(ix.file.RowGroups())) * 0.05)
	}
	ix.models = models
	return nil
}

// modelSummary lists the model names found in the file with their row counts,
// for use in error messages.
func (ix *index) modelSummary() string {
	var parts []string
	for i, n := range ix.modelNames {
		parts = append(parts, fmt.Sprintf("%q (%d rows)", n, ix.modelRows[i]))
	}
	return strings.Join(parts, ", ")
}

// modelTokens splits a model name such as "embeddinggemma-2-Q8_0.gguf" or
// "ggml-org/embeddinggemma-2-GGUF:Q8_0" into lowercase words, ignoring "gguf".
func modelTokens(name string) map[string]bool {
	words := strings.FieldsFunc(strings.ToLower(name), func(r rune) bool {
		return !(r >= 'a' && r <= 'z' || r >= '0' && r <= '9')
	})
	tokens := make(map[string]bool, len(words))
	for _, w := range words {
		if w != "gguf" {
			tokens[w] = true
		}
	}
	return tokens
}

// matchModel returns the id of the model column value matching name: an exact match, or else
// the only value whose words include all of name's words. It returns -1 if there is none or
// the match is ambiguous.
func (ix *index) matchModel(name string) int {
	if id := slices.Index(ix.modelNames, name); id >= 0 {
		return id
	}
	want := modelTokens(name)
	if len(want) == 0 {
		return -1
	}
	found := -1
	for i, n := range ix.modelNames {
		have := modelTokens(n)
		ok := true
		for w := range want {
			if !have[w] {
				ok = false
				break
			}
		}
		if !ok {
			continue
		}
		if found >= 0 {
			return -1
		}
		found = i
	}
	return found
}

// build reads the embeddings of every row for model into a normalized in-memory matrix.
// An empty model selects the file's only model. It returns the model name used, the number of
// rows kept and the number skipped for having an unexpected dimension.
func (ix *index) build(model string, progress func(float64)) (used string, kept, skipped int, err error) {
	if err := ix.scanModels(progress); err != nil {
		return "", 0, 0, err
	}
	id := ix.matchModel(model)
	if model == "" && len(ix.modelNames) == 1 {
		id = 0
	}
	if id < 0 {
		if model == "" {
			return "", 0, 0, fmt.Errorf("file contains %d models, so load a matching embedding model first: %s", len(ix.modelNames), ix.modelSummary())
		}
		return "", 0, 0, fmt.Errorf("model %q not found in file; it contains: %s", model, ix.modelSummary())
	}

	ix.matrix, ix.rows, ix.dim = nil, make([]int64, 0, ix.modelRows[id]), 0
	ix.reader.SetBlockSize(8<<20, 4)

	rgs := ix.file.RowGroups()
	var groupStart int64

	for g, rg := range rgs {
		matched := make([]bool, rg.NumRows())
		hasRows := false
		for r, m := range ix.models[g] {
			if int(m) == id {
				matched[r], hasRows = true, true
			}
		}
		if hasRows {
			s, err := ix.scanEmbeddings(rg, matched, groupStart)
			skipped += s
			if err != nil {
				return "", 0, 0, err
			}
		}
		groupStart += rg.NumRows()
		progress(0.05 + float64(g+1)/float64(len(rgs))*0.95)
	}
	if len(ix.rows) == 0 {
		return "", 0, 0, fmt.Errorf("no usable embeddings for model %q", model)
	}
	return ix.modelNames[id], len(ix.rows), skipped, nil
}

// scanEmbeddings appends the matched rows of one row group to the matrix. The embeddings column
// is a repeated float: repetition level 0 starts a row and a definition level equal to the
// maximum marks a present element. Page data is read directly as []float32 for speed.
func (ix *index) scanEmbeddings(rg pq.RowGroup, matched []bool, groupStart int64) (skipped int, err error) {
	chunk := rg.ColumnChunks()[ix.embCol]
	pages := chunk.Pages()
	defer pages.Close()

	maxDef := byte(ix.maxDefinition())
	row := int64(-1)
	open := false
	start := 0

	finish := func() {
		if !open {
			return
		}
		open = false
		n := len(ix.matrix) - start
		if ix.dim == 0 && n > 0 {
			ix.dim = n
			// Rows of the model are assumed to be about this many; avoids repeated regrowth.
			ix.matrix = slices.Grow(ix.matrix, (cap(ix.rows)-len(ix.rows))*n)
		}
		if n == 0 || n != ix.dim {
			ix.matrix = ix.matrix[:start]
			skipped++
			return
		}
		normalize(ix.matrix[start:])
		ix.rows = append(ix.rows, groupStart+row)
	}

	for {
		p, err := pages.ReadPage()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return skipped, fmt.Errorf("reading embeddings: %w", err)
		}
		if p.Dictionary() != nil {
			pq.Release(p)
			return skipped, errors.New("embeddings column is dictionary-encoded; write it with 'plain,list'")
		}
		rep, def := p.RepetitionLevels(), p.DefinitionLevels()
		data := p.Data()
		floats := data.Float()
		fi := 0
		for i := range int(p.NumValues()) {
			if rep[i] == 0 {
				finish()
				row++
				if matched[row] {
					open, start = true, len(ix.matrix)
				}
			}
			if def == nil || def[i] == maxDef {
				if open {
					ix.matrix = append(ix.matrix, floats[fi])
				}
				fi++
			}
		}
		pq.Release(p)
	}
	finish()
	return skipped, nil
}

// maxDefinition returns the maximum definition level of the embeddings column.
func (ix *index) maxDefinition() int {
	for _, c := range ix.leaves() {
		if c.ColumnIndex == ix.embCol {
			return c.MaxDefinitionLevel
		}
	}
	return 0
}

// leaves returns the leaf columns of the file schema in column order.
func (ix *index) leaves() []pq.LeafColumn {
	var out []pq.LeafColumn
	for _, path := range ix.file.Schema().Columns() {
		if c, ok := ix.file.Schema().Lookup(path...); ok {
			out = append(out, c)
		}
	}
	return out
}

// normalize scales v in place to unit length so a dot product is the cosine
// similarity. A zero vector is left unchanged.
func normalize(v []float32) {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	if sum == 0 {
		return
	}
	inv := float32(1 / math.Sqrt(sum))
	for i := range v {
		v[i] *= inv
	}
}

// search returns the k rows closest (cosine distance) to the query, nearest first.
func (ix *index) search(query []float32, k int) ([]hit, error) {
	if ix.dim == 0 || len(ix.rows) == 0 {
		return nil, errors.New("no embeddings loaded; build or import an index first")
	}
	if len(query) != ix.dim {
		return nil, fmt.Errorf("query has %d dimensions but the index has %d; was it made with the same model?", len(query), ix.dim)
	}
	q := slices.Clone(query)
	normalize(q)

	k = max(1, min(k, len(ix.rows)))
	top := make([]hit, 0, k+1)
	dim := ix.dim
	for r := range ix.rows {
		v := ix.matrix[r*dim : (r+1)*dim]
		var s0, s1, s2, s3 float32
		i := 0
		for ; i+4 <= dim; i += 4 {
			s0 += v[i] * q[i]
			s1 += v[i+1] * q[i+1]
			s2 += v[i+2] * q[i+2]
			s3 += v[i+3] * q[i+3]
		}
		for ; i < dim; i++ {
			s0 += v[i] * q[i]
		}
		d := 1 - (s0 + s1 + s2 + s3)
		if len(top) == k && d >= top[k-1].Distance {
			continue
		}
		pos := sort.Search(len(top), func(j int) bool { return top[j].Distance > d })
		top = slices.Insert(top, pos, hit{Row: int64(r), Distance: d})
		if len(top) > k {
			top = top[:k]
		}
	}
	for i := range top {
		top[i].Row = ix.rows[top[i].Row]
	}
	return top, nil
}

// fetch reads the descriptive columns for each hit from the remote file.
func (ix *index) fetch(hits []hit) ([]searchResult, error) {
	ix.reader.SetBlockSize(256<<10, 64)
	r := pq.NewGenericReader[resultRow](ix.file)
	defer r.Close()

	out := make([]searchResult, 0, len(hits))
	buf := make([]resultRow, 1)
	for _, h := range hits {
		if err := r.SeekToRow(h.Row); err != nil {
			return nil, fmt.Errorf("seeking to row %d: %w", h.Row, err)
		}
		buf[0] = resultRow{}
		if n, err := r.Read(buf); n == 0 {
			return nil, fmt.Errorf("reading row %d: %w", h.Row, err)
		}
		out = append(out, searchResult{resultRow: buf[0], Similarity: h.Distance})
	}
	return out, nil
}
