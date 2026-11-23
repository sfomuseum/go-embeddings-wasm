//go:build js && wasm && wasmjs

package main

import (
	"fmt"
	"io"
	"sync"

	"github.com/sfomuseum/go-parquet"
)

// blockReader is an io.ReaderAt that fetches fixed-size blocks from src (an HTTP range
// reader) and keeps a few of them. parquet-go issues many small reads; without this each
// would become its own HTTP request.
type blockReader struct {
	src       parquet.ReadCloserAt
	size      int64
	mu        sync.Mutex
	blockSize int64
	maxBlocks int
	blocks    map[int64][]byte
	order     []int64 // oldest first
}

// newBlockReader opens uri with parquet.OpenURI and wraps it in a blockReader with
// 512 KiB blocks, 8 of them kept.
func newBlockReader(uri string) (*blockReader, error) {
	src, size, err := parquet.OpenURI(uri)
	if err != nil {
		return nil, err
	}
	r := &blockReader{src: src, size: size}
	r.SetBlockSize(512<<10, 8)
	return r, nil
}

// SetBlockSize drops cached blocks and changes the request granularity: large for
// sequential column scans, small for random row lookups.
func (r *blockReader) SetBlockSize(blockSize int64, maxBlocks int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.blockSize = blockSize
	r.maxBlocks = maxBlocks
	r.blocks = map[int64][]byte{}
	r.order = nil
}

// Size returns the size of the file in bytes.
func (r *blockReader) Size() int64 { return r.size }

// Close closes the underlying range reader.
func (r *blockReader) Close() error { return r.src.Close() }

// ReadAt implements io.ReaderAt, serving reads from cached blocks and fetching
// missing ones. It returns io.EOF when the read reaches the end of the file.
func (r *blockReader) ReadAt(p []byte, off int64) (int, error) {
	if off < 0 {
		return 0, fmt.Errorf("negative offset")
	}
	r.mu.Lock()
	defer r.mu.Unlock()

	n := 0
	for n < len(p) {
		pos := off + int64(n)
		if pos >= r.size {
			return n, io.EOF
		}
		idx := pos / r.blockSize
		blk, err := r.block(idx)
		if err != nil {
			return n, err
		}
		n += copy(p[n:], blk[pos-idx*r.blockSize:])
	}
	return n, nil
}

// block returns block idx, fetching it (and evicting the oldest) if it is not cached.
// The caller must hold r.mu.
func (r *blockReader) block(idx int64) ([]byte, error) {
	if b, ok := r.blocks[idx]; ok {
		return b, nil
	}
	start := idx * r.blockSize
	length := min(r.blockSize, r.size-start)
	buf := make([]byte, length)
	if n, err := r.src.ReadAt(buf, start); err != nil && !(err == io.EOF && int64(n) == length) {
		return nil, fmt.Errorf("fetching bytes %d-%d: %w", start, start+length-1, err)
	}
	if len(r.order) >= r.maxBlocks {
		delete(r.blocks, r.order[0])
		r.order = r.order[1:]
	}
	r.blocks[idx] = buf
	r.order = append(r.order, idx)
	return buf, nil
}
