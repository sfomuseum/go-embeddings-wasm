//go:build js && wasm && wasmjs

// Command query searches a remote Parquet file of embeddings Records in the browser. It is
// independent of the yzma embedding code: give it an embedding vector and it returns the nearest
// records. Build with:
//
//	GOOS=js GOARCH=wasm go build -tags wasmjs -o build/wasm/yzma-query.wasm ./wasm/query
//
// It is driven by wasm/query/query-worker.js. See wasm/README.md.
package main

import (
	"encoding/binary"
	"encoding/json/v2"
	"errors"
	"fmt"
	"math"
	"syscall/js"
)

// ix is the file opened by queryOpen; it is nil until then.
var ix *index

// main registers the query* functions on the JS global scope, posts "ready", and
// blocks so the page can call into the program.
func main() {
	js.Global().Set("queryOpen", js.FuncOf(open))
	js.Global().Set("queryBuild", js.FuncOf(build))
	js.Global().Set("querySearch", js.FuncOf(search))
	js.Global().Set("queryMatrixInfo", js.FuncOf(matrixInfo))
	js.Global().Set("queryExport", js.FuncOf(export))
	js.Global().Set("queryImportBegin", js.FuncOf(importBegin))
	js.Global().Set("queryImportChunk", js.FuncOf(importChunk))

	post(map[string]any{"kind": "ready"})
	<-make(chan struct{})
}

// post sends a message to the page (the worker's postMessage).
func post(m map[string]any) {
	js.Global().Call("postMessage", m)
}

// progress posts a "progress" message with f (0 to 1) as a percentage.
func progress(f float64) {
	post(map[string]any{"kind": "progress", "text": fmt.Sprintf("%d", int(f*100))})
}

// promise runs fn on its own goroutine and returns a JS Promise for its result: parquet reads block on JS fetch callbacks, which
// can only run while the event loop is free.
func promise(fn func() (any, error)) any {
	var handler js.Func
	handler = js.FuncOf(func(this js.Value, args []js.Value) any {
		resolve, reject := args[0], args[1]
		go func() {
			defer handler.Release()
			v, err := fn()
			if err != nil {
				reject.Invoke(js.Global().Get("Error").New(err.Error()))
				return
			}
			resolve.Invoke(v)
		}()
		return nil
	})
	return js.Global().Get("Promise").New(handler)
}

// toJSON marshals v to a JSON string, the form in which results cross to JS.
func toJSON(v any) (any, error) {
	b, err := json.Marshal(v)
	return string(b), err
}

// open implements queryOpen(url): it opens the file, reads only its footer, and
// resolves with JSON {size, rows, row_groups}.
func open(this js.Value, args []js.Value) any {
	uri := args[0].String()
	return promise(func() (any, error) {
		next, err := openIndex(uri)
		if err != nil {
			return nil, err
		}
		ix = next
		return toJSON(map[string]any{"size": next.reader.Size(), "rows": next.file.NumRows(), "row_groups": len(next.file.RowGroups())})
	})
}

// build implements queryBuild(model): it loads and normalizes the embeddings of the
// rows for model (or the file's only model, if model is empty), and resolves with JSON
// {count, skipped, dim, model}.
func build(this js.Value, args []js.Value) any {
	model := args[0].String()
	return promise(func() (any, error) {
		if ix == nil {
			return nil, errors.New("open a file first")
		}
		used, kept, skipped, err := ix.build(model, progress)
		if err != nil {
			return nil, err
		}
		return toJSON(map[string]any{"count": kept, "skipped": skipped, "dim": ix.dim, "model": used})
	})
}

// search implements querySearch(Float32Array, k): it resolves with a JSON array of
// the k nearest records, nearest first.
func search(this js.Value, args []js.Value) any {
	raw := make([]byte, args[0].Get("byteLength").Int())
	js.CopyBytesToGo(raw, js.Global().Get("Uint8Array").New(args[0].Get("buffer"), args[0].Get("byteOffset"), args[0].Get("byteLength")))
	query := make([]float32, len(raw)/4)
	for i := range query {
		query[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	k := args[1].Int()

	return promise(func() (any, error) {
		if ix == nil {
			return nil, errors.New("open a file first")
		}
		hits, err := ix.search(query, k)
		if err != nil {
			return nil, err
		}
		results, err := ix.fetch(hits)
		if err != nil {
			return nil, err
		}
		return toJSON(results)
	})
}

// matrixInfo implements queryMatrixInfo(): it describes the in-memory matrix as JSON
// {count, dim} so JS can cache it.
func matrixInfo(this js.Value, args []js.Value) any {
	if ix == nil {
		return "{}"
	}
	s, _ := toJSON(map[string]any{"count": len(ix.rows), "dim": ix.dim})
	return s
}

// matrixSize returns the byte length of the "vec" (float32) or "idx" (int64) matrix.
func matrixSize(name string) int {
	switch name {
	case "vec":
		return len(ix.matrix) * 4
	case "idx":
		return len(ix.rows) * 8
	}
	return 0
}

// export implements queryExport(name, offset, length): it encodes a slice of the
// "vec" or "idx" bytes (little-endian) into a new Uint8Array. Offsets and lengths
// must be multiples of the element size (4 for vec, 8 for idx).
func export(this js.Value, args []js.Value) any {
	name := args[0].String()
	off, n := args[1].Int(), args[2].Int()
	n = max(min(n, matrixSize(name)-off), 0)
	buf := make([]byte, n)
	switch name {
	case "vec":
		for i := 0; i < n/4; i++ {
			binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(ix.matrix[off/4+i]))
		}
	case "idx":
		for i := 0; i < n/8; i++ {
			binary.LittleEndian.PutUint64(buf[i*8:], uint64(ix.rows[off/8+i]))
		}
	}
	out := js.Global().Get("Uint8Array").New(n)
	if n > 0 {
		js.CopyBytesToJS(out, buf)
	}
	return out
}

// importBegin implements queryImportBegin(count, dim): it allocates the matrix so
// queryImportChunk can fill it from a cache.
func importBegin(this js.Value, args []js.Value) any {
	if ix == nil {
		return "open a file first"
	}
	count, dim := args[0].Int(), args[1].Int()
	ix.matrix = make([]float32, count*dim)
	ix.rows = make([]int64, count)
	ix.dim = dim
	return nil
}

// importChunk implements queryImportChunk(name, offset, Uint8Array): it writes cached
// bytes into the matrix.
func importChunk(this js.Value, args []js.Value) any {
	name := args[0].String()
	off := args[1].Int()
	buf := make([]byte, args[2].Get("byteLength").Int())
	js.CopyBytesToGo(buf, args[2])
	switch name {
	case "vec":
		for i := 0; i < len(buf)/4; i++ {
			ix.matrix[off/4+i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[i*4:]))
		}
	case "idx":
		for i := 0; i < len(buf)/8; i++ {
			ix.rows[off/8+i] = int64(binary.LittleEndian.Uint64(buf[i*8:]))
		}
	}
	return nil
}
