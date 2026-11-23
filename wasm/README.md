# Vector embeddings from WebAssembly

`embeddings/main.go` is the WASM counterpart of `YzmaEmbedder.TextEmbeddings` and `YzmaEmbedder.ImageEmbeddings` in `yzma.go`. Instead of `pkg/llama` (which `dlopen`s the llama.cpp shared libraries) it uses [`pkg/llamawasm`](https://github.com/hybridgroup/yzma/tree/main/pkg/llamawasm), which drives a separate Emscripten build of llama.cpp. The call names and order are the same:

| yzma.go (native)                              | wasm/embeddings/main.go (llamawasm)              |
| --------------------------------------------- | ------------------------------------------------ |
| `llama.Load(lib_path)`                        | `llamawasm.Load("")` (loader glue provides it)   |
| `llama.ModelLoadFromFile(path)`               | `llamawasm.FetchModelFile(...)` then `ModelLoadFromFile` |
| `ContextDefaultParams` + `Embeddings=1` + `PoolingType` | same                                   |
| `Tokenize` / `BatchGetOne` / `Decode`         | same                                             |
| `ModelNEmbdOut` + `GetEmbeddingsSeq`          | same                                             |
| `mtmd.*` (image: `InitFromFile`, `Tokenize`, `EncodeChunk`, `GetOutputEmbd`) | `llamawasm.Mtmd*` (same names with an `Mtmd` prefix) |
| `llama.BatchExt*` + `Process`                 | same                                             |
| L2 normalize in `generateResponse`            | same                                             |

Differences: models are fetched over HTTP into the module's in-memory filesystem (no local paths), there is no multi-slot pool (llama.cpp accepts one call at a time), and images are decoded by the page (canvas) and sent as RGBA pixels, so there is no `ResizeIfOversized`/`BitmapInitFromBuf` step; downscale large images in the page. Unlike `yzma.go`, the projector URL is passed explicitly rather than derived as `mmproj-<model file name>`.

Image embeddings need yzma **v1.33.0 or newer** (the `Mtmd*` chunk calls and `BatchExt*` are not in v1.32.0, which `go.mod` currently pins) and a llama.cpp WASM module with ABI 13+.

## Build

Everything is run from the root of this package (`go-embeddings-wasm`) and written to `build/wasm/` (git-ignored). It uses a yzma checkout in `build/yzma`, cloned from <https://github.com/hybridgroup/yzma> if it is not already there, and wired in with a Go workspace file so that `vendor/` (which does not include `llamawasm`) is bypassed and the yzma tools can be run in place. No files are edited in the yzma checkout.

```
./wasm/build.sh         # build into build/wasm
./wasm/build.sh serve   # build, then serve (sets the COOP/COEP headers the multithreaded build needs)
```

The script can be run from any directory. Set `YZMA_SRC` to use a yzma checkout other than `build/yzma` (it is not cloned if the directory exists). It:

1. Clones yzma into `build/yzma` if that directory is missing.
2. Creates `build/go.work` (recreated if it points at a different yzma checkout) so yzma resolves to the checkout and `vendor/` (which lacks `llamawasm`) is bypassed.
3. Downloads the llama.cpp WebAssembly build with `go run github.com/hybridgroup/yzma install -os wasm` (first run only) and copies `yzma-loader.js`, the one file taken from yzma, since `embed-worker.js` loads it from its own directory.
4. Builds `wasm/embeddings` with `GOOS=js GOARCH=wasm` and copies in `wasm_exec.js` (edit the script to use the TinyGo copy if building with TinyGo), `embed-worker.js` and `index.html`.


Open <http://localhost:8080/index.html>, paste a GGUF embedding model URL (for example an `embeddinggemma` or `nomic-embed-text` GGUF from Hugging Face), click **Load**, then **Embed**. For images, also paste the matching `mmproj` GGUF URL before loading, choose a file, and click **Embed image**.

## Model cache

Downloaded models are stored in the browser's Origin Private File System (`yzma-models/`, keyed by a SHA-256 of the URL), so later loads skip the download. A download only replaces the entry once it completes. If OPFS is unavailable the model is downloaded directly each time. The **Purge cache** button removes the entries for the model and projector URLs in the page.

## Local files

The model, `mmproj` and Parquet fields each have a file chooser as an alternative to the URL. A chosen file takes precedence over the URL. Local models are used directly and are not cached in OPFS; the worker message takes `file` and `mmprojFile` (`File` objects) alongside `url` and `mmproj`, which then act as `local://<name>` keys. A local Parquet file is read in place (`ParquetQuery.open` accepts a URL or a `File`), so no CORS or Range support is needed; its index cache is keyed by name, size and modified time.

## JavaScript API

The page talks to a Web Worker (`embed-worker.js`) that starts llama.cpp and the Go program:

```js
const worker = new Worker("./embed-worker.js");
worker.postMessage({ kind: "load", url: "https://.../model.gguf", pooling: "mean", mmproj: "https://.../mmproj-model.gguf" });
worker.postMessage({ kind: "purge", urls: ["https://.../model.gguf"] }); // remove from the OPFS cache
worker.postMessage({ kind: "embed", id: "1", text: "A walrus on a beach" });
worker.postMessage({ kind: "embed-image", id: "2", width, height, rgba }); // RGBA ArrayBuffer from a canvas

worker.onmessage = ({ data }) => {
  // kinds: ready, status, progress, loaded, embedding, purged, error
  if (data.kind === "embedding") console.log(data.id, data.embedding); // Float32Array, L2-normalized
};
```

## Searching a Parquet file in the browser

`wasm/query/` is a second, independent WebAssembly module (`yzma-query.wasm`, Go + `go-parquet`) with its own worker
(`query-worker.js`) and client (`query.js`, class `ParquetQuery`). It shares nothing with the yzma code; the page just passes it
an embedding vector and the model name.

1. **Open** reads only the Parquet footer, using HTTP range requests.
2. **Build** reads the `model` column, then the `embeddings` column of the rows whose `model` equals the loaded model's file
   name, L2-normalizes them into one in-memory matrix, and caches the matrix in OPFS (`query-cache`; use *Purge index cache*
   if the file changes). The `embeddings` column must be written `plain,list`.
3. **Search** does a brute-force cosine scan and then fetches `provider`, `depiction_id`, `subject_id` and `attributes` for the
   top 20 rows only. `similarity` in the results is the cosine distance (0 = identical).

The server hosting the Parquet file must send CORS headers and support `HEAD` and `Range` requests, and (because the page is
cross-origin isolated) must allow the `Range` header, for example `Access-Control-Allow-Headers: Range` and
`Access-Control-Allow-Origin: *`. `build.sh` builds the module with `-tags wasmjs`, which selects go-parquet's browser range
reader, and adds `$PARQUET_SRC` (default `/usr/local/sfomuseum/go-parquet`) to the workspace.
