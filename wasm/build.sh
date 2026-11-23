#!/usr/bin/env bash
# Builds the WebAssembly embeddings demo into build/wasm and optionally serves it.
# Run from anywhere: ./wasm/build.sh [serve]
# Uses the yzma checkout in $YZMA_SRC (default build/yzma), cloning it first if it is missing.
set -euo pipefail

PARQUET_SRC="${PARQUET_SRC:-/usr/local/sfomuseum/go-parquet}"

cd "$(dirname "$0")/.."
ROOT="$PWD"
OUT="$ROOT/build/wasm"
YZMA_SRC="${YZMA_SRC:-$ROOT/build/yzma}"

mkdir -p "$OUT"

if [ ! -d "$YZMA_SRC" ]; then
	git clone https://github.com/hybridgroup/yzma "$YZMA_SRC"
fi

# A workspace file resolves yzma to the local checkout and bypasses vendor/
# (which does not include llamawasm). It is recreated if it points elsewhere.
if [ ! -f build/go.work ] || ! grep -qF "$YZMA_SRC" build/go.work; then
	rm -f build/go.work
	(cd build && go work init .. "$YZMA_SRC")
fi
export GOWORK="$ROOT/build/go.work"
# The Parquet query module (wasm/query) uses go-parquet; the call is idempotent.
(cd build && go work use "$PARQUET_SRC")

# llama.cpp WebAssembly build + JS glue.
if [ ! -e "$OUT/yzma-loader.js" ]; then
	go run github.com/hybridgroup/yzma install -lib "$OUT" -os wasm
fi
cp "$(go list -m -f '{{.Dir}}' github.com/hybridgroup/yzma)/wasm/yzma-loader.js" "$OUT/"

GOOS=js GOARCH=wasm go build -o "$OUT/yzma-embeddings.wasm" ./wasm/embeddings
# The Parquet query module is separate from the embeddings code; -tags wasmjs selects go-parquet's browser range reader.
GOOS=js GOARCH=wasm go build -tags wasmjs -ldflags="-s -w" -o "$OUT/yzma-query.wasm" ./wasm/query
cp "$(go env GOROOT)/lib/wasm/wasm_exec.js" "$OUT/" # use the TinyGo copy if building with TinyGo
cp wasm/embeddings/embed-worker.js wasm/embeddings/index.html wasm/query/query.js wasm/embeddings/embeddings.init.js wasm/embeddings/embeddings.css wasm/query/query-worker.js wasm/query/bootstrap.5.3.8.min.css "$OUT/"

echo "Built $OUT"

if [ "${1:-}" = "serve" ]; then
	# Sets the COOP/COEP headers the multithreaded build needs.
	exec go run github.com/hybridgroup/yzma/wasm/serve -dir "$OUT"
fi
