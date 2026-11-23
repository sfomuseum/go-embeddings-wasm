//go:build js && wasm

// Command embeddings generates text vector embeddings in a browser (or Node)
// using yzma's llamawasm package. It mirrors YzmaEmbedder.TextEmbeddings in
// yzma.go: tokenize, decode with an embeddings context, read the pooled
// sequence embedding, then L2-normalize it.
//
// Build with the standard Go toolchain (TinyGo is not supported):
//
//	GOOS=js GOARCH=wasm go build -o build/wasm/yzma-embeddings.wasm ./wasm/embeddings
//
// See wasm/README.md for serving it.
package main

import (
	"fmt"
	"math"
	"strings"
	"syscall/js"
	"time"

	"github.com/hybridgroup/yzma/pkg/llamawasm"
)

// Paths of the model and projector inside the llama.cpp module's filesystem.
const (
	modelPath  = "/models/model.gguf"
	mmprojPath = "/models/mmproj.gguf"
)

// The loaded model, its embeddings context, vocabulary, and, when a projector was
// loaded, the multimodal context.
var (
	model llamawasm.Model
	ctx   llamawasm.Context
	mctx  llamawasm.MtmdContext // zero unless a projector was loaded
	vocab llamawasm.Vocab
)

// main loads llama.cpp, registers yzmaLoadModel, yzmaEmbed and yzmaEmbedImage on the
// JS global scope, posts "ready", and blocks so the page can call into the program.
func main() {
	if err := llamawasm.Load(""); err != nil {
		post(map[string]any{"kind": "error", "text": err.Error()})
		return
	}

	llamawasm.LogSet(llamawasm.LogSilent())
	llamawasm.Init()

	js.Global().Set("yzmaLoadModel", js.FuncOf(loadModel))
	js.Global().Set("yzmaEmbed", js.FuncOf(embed))
	js.Global().Set("yzmaEmbedImage", js.FuncOf(embedImage))

	post(map[string]any{"kind": "ready", "text": fmt.Sprintf("backend: %s", llamawasm.Backend())})

	// Keep the program alive so the page can call into it.
	<-make(chan struct{})
}

// loadModel(url, pooling, mmprojURL) downloads a GGUF embedding model into the
// loadModel implements yzmaLoadModel(url, pooling, mmprojURL). It downloads a GGUF
// embedding model into the llama.cpp module's filesystem and creates an embeddings
// context for it. pooling is optional ("mean", "cls", "last", "none"; default "mean").
// mmprojURL is optional; when set, the projector is loaded for image embeddings
// (yzma.go derives it as "mmproj-<model file name>" next to the model).
func loadModel(this js.Value, args []js.Value) any {
	if len(args) < 1 {
		post(map[string]any{"kind": "error", "text": "loadModel needs a URL"})
		return nil
	}

	url := args[0].String()
	pooling := "mean"
	if len(args) > 1 && args[1].Truthy() {
		pooling = args[1].String()
	}
	mmprojURL := ""
	if len(args) > 2 && args[2].Truthy() {
		mmprojURL = args[2].String()
	}

	go func() {
		fetchURL := resolveCached(url, "")
		defer revokeBlobURL(fetchURL)

		err := llamawasm.FetchModelFile(modelPath, fetchURL, func(done, total int64) {
			if total > 0 {
				post(map[string]any{"kind": "progress", "text": fmt.Sprintf("%d%%", done*100/total)})
			}
		})
		if err != nil {
			post(map[string]any{"kind": "error", "text": err.Error()})
			return
		}

		if mmprojURL != "" {
			fetchURL := resolveCached(mmprojURL, "mmproj")
			defer revokeBlobURL(fetchURL)

			if err := llamawasm.FetchModelFile(mmprojPath, fetchURL, func(done, total int64) {
				if total > 0 {
					post(map[string]any{"kind": "progress", "file": "mmproj", "text": fmt.Sprintf("%d%%", done*100/total)})
				}
			}); err != nil {
				post(map[string]any{"kind": "error", "text": err.Error()})
				return
			}
		}

		if err := open(modelPath, pooling, mmprojURL != ""); err != nil {
			post(map[string]any{"kind": "error", "text": err.Error()})
		}
	}()

	return nil
}

// resolveCached asks the worker (embed-worker.js) to download url into the OPFS
// cache, or find it there, and returns a URL to fetch from. file is "mmproj" for the
// projector and "" for the model. It returns url itself if the worker has no cache.
func resolveCached(url, file string) string {
	resolve := js.Global().Get("yzmaCacheResolve")
	if resolve.Type() != js.TypeFunction {
		return url
	}

	result := make(chan string, 1)
	onResolve := js.FuncOf(func(this js.Value, args []js.Value) any {
		result <- args[0].String()
		return nil
	})
	onReject := js.FuncOf(func(this js.Value, args []js.Value) any {
		result <- url
		return nil
	})
	defer onResolve.Release()
	defer onReject.Release()

	resolve.Invoke(url, file).Call("then", onResolve, onReject)
	return <-result
}

// revokeBlobURL releases u if it is a blob: URL made by the worker.
func revokeBlobURL(u string) {
	if strings.HasPrefix(u, "blob:") {
		js.Global().Get("URL").Call("revokeObjectURL", u)
	}
}

// open loads the model at path with the given pooling type, and the projector too
// when withProjector is set, replacing any model that was loaded before.
func open(path, pooling string, withProjector bool) error {
	params := llamawasm.ModelDefaultParams()
	if llamawasm.GPUDevice() != "" {
		params.NGpuLayers = 999
	}

	var err error
	if model, err = llamawasm.ModelLoadFromFile(path, params); err != nil {
		return err
	}

	// Same context setup as acquireSlot in yzma.go.
	ctxParams := llamawasm.ContextDefaultParams()
	ctxParams.NCtx = 2048
	ctxParams.NBatch = 2048
	ctxParams.Embeddings = 1
	ctxParams.PoolingType = poolingType(pooling)

	if ctx, err = llamawasm.InitFromModel(model, ctxParams); err != nil {
		return err
	}

	vocab = llamawasm.ModelGetVocab(model)

	if withProjector {
		if !llamawasm.MtmdSupported() {
			return fmt.Errorf("this llama.cpp module has no multimodal support")
		}

		if mctx, err = llamawasm.MtmdInitFromFile(mmprojPath, model, llamawasm.MtmdContextParamsDefault()); err != nil {
			return err
		}

		if !llamawasm.MtmdSupportVision(mctx) {
			return fmt.Errorf("the projector does not accept images")
		}
	}

	post(map[string]any{
		"kind": "loaded",
		"text": fmt.Sprintf("%s, %d dimensions, images: %t", llamawasm.ModelDesc(model), llamawasm.ModelNEmbdOut(model), mctx != 0),
	})
	return nil
}

// poolingType maps a pooling name to its llama.cpp type; unknown names mean "mean".
func poolingType(s string) llamawasm.PoolingType {
	switch s {
	case "cls":
		return llamawasm.PoolingTypeCLS
	case "last":
		return llamawasm.PoolingTypeLast
	case "none":
		return llamawasm.PoolingTypeNone
	default:
		return llamawasm.PoolingTypeMean
	}
}

// embed implements yzmaEmbed(id, text): it posts {kind: "embedding", id, text,
// embedding: Float32Array}.
func embed(this js.Value, args []js.Value) any {
	if len(args) < 2 {
		post(map[string]any{"kind": "error", "text": "embed needs an id and text"})
		return nil
	}

	id, text := args[0].String(), args[1].String()

	go func() {
		vec, err := textEmbedding(text)
		if err != nil {
			post(map[string]any{"kind": "error", "text": err.Error()})
			return
		}

		post(map[string]any{"kind": "embedding", "id": id, "text": text, "embedding": float32Array(vec)})
	}()

	return nil
}

// resetMemory is the equivalent of yzmaSlot.reset in yzma.go. Non-causal
// embedding models may have no KV cache, in which case the module reports an
// invalid handle (-2); that is expected, so the result is ignored.
func resetMemory() {
	_ = llamawasm.MemoryClear(ctx, false)
}

// textEmbedding is the WASM equivalent of YzmaEmbedder.TextEmbeddings.
func textEmbedding(text string) ([]float32, error) {
	if model == 0 || ctx == 0 {
		return nil, fmt.Errorf("load a model first")
	}

	// Each request starts from an empty state (slot.reset in yzma.go).
	resetMemory()

	tokens := llamawasm.Tokenize(vocab, text, true, true)
	if len(tokens) == 0 {
		return nil, fmt.Errorf("the text has no tokens")
	}

	batch := llamawasm.BatchGetOne(tokens)

	ret, err := llamawasm.Decode(ctx, batch)
	if err != nil {
		return nil, fmt.Errorf("failed to decode tokens: %w", err)
	}
	if ret != 0 {
		return nil, fmt.Errorf("decode returned non-zero response, %d", ret)
	}

	return pooledEmbedding()
}

// pooledEmbedding reads the sequence embedding from the last decode and
// L2-normalizes it, as generateResponse does in yzma.go.
func pooledEmbedding() ([]float32, error) {
	// Use the output size: it can differ from n_embd (e.g. embeddinggemma-2).
	vec, err := llamawasm.GetEmbeddingsSeq(ctx, 0, llamawasm.ModelNEmbdOut(model))
	if err != nil {
		return nil, fmt.Errorf("failed to derive embeddings: %w", err)
	}

	// The returned slice may belong to the module's scratch memory; copy it.
	out := make([]float32, len(vec))
	copy(out, vec)

	var sum float64
	for _, v := range out {
		sum += float64(v) * float64(v)
	}
	if sum > 0 {
		norm := float32(1.0 / math.Sqrt(sum))
		for i := range out {
			out[i] *= norm
		}
	}

	return out, nil
}

// embedImage implements yzmaEmbedImage(id, width, height, rgba). It takes the pixels
// of an image as RGBA (e.g. from canvas getImageData().data) and posts the same
// message as embed.
func embedImage(this js.Value, args []js.Value) any {
	if len(args) < 4 {
		post(map[string]any{"kind": "error", "text": "embedImage needs an id, a size, and the pixels"})
		return nil
	}

	id := args[0].String()
	width, height := int32(args[1].Int()), int32(args[2].Int())

	rgba := make([]byte, args[3].Get("length").Int())
	js.CopyBytesToGo(rgba, args[3])

	go func() {
		vec, err := imageEmbedding(width, height, rgba)
		if err != nil {
			post(map[string]any{"kind": "error", "text": err.Error()})
			return
		}

		post(map[string]any{"kind": "embedding", "id": id, "text": "", "embedding": float32Array(vec)})
	}()

	return nil
}

// imageEmbedding is the WASM equivalent of YzmaEmbedder.ImageEmbeddings. The
// page decodes (and should downscale) the image, so there is no
// ResizeIfOversized or BitmapInitFromBuf step here.
func imageEmbedding(width, height int32, rgba []byte) ([]float32, error) {
	if model == 0 || ctx == 0 || mctx == 0 {
		return nil, fmt.Errorf("load a model with a projector first")
	}

	resetMemory()

	start := time.Now()
	stage := func(name string) {
		post(map[string]any{"kind": "status", "text": fmt.Sprintf("image: %s (%.1fs)", name, time.Since(start).Seconds())})
	}

	if !llamawasm.MtmdPartsSupported() {
		return nil, fmt.Errorf("this llama.cpp module is too old to read image chunks (needs ABI 13)")
	}

	bitmap, err := llamawasm.MtmdBitmapInit(width, height, dropAlpha(rgba))
	if err != nil {
		return nil, fmt.Errorf("failed to create bitmap: %w", err)
	}
	defer llamawasm.MtmdBitmapFree(bitmap)

	chunks, err := llamawasm.MtmdInputChunksInit()
	if err != nil {
		return nil, fmt.Errorf("failed to create chunks: %w", err)
	}
	defer llamawasm.MtmdInputChunksFree(chunks)

	marker := llamawasm.MtmdDefaultMarker()
	if err := llamawasm.MtmdTokenize(mctx, chunks, marker, true, true, []llamawasm.MtmdBitmap{bitmap}); err != nil {
		return nil, fmt.Errorf("multimodal tokenize failed: %w", err)
	}

	// Put the text tokens and projected image embeddings in one batch so the
	// pooled vector covers all of them (see ImageEmbeddings in yzma.go).
	batch, err := llamawasm.BatchExtInit(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to create batch: %w", err)
	}
	defer llamawasm.BatchExtFree(batch)

	nEmbdInp := int(llamawasm.ModelNEmbdInp(model))

	var pos llamawasm.Pos

	add := func(idx int32, err error) error {
		if err != nil {
			return err
		}
		if err := llamawasm.BatchExtSetPos(batch, idx, pos); err != nil {
			return err
		}
		pos++
		return llamawasm.BatchExtSetOutputEmbd(batch, idx, true)
	}

	for i := uint64(0); i < uint64(llamawasm.MtmdInputChunksSize(chunks)); i++ {
		chunk := llamawasm.MtmdInputChunksGet(chunks, i)

		switch llamawasm.MtmdInputChunkGetType(chunk) {
		case llamawasm.MtmdInputChunkTypeText:
			for _, token := range llamawasm.MtmdInputChunkGetTokensText(chunk) {
				if err := add(llamawasm.BatchExtAddToken(batch, 0, token)); err != nil {
					return nil, fmt.Errorf("failed to add text token: %w", err)
				}
			}

		default:
			stage("encoding with the vision model")
			if err := llamawasm.MtmdEncodeChunk(mctx, chunk); err != nil {
				return nil, fmt.Errorf("failed to encode image chunk: %w", err)
			}
			stage("vision encode done")

			nTokens := int(llamawasm.MtmdInputChunkGetNTokens(chunk))

			embd, err := llamawasm.MtmdGetOutputEmbd(mctx, int32(nEmbdInp*nTokens))
			if err != nil {
				return nil, fmt.Errorf("failed to read image embeddings: %w", err)
			}

			for row := 0; row < nTokens; row++ {
				if err := add(llamawasm.BatchExtAddEmbd(batch, 0, embd[row*nEmbdInp:(row+1)*nEmbdInp], nEmbdInp)); err != nil {
					return nil, fmt.Errorf("failed to add image embedding: %w", err)
				}
			}
		}
	}

	stage("running the text model over the image tokens")
	ret, err := llamawasm.Process(ctx, llamawasm.ProcessTypeDecode, batch)
	if err != nil || ret != 0 {
		return nil, fmt.Errorf("image evaluation failed with status %d: %v", ret, err)
	}
	stage("text model done")

	return pooledEmbedding()
}

// dropAlpha converts RGBA to the RGB that MtmdBitmapInit takes.
func dropAlpha(rgba []byte) []byte {
	rgb := make([]byte, 0, len(rgba)/4*3)
	for i := 0; i+3 < len(rgba); i += 4 {
		rgb = append(rgb, rgba[i], rgba[i+1], rgba[i+2])
	}
	return rgb
}

// float32Array copies vec into a JS Float32Array so it crosses postMessage cheaply.
func float32Array(vec []float32) js.Value {
	arr := js.Global().Get("Float32Array").New(len(vec))
	for i, v := range vec {
		arr.SetIndex(i, v)
	}
	return arr
}

// post sends a message to the host: the page when running in a worker, or
// yzmaOnMessage / the console under Node.
func post(message map[string]any) {
	if fn := js.Global().Get("postMessage"); fn.Type() == js.TypeFunction {
		fn.Invoke(message)
		return
	}
	if fn := js.Global().Get("yzmaOnMessage"); fn.Type() == js.TypeFunction {
		fn.Invoke(message)
		return
	}
	fmt.Printf("%v\n", message)
}
