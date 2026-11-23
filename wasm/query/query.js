/**
 * @file Main-thread client for the Parquet similarity query worker.
 */

/**
 * Client for the Parquet similarity query worker (query-worker.js). Independent of the yzma
 * embedding code; it only needs an embedding vector and the name of the model that made it.
 *
 * @example
 * const q = new ParquetQuery({ onProgress: (pct) => {}, onLog: (msg) => {} });
 * await q.open("https://example.org/embeddings.parquet"); // reads the footer
 * await q.build("embeddinggemma-2-Q8_0.gguf"); // scans (or loads from cache) one model's rows
 * const records = await q.search(embedding, 20);
 *
 * similarity in each record is the cosine distance (0 = identical).
 */
class ParquetQuery {
  /**
   * Starts the query worker.
   * @param {Object} [options]
   * @param {string} [options.workerURL="./query-worker.js"]
   * @param {(pct: number) => void} [options.onProgress] - Percent done while scanning.
   * @param {(msg: string) => void} [options.onLog] - Log lines from the worker.
   * @param {() => void} [options.onReady] - Called once yzma-query.wasm is running.
   * @param {(msg: string) => void} [options.onError] - Called if the module fails to load or run.
   */
  constructor({ workerURL = "./query-worker.js", onProgress = () => {}, onLog = () => {}, onReady = () => {}, onError = () => {} } = {}) {
    this.onProgress = onProgress;
    this.onLog = onLog;
    this.onReady = onReady;
    this.onError = onError;
    this.pending = new Map();
    this.n = 0;
    this.worker = new Worker(workerURL);
    this.worker.onmessage = ({ data }) => {
      if (data.kind === "result" || data.kind === "error-result") {
        const p = this.pending.get(data.id);
        if (!p) {
          return;
        }
        this.pending.delete(data.id);
        if (data.kind === "result") {
          p.resolve(data.result);
        } else {
          p.reject(new Error(data.text));
        }
      } else if (data.kind === "progress") {
        this.onProgress(parseInt(data.text, 10) || 0);
      } else if (data.kind === "ready") {
        this.onReady();
      } else {
        if (data.kind === "error") {
          this.onError(data.text ?? "");
        }
        this.onLog(`${data.kind}: ${data.text ?? ""}`);
      }
    };
    this.worker.onerror = (e) => {
      this.onLog(`query worker error: ${e.message}`);
      this.onError(e.message || "query worker failed to start");
    };
  }

  /**
   * Sends a request to the worker.
   * @param {string} kind - Handler name: open, build, search or purge.
   * @param {Object} [args] - Extra message fields.
   * @returns {Promise<any>} Resolves with the handler result; rejects with its error.
   */
  call(kind, args = {}) {
    const id = ++this.n;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ kind, id, ...args });
    });
  }

  /**
   * Opens a Parquet file and reads its footer. The server must allow CORS and Range
   * requests (and HEAD, for the file size).
   * @param {string|File} source - A URL, or a File chosen by the user.
   * @returns {Promise<{rows: number, row_groups: number, size: number}>}
   */
  open(source) {
    if (typeof source === "string") {
      return this.call("open", { url: source });
    }
    return this.call("open", { file: source });
  }

  /**
   * Scans (or loads from the cache) the embeddings of one model.
   * @param {string} model - Model name as stored in the file; "" selects the file's only model.
   * @returns {Promise<{count: number, dim: number, cached: boolean, skipped?: number, model?: string}>}
   */
  build(model) {
    return this.call("build", { model });
  }

  /**
   * Finds the records nearest to an embedding.
   * @param {number[]|Float32Array} embedding
   * @param {number} [k=20] - Number of records to return.
   * @returns {Promise<Array<{provider: string, depiction_id: string, subject_id: string, similarity: number, attributes: Object}>>}
   */
  search(embedding, k = 20) {
    return this.call("search", { embedding: Float32Array.from(embedding), k });
  }

  /**
   * Lists the cached indexes that were built from a URL.
   * @returns {Promise<Array<{url: string, model: string, count: number}>>}
   */
  list() {
    return this.call("list");
  }

  /**
   * Deletes the cached indexes.
   * @returns {Promise<Object>}
   */
  purge() {
    return this.call("purge");
  }
}
