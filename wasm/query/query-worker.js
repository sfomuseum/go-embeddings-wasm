/**
 * @file Worker for similarity search over a Parquet file of embeddings Records.
 * It hosts yzma-query.wasm (Go + go-parquet) and caches the normalized embedding matrix
 * of a (file, model) pair in OPFS so later sessions skip the download. It knows nothing
 * about the yzma embedding worker. Use it through query.js.
 */

importScripts("./wasm_exec.js");

/**
 * OPFS directory that holds cached embedding matrices.
 * @type {string}
 */
const CACHE_DIR = "query-cache";
/**
 * Bytes moved to or from OPFS at a time.
 * @type {number}
 */
const CHUNK = 16 << 20;

/**
 * Files chosen by the user, keyed by their "local://<name>" URL.
 * @type {Map<string, File>}
 */
const localFiles = new Map();
/**
 * The original fetch, used for everything that is not a local file.
 * @type {typeof fetch}
 */
const realFetch = self.fetch.bind(self);
/**
 * Replacement fetch that answers "local://" URLs from the File: HEAD gives the size and
 * Range requests give slices. The Go range reader only knows URLs.
 * @param {string|Request} input
 * @param {RequestInit} [init]
 * @returns {Promise<Response>}
 */
self.fetch = (input, init = {}) => {
  const file = typeof input === "string" ? localFiles.get(input) : undefined;
  if (!file) {
    return realFetch(input, init);
  }
  const headers = new Headers(init.headers || {});
  if ((init.method || "GET") === "HEAD") {
    return Promise.resolve(new Response(null, { status: 200, headers: { "content-length": String(file.size) } }));
  }
  const m = /^bytes=(\d+)-(\d+)?$/.exec(headers.get("Range") || "");
  if (!m) {
    return Promise.resolve(new Response(file, { status: 200 }));
  }
  const start = Number(m[1]);
  const end = Math.min(m[2] === undefined ? file.size - 1 : Number(m[2]), file.size - 1);
  return Promise.resolve(new Response(file.slice(start, end + 1), { status: 206 }));
};

/**
 * Posts a message to the page.
 * @param {Object} m
 */
const post = (m) => {
  self.postMessage(m);
};

/**
 * Posts a log line to the page.
 * @param {string} text
 */
const log = (text) => {
  post({ kind: "log", text });
};

// --- OPFS cache: <key>.vec (float32 matrix), <key>.idx (int64 row numbers), <key>.json (written last) ---

/**
 * Opens (creating if needed) the cache directory.
 * @returns {Promise<FileSystemDirectoryHandle>}
 */
const cacheDir = async () => {
  const root = await navigator.storage.getDirectory();
  return root.getDirectoryHandle(CACHE_DIR, { create: true });
};

/**
 * Names the cache entry for a file, model and file size.
 * @param {string} url
 * @param {string} model
 * @param {number} size
 * @returns {Promise<string>} A short SHA-256 based key.
 */
const cacheKey = async (url, model, size) => {
  const data = new TextEncoder().encode(`${url}\n${model}\n${size}`);
  const digest = await crypto.subtle.digest("SHA-256", data);
  const bytes = [...new Uint8Array(digest).slice(0, 12)];
  return "q" + bytes.map((b) => {
    return b.toString(16).padStart(2, "0");
  }).join("");
};

/**
 * Writes a cache file in chunks.
 * @param {FileSystemDirectoryHandle} dir
 * @param {string} name
 * @param {number} len - Total bytes to write.
 * @param {(off: number, n: number) => Uint8Array} chunkAt - Returns n bytes starting at off.
 * @returns {Promise<void>}
 */
const writeFile = async (dir, name, len, chunkAt) => {
  const acc = await (await dir.getFileHandle(name, { create: true })).createSyncAccessHandle();
  try {
    acc.truncate(0);
    for (let off = 0; off < len; off += CHUNK) {
      acc.write(chunkAt(off, Math.min(CHUNK, len - off)), { at: off });
    }
    acc.flush();
  } finally {
    acc.close();
  }
};

/**
 * Reads a cache file in chunks.
 * @param {FileSystemDirectoryHandle} dir
 * @param {string} name
 * @param {number} len - Expected size; a mismatch throws.
 * @param {(off: number, bytes: Uint8Array) => void} onChunk
 * @returns {Promise<void>}
 */
const readFile = async (dir, name, len, onChunk) => {
  const acc = await (await dir.getFileHandle(name)).createSyncAccessHandle();
  try {
    if (acc.getSize() !== len) {
      throw new Error(`${name}: unexpected size`);
    }
    const buf = new Uint8Array(Math.min(CHUNK, len));
    for (let off = 0; off < len; off += CHUNK) {
      const n = Math.min(CHUNK, len - off);
      acc.read(buf.subarray(0, n), { at: off });
      onChunk(off, buf.subarray(0, n));
    }
  } finally {
    acc.close();
  }
};

/**
 * Saves the matrix held by the wasm module: .vec (float32), .idx (int64 row numbers),
 * then .json, written last so a partial write is never used.
 * @param {string} key
 * @param {{url: string, model: string}} origin - What the index was built from, so it can be listed.
 * @returns {Promise<void>}
 */
const saveCache = async (key, origin) => {
  const dir = await cacheDir();
  const info = { ...JSON.parse(queryMatrixInfo()), ...origin };
  const sizes = { vec: info.count * info.dim * 4, idx: info.count * 8 };
  for (const name of ["vec", "idx"]) {
    await writeFile(dir, `${key}.${name}`, sizes[name], (off, n) => {
      return queryExport(name, off, n);
    });
  }
  const meta = new TextEncoder().encode(JSON.stringify(info));
  await writeFile(dir, `${key}.json`, meta.length, () => {
    return meta;
  });
};

/**
 * Lists the cached indexes with a recorded origin: URLs, and "local://" chosen files (which
 * are reopened from a copy in the cache). Ones cached before the origin was recorded are skipped.
 * @returns {Promise<Array<{url: string, model: string, count: number}>>}
 */
const listCache = async () => {
  const items = [];
  try {
    const dir = await cacheDir();
    for await (const [name, handle] of dir.entries()) {
      if (!name.endsWith(".json")) {
        continue;
      }
      try {
        const meta = JSON.parse(await (await handle.getFile()).text());
        if (meta.url) {
          items.push({ url: meta.url, model: meta.model || "", count: meta.count });
        }
      } catch {
        // Unreadable entry: not listed.
      }
    }
  } catch {
    return [];
  }
  items.sort((a, b) => {
    return a.url.localeCompare(b.url);
  });
  return items;
};

/**
 * Names the OPFS copy of a chosen local file.
 * @param {string} url - The "local://" URL.
 * @returns {Promise<string>}
 */
const localCopyName = async (url) => {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(url));
  const bytes = [...new Uint8Array(digest).slice(0, 12)];
  return "l" + bytes.map((b) => {
    return b.toString(16).padStart(2, "0");
  }).join("") + ".parquet";
};

/**
 * Copies a chosen file into the cache, so its index can be used again without choosing it.
 * @param {string} url
 * @param {File} file
 * @returns {Promise<void>}
 */
const saveLocalCopy = async (url, file) => {
  const dir = await cacheDir();
  const name = await localCopyName(url);
  const handle = await dir.getFileHandle(name, { create: true });
  try {
    // A writable stream commits only when closed, so a failed copy leaves nothing behind.
    await file.stream().pipeTo(await handle.createWritable());
    log(`cached a copy of ${file.name} (${(file.size / 1e6).toFixed(1)} MB)`);
  } catch (err) {
    await dir.removeEntry(name).catch(() => {});
    throw err;
  }
};

/**
 * Reports whether a cached copy of a "local://" file exists.
 * @param {string} url
 * @returns {Promise<boolean>}
 */
const hasLocalCopy = async (url) => {
  try {
    const file = await (await (await cacheDir()).getFileHandle(await localCopyName(url))).getFile();
    return file.size > 0;
  } catch {
    return false;
  }
};

/**
 * Registers the cached copy of a "local://" URL, if there is one.
 * @param {string} url
 * @returns {Promise<boolean>}
 */
const restoreLocalCopy = async (url) => {
  try {
    const dir = await cacheDir();
    const file = await (await dir.getFileHandle(await localCopyName(url))).getFile();
    if (file.size === 0) {
      log(`the cached copy of ${url} is empty`);
      return false;
    }
    localFiles.set(url, file);
    return true;
  } catch (err) {
    log(`no cached copy of ${url}: ${err}`);
    return false;
  }
};

/**
 * Loads a cached matrix into the wasm module.
 * @param {string} key
 * @returns {Promise<?{count: number, dim: number}>} The matrix info, or null if not cached.
 */
const loadCache = async (key) => {
  let dir, info;
  try {
    dir = await cacheDir();
    const f = await (await dir.getFileHandle(`${key}.json`)).getFile();
    info = JSON.parse(await f.text());
    queryImportBegin(info.count, info.dim);
    for (const [name, len] of [["vec", info.count * info.dim * 4], ["idx", info.count * 8]]) {
      await readFile(dir, `${key}.${name}`, len, (off, bytes) => {
        return queryImportChunk(name, off, bytes);
      });
    }
    return info;
  } catch {
    return null;
  }
};

/**
 * Deletes the whole cache directory.
 * @returns {Promise<void>}
 */
const purge = async () => {
  try {
    await (await navigator.storage.getDirectory()).removeEntry(CACHE_DIR, { recursive: true });
  } catch (e) {
    if (e.name !== "NotFoundError") {
      throw e;
    }
  }
};

// --- wasm module ---

/**
 * The open file, if any.
 * @type {?{url: string, size: number}}
 */
let opened = null; // { url, size }

/**
 * Resolves once yzma-query.wasm is running.
 * @type {Promise<void>}
 */
const ready = (async () => {
  const go = new Go();
  const result = await WebAssembly.instantiateStreaming(fetch("./yzma-query.wasm"), go.importObject);
  go.run(result.instance);
})();
ready.catch((err) => {
  post({ kind: "error", text: `loading yzma-query.wasm: ${err && err.message ? err.message : err}` });
});

/**
 * Request handlers by message kind. Go posts its own "ready" and "progress" messages;
 * those are forwarded as they are.
 * @type {Object<string, function(Object): Promise<Object>>}
 */
const handlers = {
  async open({ url, file }) {
    if (file) {
      url = `local://${file.name}?${file.lastModified}`;
      localFiles.set(url, file);
    } else if (url.startsWith("local://") && !localFiles.has(url) && !(await restoreLocalCopy(url))) {
      throw new Error("no cached copy of this file; choose it again");
    }
    const info = JSON.parse(await queryOpen(url));
    opened = { url, size: info.size };
    return info;
  },
  async build({ model }) {
    if (!opened) {
      throw new Error("open a file first");
    }
    const key = await cacheKey(opened.url, model, opened.size);
    const cached = await loadCache(key);
    if (cached) {
      log(`loaded ${cached.count} embeddings (${cached.dim} dims) from cache`);
      // Indexes cached before local copies were kept get their copy when the file is chosen again.
      const file = opened.url.startsWith("local://") ? localFiles.get(opened.url) : null;
      if (file && !(await hasLocalCopy(opened.url))) {
        try {
          await saveLocalCopy(opened.url, file);
        } catch (err) {
          log(`could not cache a copy of the file: ${err}`);
        }
      }
      return { ...cached, cached: true };
    }
    const info = JSON.parse(await queryBuild(model));
    try {
      await saveCache(key, { url: opened.url, model: info.model || model });
      const file = localFiles.get(opened.url);
      if (file) {
        await saveLocalCopy(opened.url, file);
      } else if (opened.url.startsWith("local://")) {
        log("the file was not available to copy into the cache");
      }
    } catch (err) {
      log(`could not cache the index: ${err}`);
    }
    return { ...info, cached: false };
  },
  async search({ embedding, k }) {
    return JSON.parse(await querySearch(new Float32Array(embedding), k));
  },
  async purge() {
    await purge();
    return {};
  },
  async list() {
    return listCache();
  },
};

/**
 * Runs the handler for a request and posts a "result" or "error-result" with its id.
 * @param {MessageEvent} event
 */
self.onmessage = async ({ data }) => {
  const { id, kind } = data;
  try {
    await ready;
    if (!handlers[kind]) {
      throw new Error(`unknown message ${kind}`);
    }
    post({ kind: "result", id, result: await handlers[kind](data) });
  } catch (err) {
    let text = String(err);
    if (err && err.message) {
      text = String(err.message);
    }
    post({ kind: "error-result", id, text });
  }
};
