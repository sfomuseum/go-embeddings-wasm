/**
 * @file Web Worker that runs llama.cpp and yzma-embeddings.wasm to make text and image embeddings,
 * with an OPFS cache for models. The message protocol is described below.
 */

// embed-worker.js runs llama.cpp and yzma-embeddings.wasm in a Web Worker.
// Based on yzma's wasm/worker.js; see that file for the details of each step.
//
// Messages from the page:
//   { kind: "load", url: "<model URL>", pooling: "mean", mmproj: "<optional projector URL>",
//     file: <optional File for url>, mmprojFile: <optional File for mmproj> }
//   { kind: "embed", id: "<id>", text: "<text>" }
//   { kind: "embed-image", id: "<id>", width, height, rgba: <ArrayBuffer of RGBA pixels> }
//   { kind: "purge", urls: ["<model URL>", "<projector URL>"] } removes them from the OPFS cache
//   { kind: "list-cache" } asks for the cached models
//
// Messages to the page: { kind, text, id?, embedding?, items? } where kind is one of
// ready, status, progress, loaded, embedding, purged, cached or error. A "cached" message
// carries items: [{ url, role, size }], role being "model" or "mmproj".

/**
 * True inside an Emscripten pthread worker, which only loads llama.cpp.
 * @type {boolean}
 */
const isThread = globalThis.name === "em-pthread";

self.yzmaBase = ".";

const workerQuery = new URLSearchParams((self.location.search || "").slice(1));
if (workerQuery.get("mode")) {
  self.yzmaMode = workerQuery.get("mode");
}
if (isThread) {
  self.yzmaMode = "cpu";
}

importScripts("./yzma-loader.js");

if (!isThread) {
  run();
}

// Models are cached in the Origin Private File System, keyed by a hash of their URL.
// A download only replaces the cache entry once it completes (createWritable commits on close).
/**
 * User-chosen files, keyed by their "local://<name>" URL; never cached.
 * @type {Map<string, File>}
 */
const localFiles = new Map(); // "local://<name>" -> File

/**
 * OPFS directory that holds the cached GGUF files.
 * @type {string}
 */
const cacheDirName = "yzma-models";

/**
 * Finds the OPFS directory and file name that cache a URL.
 * @param {string} url - Model or projector URL; its SHA-256 names the file.
 * @returns {Promise<{dir: FileSystemDirectoryHandle, name: string}>}
 */
async function cacheEntry(url) {
  const root = await navigator.storage.getDirectory();
  const dir = await root.getDirectoryHandle(cacheDirName, { create: true });
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(url));
  const hex = Array.from(new Uint8Array(digest), (b) => {
    return b.toString(16).padStart(2, "0");
  }).join("");
  return { dir, name: hex + ".gguf" };
}

/**
 * Records which URL (and role) a cached file came from; the cache file name is only a hash.
 * The metadata is a small JSON file beside it, named like it but ending in .json.
 * @param {FileSystemDirectoryHandle} dir
 * @param {string} name - Cache file name, as returned by cacheEntry.
 * @param {string} url
 * @param {string} role - "model" or "mmproj".
 * @returns {Promise<void>}
 */
async function cacheWriteMeta(dir, name, url, role) {
  const writable = await (await dir.getFileHandle(name.replace(/\.gguf$/, ".json"), { create: true })).createWritable();
  await writable.write(JSON.stringify({ url, role }));
  await writable.close();
}

/**
 * Lists the cached files whose origin is known.
 * @returns {Promise<Array<{url: string, role: string, size: number}>>}
 */
async function cacheList() {
  const root = await navigator.storage.getDirectory();
  let dir;
  try {
    dir = await root.getDirectoryHandle(cacheDirName);
  } catch (err) {
    if (err.name === "NotFoundError") {
      return [];
    }
    throw err;
  }
  const items = [];
  for await (const [name, handle] of dir.entries()) {
    if (!name.endsWith(".json")) {
      continue;
    }
    try {
      const meta = JSON.parse(await (await handle.getFile()).text());
      const gguf = await (await dir.getFileHandle(name.replace(/\.json$/, ".gguf"))).getFile();
      items.push({ url: meta.url, role: meta.role === "mmproj" ? "mmproj" : "model", size: gguf.size });
    } catch (err) {
      // Metadata without its model, or unreadable: not listed.
    }
  }
  items.sort((a, b) => {
    return a.url.localeCompare(b.url);
  });
  return items;
}

/**
 * Returns a URL for the Go program to fetch: a blob URL of the cached file, or the
 * original URL if the cache is unavailable. Downloads and caches the file when needed.
 * @param {string} url - Model or projector URL (or a "local://" key).
 * @param {string} file - "mmproj" for the projector, "" for the model.
 * @returns {Promise<string>}
 */
async function cacheResolve(url, file) {
  const label = file || "model";
  // Local files chosen by the user are used as they are and never cached.
  if (localFiles.has(url)) {
    return URL.createObjectURL(localFiles.get(url));
  }

  /**
   * Posts a status or progress message to the page.
   * @param {string} kind
   * @param {string} text
   */
  const note = (kind, text) => {
    if (file) {
      self.postMessage({ kind, text, file });
    } else {
      self.postMessage({ kind, text });
    }
  };

  try {
    const { dir, name } = await cacheEntry(url);

    try {
      const cached = await (await dir.getFileHandle(name)).getFile();
      note("status", `using cached ${label} (${cached.size} bytes)`);
      // Files cached before metadata was kept become selectable once they are used again.
      await cacheWriteMeta(dir, name, url, label).catch(() => {});
      return URL.createObjectURL(cached);
    } catch (err) {
      if (err.name !== "NotFoundError") {
        throw err;
      }
    }

    note("status", `downloading the ${label}`);
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`cannot fetch ${url}: status ${response.status}`);
    }

    const total = Number(response.headers.get("content-length")) || 0;
    const encoded = !!response.headers.get("content-encoding");
    let done = 0;

    const progress = new TransformStream({
      transform(chunk, controller) {
        done += chunk.length;
        if (total) {
          note("progress", Math.floor((done * 100) / total) + "%");
        }
        controller.enqueue(chunk);
      },
    });

    const writable = await (await dir.getFileHandle(name, { create: true })).createWritable();
    await response.body.pipeThrough(progress).pipeTo(writable);

    if (total && !encoded && done !== total) {
      await dir.removeEntry(name);
      throw new Error(`${url} gave ${done} bytes of ${total}`);
    }

    await cacheWriteMeta(dir, name, url, label);
    return URL.createObjectURL(await (await dir.getFileHandle(name)).getFile());
  } catch (err) {
    note("status", `cache unavailable for the ${label} (${err}); downloading without it`);
    return url;
  }
}

/**
 * Removes cached files from OPFS.
 * @param {string[]} urls - URLs whose cache entries should be deleted.
 * @returns {Promise<string[]>} The URLs that were removed.
 */
async function cachePurge(urls) {
  const removed = [];
  for (const url of urls.filter(Boolean)) {
    const { dir, name } = await cacheEntry(url);
    try {
      await dir.removeEntry(name);
      removed.push(url);
    } catch (err) {
      if (err.name !== "NotFoundError") {
        throw err;
      }
    }
    try {
      await dir.removeEntry(name.replace(/\.gguf$/, ".json"));
    } catch (err) {
      if (err.name !== "NotFoundError") {
        throw err;
      }
    }
  }
  return removed;
}

/**
 * Starts the worker: loads wasm_exec.js and yzma-embeddings.wasm, then handles page messages.
 * @returns {void}
 */
function run() {
  self.yzmaCacheResolve = cacheResolve;

  self.onerror = (event) => {
    self.postMessage({ kind: "error", text: String((event && event.message) || event) });
  };

  importScripts("./wasm_exec.js");

  let programIsReady;
  const programReady = new Promise((resolve) => {
    programIsReady = resolve;
  });

  const sendToPage = self.postMessage.bind(self);
  self.postMessage = (message) => {
    if (message && (message.kind === "ready" || message.kind === "error")) {
      programIsReady();
    }
    sendToPage(message, message && message.embedding ? [message.embedding.buffer] : []);
  };

  /**
   * Resolves once the Go program has loaded and posted "ready".
   * @type {Promise<void>}
   */
  const started = (async () => {
    await self.yzmaReady;

    const go = new Go();
    const result = await WebAssembly.instantiateStreaming(fetch("./yzma-embeddings.wasm"), go.importObject);
    go.run(result.instance);
    await programReady;
  })();
  started.catch((err) => {
    self.postMessage({ kind: "error", text: `loading yzma-embeddings.wasm: ${err && err.message ? err.message : err}` });
  });

  self.onmessage = async (event) => {
    const message = event.data || {};
    try {
      await started;
      switch (message.kind) {
        case "load":
          for (const [key, f] of [[message.url, message.file], [message.mmproj, message.mmprojFile]]) {
            if (f) {
              localFiles.set(key, f);
            }
          }
          self.yzmaLoadModel(message.url, message.pooling || "mean", message.mmproj || "");
          break;
        case "purge": {
          const removed = await cachePurge(message.urls || []);
          let text = "nothing to remove from cache";
          if (removed.length) {
            text = `removed from cache: ${removed.join(", ")}`;
          }
          self.postMessage({ kind: "purged", text });
          break;
        }
        case "list-cache":
          self.postMessage({ kind: "cached", items: await cacheList() });
          break;
        case "embed":
          self.yzmaEmbed(String(message.id), message.text);
          break;
        case "embed-image":
          self.yzmaEmbedImage(String(message.id), message.width, message.height, new Uint8Array(message.rgba));
          break;
        default:
          self.postMessage({ kind: "error", text: "unknown message: " + message.kind });
      }
    } catch (err) {
      self.postMessage({ kind: "error", text: String(err) });
    }
  };
}
