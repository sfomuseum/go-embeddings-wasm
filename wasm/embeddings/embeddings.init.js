/**
 * @file Page logic: loads models in the embeddings worker, makes text and image embeddings,
 * and searches a Parquet file of records through ParquetQuery.
 */

/**
 * Images larger than this on either side are scaled down before embedding.
 * @type {number}
 */
const MAX_IMAGE_SIDE = 512;

window.addEventListener("load", function load(event){
    
    /**
     * Appends a line to the on-page log.
     * @param {string} s
     */
    const log = (s) => {
	document.getElementById("log").textContent += s + "\n";
    };
    
    const progress = document.getElementById("progress");
    
    const mmprojProgress = document.getElementById("mmproj-progress");

    /**
     * Number of WebAssembly modules (embeddings and query) that must be running before the page is usable.
     * @type {number}
     */
    const MODULE_COUNT = 2;

    /**
     * Names of the modules that have finished loading.
     * @type {Set<string>}
     */
    const loadedModules = new Set();

    /**
     * Records that a module is running; once all are, hides the loading row and shows the page.
     * @param {string} name
     * @returns {void}
     */
    const moduleLoaded = (name) => {
	loadedModules.add(name);
	document.getElementById("loading-progress").value = loadedModules.size;
	document.getElementById("loading-status").textContent = `${loadedModules.size} of ${MODULE_COUNT}`;
	if (loadedModules.size === MODULE_COUNT) {
	    document.getElementById("loading").hidden = true;
	    document.getElementById("layout").hidden = false;
	}
    };

    /**
     * Shows why a module could not be loaded. The page stays hidden.
     * @param {string} name
     * @param {string} text
     * @returns {void}
     */
    const moduleFailed = (name, text) => {
	if (document.getElementById("layout").hidden === false) {
	    return;
	}
	const el = document.getElementById("loading-error");
	el.hidden = false;
	el.textContent += `${name}: ${text || "failed to load"}\n`;
    };
    
    /**
     * The yzma embeddings worker.
     * @type {Worker}
     */
    const worker = new Worker("./embed-worker.js");
    
    /**
     * Counter used to give each embedding request an id.
     * @type {number}
     */
    let n = 0;
    
    /**
     * Embeddings in flight, by id, mapped to their kind ("text" or "image").
     * @type {Map<string, string>}
     */
    const pending = new Map();
    /**
     * Ids of embeddings that start a search when they arrive.
     * @type {Set<string>}
     */
    const searchAfter = new Set(); // ids of embeddings that start a search when they arrive
    const busy = document.getElementById("busy");
    
    /**
     * Whether an embedding model, and an mmproj for image input, have been loaded.
     * @type {{model: boolean, mmproj: boolean}}
     */
    const loadedState = { model: false, mmproj: false };
    
    /**
     * Shows the busy indicator and enables or disables the embed buttons.
     * @returns {void}
     */
    const updateBusy = () => {
	busy.hidden = pending.size === 0;
	document.getElementById("busy-meter").active = pending.size > 0;
	const labels = [...pending.values()].map((p) => {
	    return `generating ${p} embeddings...`;
	});
	document.getElementById("busy-text").textContent = labels.join(" ");
	for (const id of ["embed", "embed-image"]) {
	    const button = document.getElementById(id);
	    button.disabled = pending.size > 0 || button.dataset.off === "1" || button.dataset.blocked === "1";
	}
	let why = "";
	if (!loadedState.model || !loadedState.mmproj) {
	    why = "An embedding (mmproj) file needs to be loaded before you can embed images";
	} else if (document.getElementById("embed-image").dataset.blocked === "1") {
	    why = "Clear the text input to embed an image.";
	}
	document.getElementById("embed-image-hint").textContent = why;
    };
    
    /**
     * Allows only one input source: text blocks "Embed image" and a chosen image blocks "Embed Text".
     * @returns {void}
     */
    const updateSources = () => {
	const hasText = document.getElementById("text").value !== "";
	const hasImage = document.getElementById("image").files.length > 0;
	document.getElementById("embed").dataset.blocked = hasImage ? "1" : "0";
	document.getElementById("embed-image").dataset.blocked = hasText ? "1" : "0";
	updateBusy();
    };
    
    /**
     * Latest embeddings per kind.
     * @type {{text: ?number[], image: ?number[]}}
     */
    const generated = {
	text: null, image: null
    };
    
    /**
     * Image embeddings take precedence over text when both exist.
     * @returns {"image"|"text"|""} The kind to search with, or "" if none.
     */
    const activeKind = () => {
	if (generated.image) {
	    return "image";
	}
	if (generated.text) {
	    return "text";
	}
	return "";
    };
    
    /**
     * Displays the active embeddings as JSON and refreshes the search button.
     * @param {boolean} [blank=false] - Clear the display instead.
     * @returns {void}
     */
    const showEmbeddings = (blank = false) => {
	const kind = blank ? "" : activeKind();
	let embeddingsText = "";
	let kindText = "";
	if (kind) {
	    embeddingsText = JSON.stringify(generated[kind], null, 2);
	    kindText = `(${kind} embeddings, ${generated[kind].length} dims)`;
	}
	document.getElementById("embeddings").textContent = embeddingsText;
	document.getElementById("embeddings-kind").textContent = kindText;
	updateLocal();
    };
    
    /**
     * Records a new in-flight embedding and clears the display.
     * @param {string} id
     * @param {string} label - "text" or "image".
     * @returns {void}
     */
    const start = (id, label) => {
	showEmbeddings(true); pending.set(id, label); log(`embedding ${id}: generating ${label} embedding`); updateBusy();
    };
	 
    /**
     * Fills the cached-model and cached-mmproj selects from the worker's OPFS cache. A select
     * is hidden when the cache holds nothing for it.
     * @param {Array<{url: string, role: string, size: number}>} items
     * @returns {void}
     */
    const showCached = (items) => {
	for (const select of document.querySelectorAll("select.cached")) {
	    const role = select.id === "mmproj-cached" ? "mmproj" : "model";
	    const mine = items.filter((item) => {
		return item.role === role;
	    });
	    const current = document.getElementById(select.dataset.target).value;
	    select.replaceChildren();
	    const none = el("option", `Cached ${role === "mmproj" ? "mmproj files" : "models"}...`);
	    none.value = "";
	    select.append(none);
	    for (const item of mine) {
		const o = el("option", `${shortModelName(item.url)} (${Math.round(item.size / 1048576)} MB)`);
		o.value = item.url;
		o.title = item.url;
		select.append(o);
	    }
	    select.value = mine.some((item) => {
		return item.url === current;
	    }) ? current : "";
	    select.hidden = mine.length === 0;
	}
    };

    worker.onerror = (e) => {
	moduleFailed("embeddings", e.message);
    };

    worker.onmessage = ({ data }) => {
	
	if (data.kind === "ready") {
	    moduleLoaded("embeddings");
	    worker.postMessage({ kind: "list-cache" });
	}
	if (data.kind === "error" && !loadedModules.has("embeddings")) {
	    moduleFailed("embeddings", data.text);
	}

	if (data.kind === "cached") {
	    showCached(data.items);
	} else if (data.kind === "embedding") {
	    const v = data.embedding;
	    const kind = pending.get(data.id) === "image" ? "image" : "text";
	    pending.delete(data.id);
	    updateBusy();
	    log(`embedding ${data.id}: ${v.length} dims`);
	    generated[kind] = Array.from(v);
	    showEmbeddings();
	    if (searchAfter.delete(data.id)) {
		searchLocal();
	    }
	} else if (data.kind === "progress") {
	    (data.file === "mmproj" ? mmprojProgress : progress).value = parseInt(data.text, 10) || 0;
	} else {
	    log(`${data.kind}: ${data.text}`);
	    if (data.kind === "loaded" || data.kind === "purged") {
		worker.postMessage({ kind: "list-cache" });
	    }
	    if (data.kind === "loaded") {
		document.getElementById("embed").dataset.off = "0";
		loadedState.model = true;
		loadedState.mmproj = !!(document.getElementById("mmproj").value || document.getElementById("mmproj-file").files[0]);
		document.getElementById("embed-image").dataset.off = (document.getElementById("mmproj").value || document.getElementById("mmproj-file").files[0]) ? "0" : "1";
		updateBusy();
		updateLocal();
	    } else if (data.kind === "error") {
		pending.clear();
		if (searchAfter.size > 0) {
		    setSearching(false);
		}
		searchAfter.clear();
		showEmbeddings();
		updateBusy();
	    }
	}
    };
	 
    /**
     * Name of the loaded model as it appears in the Parquet file.
     * @type {string}
     */
    let modelName = "";
    
    /**
     * The model name is the basename of the model URL, e.g. embeddinggemma-2-Q8_0.gguf.
     * @param {string} u - Model URL.
     * @returns {string}
     */
    const shortModelName = (u) => {
	try {
	    return new URL(u).pathname.split("/").filter(Boolean).pop() || u;
	} catch {
	    return u.split("/").filter(Boolean).pop() || u;
	}
    };
    
    /**
     * Only http(s) URLs are used for links and images.
     * @param {string} u
     * @returns {string} The normalized URL, or "" if unsafe or invalid.
     */
    const safeURL = (u) => {
	
	try {
	    const url = new URL(u);
	    return url.protocol === "http:" || url.protocol === "https:" ? url.href : "";
	} catch {
	    return "";
	}
    };
    
    /**
     * Creates an element, optionally with text content.
     * @param {string} tag
     * @param {string} [text]
     * @returns {HTMLElement}
     */
    const el = (tag, text) => {
	const e = document.createElement(tag);
	if (text !== undefined) {
	    e.textContent = text;
	}
	return e;
    };
    
    /**
     * Creates a link that opens in a new tab.
     * @param {string} href - Checked with safeURL.
     * @param {string} text
     * @returns {HTMLAnchorElement}
     */
    const link = (href, text) => {
	
	const h = safeURL(href);
	
	if (!h) {
	    return el("span", text);
	}
	
	const a = el("a", text);
	a.href = h;
	a.target = "_blank";
	a.rel = "noopener noreferrer";
	return a;
    };
	 
    /**
     * Empties the results panel; it stays hidden unless a message is given.
     * @param {string} [message] - Text to show, e.g. "Searching...".
     * @returns {HTMLElement} The results element.
     */
    const showImageModal = (src, alt) => {
	const dialog = document.getElementById("image-modal");
	const big = dialog.querySelector("img");
	big.src = src;
	big.alt = alt || "";
	dialog.showModal();
    };

    document.getElementById("image-modal").addEventListener("click", (e) => {
	// Clicking the backdrop or the close button closes; clicking the image does not.
	if (e.target.tagName !== "IMG") {
	    e.currentTarget.close();
	}
    });

    document.getElementById("image-modal").addEventListener("keydown", (e) => {
	if ((e.key === "Enter" || e.key === " ") && e.target.classList.contains("modal-close")) {
	    e.preventDefault();
	    e.currentTarget.close();
	}
    });

    const resetResults = (message) => {
	
	const root = document.getElementById("results");
	root.replaceChildren();
	
	// Shown only while there is a message or results to display.
	root.hidden = !message;
	
	if (message) {
	    root.append(el("p", message));
	}
	
	return root;
    };
    
	 /**
	  * The result a "Find similar" search started from; left out of the next results.
	  * @type {?{depictionId: string, subjectId: string}}
	  */
	 let excluded = null;
	 /**
	  * Renders SimilarRecords; attributes follow the OEmbeddings model but may hold more.
	  * The panel is hidden when there are no records.
	  * @param {Array<Object>} records
	  * @returns {void}
	  */
	 const showResults = (records) => {
	     if (excluded && Array.isArray(records)) {
		 records = records.filter((r) => {
		     const same = r.depiction_id === excluded.depictionId && r.subject_id === excluded.subjectId;
		     return !same;
		 });
	     }
	     const root = resetResults();
	     if (!Array.isArray(records) || records.length === 0) {
		 root.hidden = true;
		 log("similar: no matching records");
		 return;
	     }
	     root.hidden = false;
	     for (const r of records) {
		 const a = r.attributes || {};
		 const item = el("div");
		 item.className = "result";
		 const thumb = a.type === "image" || a.type === undefined ? safeURL(a.preview) : "";
		 let img = null;
		 if (thumb) {
		     img = el("img");
		     // COEP require-corp (needed for threads) blocks no-CORS cross-origin images.
			 img.crossOrigin = "anonymous";
		     img.src = thumb;
		     img.alt = a.subject_title || "";
		     img.loading = "lazy";
		     const href = link(thumb, "");
		     href.replaceChildren(img);
		     href.addEventListener("click", (e) => {
			 e.preventDefault();
			 showImageModal(img.currentSrc || img.src, img.alt);
		     });
		     item.append(href);
		 } else if (a.preview) {
		     item.append(el("div", String(a.preview)));
		 }
		 const dl = el("dl");
		 /**
		  * Appends a term and its description to the result's list.
		  * @param {string} k
		  * @param {string|Node} v
		  */
		 const row = (k, v) => {
		     dl.append(el("dt", k));
		     const dd = el("dd");
		     dd.append(v);
		     dl.append(dd);
		 };
		 row("Title", a.subject_url ? link(a.subject_url, a.subject_title || a.subject_url) : (a.subject_title || ""));
		 if (a.subject_creditline) {
		     row("Credit", a.subject_creditline);
		 }
		 if (a.provider_name || r.provider) {
		     const providerName = a.provider_name || r.provider;
		     if (a.provider_url) {
			 row("Provider", link(a.provider_url, providerName));
		     } else {
			 row("Provider", providerName);
		     }
		 }
		 if (a.depiction_url) {
		     row("Depiction", link(a.depiction_url, r.depiction_id || a.depiction_url));
		 }
		 row("Distance", String(r.similarity));
		 row("IDs", `${r.depiction_id} / ${r.subject_id}`);
		 if (thumb) {
		     const btn = el("button", "Find similar");
		     btn.setAttribute("class", "btn btn-primary");
		     btn.onclick = () => {
			 findSimilarToImage(img, { depictionId: r.depiction_id, subjectId: r.subject_id });
		     };
		     dl.append(btn);
		 }
		 item.append(dl);
		 root.append(item);
	     }
	 };
	 
	 /**
	  * Client for the Parquet query worker; the Parquet-specific work lives in query.js.
	  * @type {ParquetQuery}
	  */
	 const query = new ParquetQuery({
	     onProgress: (pct) => {
		 document.getElementById("parquet-progress").value = pct;
	     },
	     onLog: (s) => {
		 log(`parquet ${s}`);
	     },
	     onReady: () => {
		 moduleLoaded("query");
		 showCachedIndexes();
	     },
	     onError: (text) => {
		 if (!loadedModules.has("query")) {
		     moduleFailed("query", text);
		 }
	     },
	 });
    
    /**
     * True once a Parquet index has been built for the loaded model.
     * @type {boolean}
     */
    let indexReady = false;

    /**
     * True while a local search is running; keeps "Search locally" disabled.
     * @type {boolean}
     */
    let searching = false;

    /**
    * Shows or hides the search progress meter, which covers the whole search including
    * any embedding generated for it.
    * @param {boolean} on
    * @returns {void}
    */
    const setSearching = (on) => {
	const meter = document.getElementById("search-meter");
	meter.hidden = !on;
	meter.active = on;
    };
    
	 /**
	  * Enables "Search locally" when an index and embeddings are both available.
	  * @returns {void}
	  */
	 const updateLocal = () => {
	     const hasSource = !!(document.getElementById("parquet-file").files[0] || document.getElementById("parquet-url").value.trim() || document.getElementById("parquet-cached").value);
	     document.getElementById("similar-local").disabled = searching || !(indexReady && activeKind() && loadedState.model && hasSource);
	     document.getElementById("parquet-load").disabled = !loadedState.model;
	     let why = "";
	     if (!loadedState.model) {
		 why = "An embedding file needs to be loaded before you can load a Parquet index.";
	     } else if (!activeKind()) {
		 why = "Generate a text or image embedding first.";
	     }
	     if (!why && !hasSource) {
		 why = "Load Parquet index to search locally";
	     }
	     document.getElementById("parquet-load-hint").textContent = why;
	 };
    
	 /**
	  * Shows a status line for the Parquet index.
	  * @param {string} s
	  */
	 const parquetStatus = (s) => {
	     document.getElementById("parquet-status").textContent = s;
	 };
	 
	 /**
	  * Fills the cached-index select with the Parquet URLs that have a cached index; it is
	  * hidden when there are none.
	  * @returns {Promise<void>}
	  */
	 const showCachedIndexes = async () => {
	     const select = document.getElementById("parquet-cached");
	     let items = [];
	     try {
		 items = await query.list();
	     } catch (err) {
		 log(`parquet: cannot list cached indexes: ${err.message}`);
	     }
	     const current = document.getElementById("parquet-url").value.trim();
	     const none = el("option", "Cached indexes...");
	     none.value = "";
	     select.replaceChildren(none);
	     const seen = new Set();
	     for (const item of items) {
		 if (seen.has(item.url)) {
		     continue;
		 }
		 seen.add(item.url);
		 const local = item.url.startsWith("local://");
		 const name = local ? decodeURIComponent(item.url.slice(8).replace(/\?\d+$/, "")) + " (local file)" : shortModelName(item.url);
		 const o = el("option", `${name} (${item.count} ${item.model} embeddings)`);
		 o.value = item.url;
		 o.title = item.url;
		 o.dataset.model = item.model;
		 select.append(o);
	     }
	     select.value = seen.has(current) ? current : "";
	     select.hidden = seen.size === 0;
	 };

	 // Choosing a cached index fills the URL and drops any chosen file, which would take precedence.
	 // A cached local file is reopened from its "local://" URL, which the query worker resolves
	 // from the copy it kept in the cache.
	 document.getElementById("parquet-cached").onchange = (e) => {
	     document.getElementById("parquet-url").value = e.target.value;
	     document.getElementById("parquet-file").value = "";
	 };
	 document.getElementById("parquet-url").addEventListener("input", () => {
	     document.getElementById("parquet-cached").value = "";
	 });
	 document.getElementById("parquet-file").addEventListener("change", () => {
	     document.getElementById("parquet-cached").value = "";
	 });

    document.getElementById("parquet-load").onclick = async () => {
	
	const source = document.getElementById("parquet-file").files[0] || document.getElementById("parquet-url").value.trim();
	
	if (!source) {
	    return;
	}
	
	// A chosen cached index knows which model it was built for, so no model need be loaded.
	const cachedSelect = document.getElementById("parquet-cached");
	const cachedModel = cachedSelect.selectedOptions[0] ? cachedSelect.selectedOptions[0].dataset.model || "" : "";
	
	// Fall back to the model chosen in the Models section, even if it has not been loaded yet.
	const chosenFile = document.getElementById("model-file").files[0];
	const chosenUrl = document.getElementById("url").value.trim();
	const chosenModel = chosenFile ? chosenFile.name : (chosenUrl ? shortModelName(chosenUrl) : "");
	
	const rowModel = (document.getElementById("parquet-model").value.trim() || cachedModel || modelName || chosenModel).replace(/^local:\/\//, "");
	
	// With no model name the query module uses the file's only model, or lists the choices.
	
	indexReady = false;
	
	updateLocal();
	
	document.getElementById("parquet-progress").value = 0;
	
	parquetStatus("opening...");
	
	     try {
		 const t0 = performance.now();
		 const info = await query.open(source);
		 log(`parquet: ${info.rows} rows in ${info.row_groups} row groups (${(info.size / 1e6).toFixed(1)} MB)`);
		 parquetStatus(`indexing rows for ${rowModel}...`);
		 const built = await query.build(rowModel);
		 const secs = ((performance.now() - t0) / 1000).toFixed(1);
		 parquetStatus(`${built.count} ${built.model || rowModel} embeddings (${built.dim} dims)${built.cached ? ", from cache" : ""}`);
		 log(`parquet: index ready in ${secs}s${built.skipped ? `, skipped ${built.skipped} rows with another dimension` : ""}`);
		 document.getElementById("parquet-progress").value = 100;
		 indexReady = true;
	     } catch (err) {
		 parquetStatus("failed");
		 log(`error: parquet: ${err.message}`);
	     }
	     updateLocal();
	     showCachedIndexes();
	 };
	 
	 for (const id of ["parquet-url", "parquet-file", "parquet-cached"]) {
	     document.getElementById(id).addEventListener("input", updateLocal);
	     document.getElementById(id).addEventListener("change", updateLocal);
	 }
	 
	 document.getElementById("parquet-purge").onclick = async () => {
	     await query.purge();
	     showCachedIndexes();
	     indexReady = false;
	     updateLocal();
	     parquetStatus("cache purged");
	 };
	 
	 /**
	  * Searches the Parquet index with the active embeddings and shows the results.
	  * @returns {Promise<void>}
	  */
	 const searchLocal = async () => {
	     const kind = activeKind();
	     if (!kind || !indexReady) {
		 return;
	     }
	     log(`similar: searching locally with ${kind} embeddings`);
	     resetResults();
	     searching = true;
	     setSearching(true);
	     updateLocal();
	     try {
		 const t0 = performance.now();
		 const records = await query.search(generated[kind], 20);
		 log(`similar: local search took ${Math.round(performance.now() - t0)}ms`);
		 showResults(records);
	     } catch (err) {
		 log(`error: local search failed: ${err.message}`);
	     } finally {
		 searching = false;
		 setSearching(false);
		 updateLocal();
	     }
	 };
    
	 document.getElementById("similar-local").onclick = searchLocal;
	 
	 /**
	  * Picks the source of a model: a chosen file takes precedence over the URL field,
	  * and the worker keys it by a "local://" name.
	  * @param {string} urlId - Id of the URL input.
	  * @param {string} fileId - Id of the file input.
	  * @returns {{url: string, file: ?File}}
	  */
	 const modelSource = (urlId, fileId) => {
	     const file = document.getElementById(fileId).files[0];
	     return file ? { url: `local://${file.name}`, file } : { url: document.getElementById(urlId).value, file: null };
	 };
    
	 document.getElementById("load").onclick = () => {
	     const model = modelSource("url", "model-file");
	     const proj = modelSource("mmproj", "mmproj-file");
	     modelName = model.file ? model.file.name : shortModelName(model.url);
	     generated.text = generated.image = null;
	     loadedState.model = loadedState.mmproj = false;
	     updateBusy();
	     showEmbeddings();
	     worker.postMessage({ kind: "load", url: model.url, file: model.file, pooling: "mean", mmproj: proj.url, mmprojFile: proj.file });
	 };
    
	 // Removes the model (and projector, if given) from the worker's OPFS cache.
	 document.getElementById("purge").onclick = () =>
	     worker.postMessage({ kind: "purge", urls: [document.getElementById("url").value, document.getElementById("mmproj").value] });
    
    document.getElementById("embed").onclick = () => {
	// A new text embedding replaces any image, its embeddings, and earlier similar results.
	generated.image = null;
	clearPreview();
	
      document.getElementById("image").value = "";
	     excluded = null;
	     // resetResults("Similar records will appear here.");
	     const id = String(++n);
	     start(id, "text");
	     worker.postMessage({ kind: "embed", id, text: document.getElementById("text").value });
	 };
	 
	 // Choosing a cached model fills its URL field and drops any chosen file, which would take precedence.
	 for (const select of document.querySelectorAll("select.cached")) {
	     select.onchange = () => {
		 document.getElementById(select.dataset.target).value = select.value;
		 document.getElementById(select.dataset.file).value = "";
	     };
	     document.getElementById(select.dataset.target).addEventListener("input", () => {
		 select.value = "";
	     });
	     document.getElementById(select.dataset.file).addEventListener("change", () => {
		 select.value = "";
	     });
	 }

	 // Clear controls empty the inputs named in data-clear (URL, file, cached select).
	 for (const control of document.querySelectorAll(".clear")) {
	     const clear = () => {
		 for (const id of control.dataset.clear.split(" ")) {
		     const target = document.getElementById(id);
		     target.value = "";
		     // The image input's change handler also hides its preview.
		     target.dispatchEvent(new Event(target.type === "file" ? "change" : "input", { bubbles: true }));
		 }
	     };
	     // The control sits inside a <label>, so stop the click from activating the input.
	     control.onclick = (e) => {
		 e.preventDefault();
		 clear();
	     };
	     control.onkeydown = (e) => {
		 if (e.key === "Enter" || e.key === " ") {
		     e.preventDefault();
		     clear();
		 }
	     };
	 }

	 for (const a of document.querySelectorAll("a.pick")) {
	     a.onclick = (e) => {
		 e.preventDefault();
		 const el = document.getElementById(a.dataset.target);
		 el.value = a.dataset.value;
		 el.dispatchEvent(new Event("input", { bubbles: true }));
	     };
	 }
	 /**
	  * Hides and releases the preview image.
	  * @returns {void}
	  */
	 const clearPreview = () => {
	     const img = document.getElementById("preview");
	     if (img.src.startsWith("blob:")) {
		 URL.revokeObjectURL(img.src);
	     }
	     img.hidden = true;
	     img.removeAttribute("src");
	 };
	 // Show a small preview of the chosen image.
	 document.getElementById("image").onchange = (e) => {
	     const img = document.getElementById("preview");
	     if (img.src.startsWith("blob:")) {
		 URL.revokeObjectURL(img.src);
	     }
	     const f = e.target.files[0];
	     img.hidden = !f;
	     if (f) {
		 img.src = URL.createObjectURL(f);
		 document.getElementById("text").value = "";
		 generated.text = null;
	     } else {
		 img.removeAttribute("src");
		 generated.image = null;
	     }
	     showEmbeddings();
	     updateSources();
	 };
	 // Typing text zeroes out the image input (and its preview).
	 document.getElementById("text").addEventListener("input", () => {
	     if (document.getElementById("text").value !== "") {
		 document.getElementById("image").value = "";
		 generated.image = null;
		 clearPreview();
	     } else {
		 generated.text = null;
	     }
	     showEmbeddings();
	     updateSources();
	 });
	 updateSources();
	 updateLocal();
	 // Decode the image in the page and send RGBA pixels; the Go side drops the alpha.
	 document.getElementById("embed-image").onclick = () => {
	     const file = document.getElementById("image").files[0];
	     if (!file) {
		 return;
	     }
	     excluded = null;
	     embedImage(file);
	 };
	 
	 /**
	  * Embeds an image (Blob or <img>), scaled so neither side exceeds MAX_IMAGE_SIDE;
	  * with andSearch, the local search follows.
	  * @param {Blob|HTMLImageElement} file
	  * @param {boolean} [andSearch=false]
	  * @returns {Promise<void>}
	  */
	 const embedImage = async (file, andSearch = false) => {
	     const id = String(++n);
	     start(id, "image");
	     let bmp;
	     try {
		 bmp = await createImageBitmap(file);
	     } catch (err) {
		 pending.delete(id);
		 updateBusy();
		 if (andSearch) {
		     setSearching(false);
		 }
		 log(`error: cannot decode image: ${err}`);
		 return;
	     }
	     if (andSearch) {
		 searchAfter.add(id);
	     }
	     // Downscale so the longest side is at most MAX_IMAGE_SIDE; encoder cost grows with pixel count.
	     const scale = Math.min(1, MAX_IMAGE_SIDE / Math.max(bmp.width, bmp.height));
	     const w = Math.max(1, Math.round(bmp.width * scale));
	     const h = Math.max(1, Math.round(bmp.height * scale));
	     const canvas = new OffscreenCanvas(w, h);
	     const g = canvas.getContext("2d");
	     g.imageSmoothingQuality = "high";
	     g.drawImage(bmp, 0, 0, w, h);
	     const px = g.getImageData(0, 0, w, h).data;
	     log(`image: ${bmp.width}x${bmp.height} sent as ${w}x${h}`);
	     worker.postMessage(
		 { kind: "embed-image", id, width: w, height: h, rgba: px.buffer },
		 [px.buffer],
	     );
	 };
	 
	 /**
	  * Embeds the already-loaded result <img> (no second fetch) and searches the local
	  * Parquet index with it, leaving the source record out of the results.
	  * @param {HTMLImageElement} img
	  * @param {{depictionId: string, subjectId: string}} exclude
	  * @returns {Promise<void>}
	  */
	 const findSimilarToImage = async (img, exclude) => {
	     if (document.getElementById("embed-image").dataset.off === "1") {
		 log("error: image embeddings need a loaded model with an mmproj");
		 return;
	     }
	     if (!indexReady) {
		 log("error: the Parquet index is no longer loaded");
		 return;
	     }
	     document.getElementById("image").value = "";
	     document.getElementById("text").value = "";
	     clearPreview();
	     const preview = document.getElementById("preview");
	     preview.crossOrigin = "anonymous";
	     preview.src = img.currentSrc || img.src;
	     preview.hidden = false;
	     updateSources();
	     excluded = exclude;
	     resetResults();
	     setSearching(true);
	     await embedImage(img, true);
	 };

});
