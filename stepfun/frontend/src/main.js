// Browser-side embedding inference for the StepFun mini knowledge base.
// - Runtime: Transformers.js (ONNX Runtime Web)
// - Model:   Xenova/all-MiniLM-L6-v2, INT8 ("q8") quantized, 384 dims
// - Backend: WebGPU when available, automatic fallback to WASM
// No cloud inference API is ever called; vectors are computed on this device.

const TF_BASE = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.6";
const MODEL_ID = "Xenova/all-MiniLM-L6-v2";
const DIM = 384;

const $ = (id) => document.getElementById(id);

const state = {
  extractor: null,
  device: null,
  loadMs: null,
  docSeq: 0,
  timing: {},
};

function recordTiming(key, ms) {
  const entry = state.timing[key] ??= { first: ms, last: ms };
  entry.last = ms;
  $(`t-${key}-first`).textContent = fmt(entry.first);
  $(`t-${key}-last`).textContent = fmt(entry.last);
}

const fmt = (ms) => (ms == null ? "-" : `${ms.toFixed(1)} ms`);

function setStatus(id, text, isError = false) {
  const el = $(id);
  el.textContent = text;
  el.classList.toggle("error", isError);
}

function setProgress(id, ratio) {
  const wrap = $(id);
  wrap.classList.remove("hidden");
  wrap.querySelector(".bar").style.width = `${Math.round(ratio * 100)}%`;
}

function banner(text, kind = "info") {
  const el = $("env-banner");
  el.textContent = text;
  el.className = `banner ${kind}`;
}

// ---------------------------------------------------------------------------
// Model loading (WebGPU with explicit WASM fallback)
// ---------------------------------------------------------------------------

async function loadModel() {
  const host = $("hf-host").value;
  $("load-model-btn").disabled = true;
  setStatus("model-status", "正在加载 transformers.js …");

  const { pipeline, env } = await import(`${TF_BASE}/transformers.min.js`);
  env.remoteHost = host; // hf-mirror works better in CN networks
  env.allowLocalModels = false;

  const hasWebGPU = typeof navigator !== "undefined" && !!navigator.gpu;
  const devices = hasWebGPU ? ["webgpu", "wasm"] : ["wasm"];
  let lastError = null;

  for (const device of devices) {
    const started = performance.now();
    try {
      setStatus("model-status", `尝试 ${device} 后端加载 ${MODEL_ID} (INT8)…`);
      state.extractor = await pipeline("feature-extraction", MODEL_ID, {
        dtype: "q8",
        device,
        progress_callback: (info) => {
          if (info.status === "progress" && info.total) {
            setProgress("model-progress", info.loaded / info.total);
            setStatus(
              "model-status",
              `下载模型 ${device}: ${(info.loaded / 1e6).toFixed(1)} / ${(info.total / 1e6).toFixed(1)} MB`
            );
          }
        },
      });
      // warmup so first-query latency is not dominated by shader compilation
      await state.extractor("warmup", { pooling: "mean", normalize: true });
      state.device = device;
      state.loadMs = performance.now() - started;
      recordTiming("load", state.loadMs);
      setProgress("model-progress", 1);
      $("backend-label").textContent = device.toUpperCase();
      setStatus(
        "model-status",
        `模型已就绪（${device.toUpperCase()}，INT8 量化）。加载耗时 ${fmt(state.loadMs)}。`,
        false
      );
      $("ingest-btn").disabled = false;
      $("search-btn").disabled = false;
      return;
    } catch (error) {
      lastError = error;
      console.warn(`${device} backend failed`, error);
    }
  }
  setStatus("model-status", `模型加载失败：${lastError}`, true);
  $("load-model-btn").disabled = false;
}

// ---------------------------------------------------------------------------
// Embedding helpers
// ---------------------------------------------------------------------------

async function embed(texts) {
  const started = performance.now();
  const output = await state.extractor(texts, { pooling: "mean", normalize: true });
  const vectors = output.tolist(); // Array<[float]>
  recordTiming("embed", performance.now() - started);
  return vectors;
}

async function embedQuery(text) {
  const started = performance.now();
  const output = await state.extractor(text, { pooling: "mean", normalize: true });
  const vector = output.tolist()[0];
  recordTiming("qembed", performance.now() - started);
  return vector;
}

// ---------------------------------------------------------------------------
// Corpus import
// ---------------------------------------------------------------------------

function splitCorpus(raw) {
  return raw
    .split(/\n\s*\n/)
    .map((chunk) => chunk.trim())
    .filter((chunk) => chunk.length > 0);
}

async function ingest() {
  if (!state.extractor) return;
  const chunks = splitCorpus($("corpus").value);
  if (chunks.length === 0) {
    setStatus("ingest-status", "请先粘贴文本或选择文件。", true);
    return;
  }
  $("ingest-btn").disabled = true;
  try {
    const batchSize = 8;
    const vectors = [];
    for (let i = 0; i < chunks.length; i += batchSize) {
      const batch = chunks.slice(i, i + batchSize);
      const batchVectors = await embed(batch);
      vectors.push(...batchVectors);
      setProgress("ingest-progress", (i + batch.length) / chunks.length);
      setStatus(
        "ingest-status",
        `本地向量化中 ${Math.min(i + batchSize, chunks.length)} / ${chunks.length} 段…`
      );
    }

    const documents = chunks.map((text, index) => ({
      id: `doc_${Date.now()}_${state.docSeq++}`,
      text,
      embedding: vectors[index],
      metadata: {
        source: "manual-input",
        chars: text.length,
        imported_at: new Date().toISOString(),
      },
    }));

    setStatus("ingest-status", "写入 seekdb …");
    const response = await fetch("/api/ingest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ documents }),
    });
    if (!response.ok) throw new Error((await response.json()).detail ?? response.statusText);
    const result = await response.json();
    recordTiming("ingest", result.elapsed_ms);
    setProgress("ingest-progress", 1);
    setStatus(
      "ingest-status",
      `已写入 ${result.added} 段文档（服务端写入 ${fmt(result.elapsed_ms)}，含索引刷新）。`
    );
    refreshStats();
  } catch (error) {
    setStatus("ingest-status", `导入失败：${error.message}`, true);
  } finally {
    $("ingest-btn").disabled = false;
  }
}

async function refreshStats() {
  try {
    const response = await fetch("/api/stats");
    const data = await response.json();
    $("doc-count").textContent = `seekdb 文档数：${data.count}（collection: ${data.collection}）`;
  } catch {
    $("doc-count").textContent = "seekdb 文档数：后端不可用";
  }
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

async function search() {
  if (!state.extractor) return;
  const queryText = $("query").value.trim();
  if (!queryText) return;
  $("search-btn").disabled = true;
  const e2eStarted = performance.now();
  try {
    setStatus("search-status", "本地生成查询向量…");
    const queryEmbedding = await embedQuery(queryText);
    setStatus("search-status", "seekdb 混合检索中…");
    const searchStarted = performance.now();
    const response = await fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query_text: queryText, query_embedding: queryEmbedding, top_k: 5 }),
    });
    if (!response.ok) throw new Error((await response.json()).detail ?? response.statusText);
    const result = await response.json();
    recordTiming("search", performance.now() - searchStarted);
    recordTiming("e2e", performance.now() - e2eStarted);
    renderResults(result);
    setStatus(
      "search-status",
      `seekdb ${result.backend} · 检索 ${fmt(result.elapsed_ms)} · 端到端 ${fmt(performance.now() - e2eStarted)}`
    );
  } catch (error) {
    setStatus("search-status", `检索失败：${error.message}`, true);
  } finally {
    $("search-btn").disabled = false;
  }
}

function renderResults(result) {
  const container = $("results");
  if (!result.hits.length) {
    container.innerHTML = `<p class="hint">没有命中结果。试试别的关键词，或先导入语料。</p>`;
    return;
  }
  container.innerHTML = result.hits
    .map(
      (hit, index) => `
      <div class="hit">
        <div class="hit-head">#${index + 1} <span class="hit-id">${escapeHtml(hit.id)}</span></div>
        <div class="hit-text">${escapeHtml(hit.text ?? "")}</div>
      </div>`
    )
    .join("");
}

const escapeHtml = (text) =>
  text.replace(/[&<>"']/g, (char) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char]
  );

// ---------------------------------------------------------------------------
// File input
// ---------------------------------------------------------------------------

async function loadFiles(files) {
  const parts = [];
  for (const file of files) {
    parts.push(await file.text());
  }
  const existing = $("corpus").value.trim();
  $("corpus").value = existing ? `${existing}\n\n${parts.join("\n\n")}` : parts.join("\n\n");
}

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------

function checkEnv() {
  const hasWebGPU = typeof navigator !== "undefined" && !!navigator.gpu;
  const isSecure = window.isSecureContext;
  banner(
    hasWebGPU
      ? "✓ WebGPU 可用，将优先使用 WebGPU 后端（不支持时自动降级 WASM）"
      : "⚠ 当前浏览器/上下文不支持 WebGPU，将使用 WASM 后端（需要安全上下文）",
    hasWebGPU ? "ok" : "warn"
  );
  if (!isSecure && !["localhost", "127.0.0.1"].includes(location.hostname)) {
    banner("⚠ 非 localhost 的非安全上下文：WebGPU 与 WASM SIMD 可能被禁用，请用 http://localhost 访问", "warn");
  }
}

$("load-model-btn").addEventListener("click", loadModel);
$("ingest-btn").addEventListener("click", ingest);
$("search-btn").addEventListener("click", search);
$("file-input").addEventListener("change", (event) => loadFiles([...event.target.files]));
$("query").addEventListener("keydown", (event) => {
  if (event.key === "Enter") search();
});

checkEnv();
refreshStats();
