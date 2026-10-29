# 端侧 AI 推理 + seekdb 混合检索 · 迷你知识库（StepFun 赛题）

> 提交产物：可运行的 Web 应用、完整源码（本目录）和本页技术说明。

## 1. 应用概述

用户导入一批文本文档后，应用**在浏览器本地**将内容编码为向量，把向量、原文和元数据写入 seekdb；用户输入查询后，应用同样**在本地**生成查询向量，由 seekdb 完成"全文 + 向量"混合检索（RRF 融合），展示排序结果及对应原文。

- 前端：纯静态单页应用（原生 ES Modules），运行于浏览器，负责端侧推理与交互
- 后端：FastAPI 薄服务，仅作为 seekdb 的存储/检索代理，**不调用任何云端推理 API**
- 数据层：seekdb（服务器模式），`hybrid_search()` 一条调用内完成全文候选 + 向量候选 + RRF 融合

## 2. 端侧推理与模型量化

| 项目 | 取值 |
| --- | --- |
| 运行时 | Transformers.js v3（ONNX Runtime Web） |
| 模型 | `Xenova/all-MiniLM-L6-v2`（开源，384 维；来源 huggingface.co/Xenova/all-MiniLM-L6-v2，ONNX 权重） |
| 量化 | INT8（Transformers.js `dtype: "q8"`） |
| 推理后端 | 优先 WebGPU；`navigator.gpu` 不可用或初始化失败时自动降级 WASM，并在界面上明确提示 |
| 模型分发 | 支持 hf-mirror.com / huggingface.co 两个源切换（`模型源`下拉框），适配不同网络环境 |

### 量化报告（提交前在本机实测后填写）

| 指标 | FP32 | INT8 (q8) |
| --- | --- | --- |
| 模型文件总体积 | 约 90 MB | 约 23 MB |
| 压缩比 | — | ≈ 3.9x |
| 检索质量（Recall@5，自建 20 问评测集） | 基线 | 变化 ≤ 1%（实测后填写） |

> 测量口径：同一浏览器、同一语料与查询集，分别以 `dtype: "fp32"` 与 `dtype: "q8"` 加载模型；质量用 Recall@K 对比，报告中保留原始数据。

### 分阶段计时（示例表格，实测后填写）

| 阶段 | 首次 | 预热后（3 次中位数） |
| --- | --- | --- |
| 模型下载/加载 | ___ ms | — |
| 文档向量化（20 段，batch=8） | ___ ms | ___ ms |
| 写入 + 索引刷新 | ___ ms | ___ ms |
| 查询向量化 | ___ ms | ___ ms |
| seekdb 混合检索 | ___ ms | ___ ms |
| 端到端（查询→结果） | ___ ms | ___ ms |

应用内置"性能计时"面板自动记录以上各阶段，支持重复测量。

## 3. seekdb 数据层

- Collection 建表时显式 `embedding_function=None` + `HNSWConfiguration(dimension=384, distance="cosine")`：**服务端不运行任何 embedding 模型**，向量由浏览器传入
- 写入：`collection.add(ids, documents, embeddings, metadatas)`，随后 `refresh_index()` 推进异步 HNSW 索引可见性（seekdb ≥ 1.3.0）
- 检索：`collection.hybrid_search(query={"where_document": {"$contains": ...}}, knn={"query_embeddings": [...]}, rank={"rrf": {}}, n_results=K)` —— 全文（BM25/IK）与向量（HNSW）两路召回在 seekdb 内部用 RRF 融合
- 支持增量导入：每次导入生成唯一 `doc_id`，写入并刷新索引后立即可检索

## 4. 合规声明

- 文档向量化与查询向量化均在本页浏览器内完成，无云端推理 API 调用，无服务端预计算回放
- 向量、原文、元数据的存储与混合检索全部由 seekdb 完成，未使用其他数据库或客户端内存检索替代

## 5. 运行步骤

```bash
# 1. 启动 seekdb（Windows 推荐 OceanBase Desktop，或任意 Docker/远程实例）
#    https://www.oceanbase.ai/download  → 默认 127.0.0.1:2881, root, 空密码

# 2. 启动后端（Python 3.11+）
cd backend
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # Windows
# 复制 .env.example 为 .env 并按需修改（或直接 export 环境变量）
.venv/Scripts/python app.py            # http://127.0.0.1:8000

# 3. 浏览器打开 http://127.0.0.1:8000
#    ① 加载模型（INT8，WebGPU/WASM 自动选择） → ② 粘贴文本导入 → ③ 输入查询检索
```

> 注意：WebGPU 需要 Chrome/Edge 113+ 且安全上下文；本机用 `http://127.0.0.1:8000` 即满足。首次加载需下载约 23 MB 模型文件。
