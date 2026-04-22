# efficient_retrieval

面向 RAG / 知识库问答场景的检索增强示例集合，重点演示如何在基础向量检索之上加入：

- Multi-Query Retrieval（多查询改写召回）
- Hybrid Retrieval（BM25 + 向量混合检索）
- Rerank（召回后重排）
- PDF 文档抽取、切片、落盘缓存与问答

这个目录适合用于：

- 学习检索增强的典型工程套路
- 搭建一个比“纯向量检索”更强的 RAG baseline
- 作为后续团队 Skill / Harness 的 `retrieval` pattern 参考

---

## 目录结构

```text
efficient_retrieval/
├─ chatpdf-faiss-HybridSearch.py
├─ chatpdf-faiss-HybridSearch-Rerank.py
├─ multi-query-retriever.py
├─ requirements.txt
├─ README.md
├─ vector_db/
│  ├─ index.faiss
│  ├─ index.pkl
│  └─ page_info.pkl
└─ vector_db_hybrid/
   ├─ chunks.pkl
   ├─ index.faiss
   ├─ index.pkl
   └─ page_info.pkl
```

---

## 文件说明

### 1. `multi-query-retriever.py`

演示如何在已有向量库基础上，通过 LLM 为同一个问题生成多个查询变体，然后分别检索、合并、去重，从而提升召回覆盖率。

核心能力：

- 基于 LLM 生成多个 query 变体
- 对每个 query 执行向量检索
- 合并结果并按内容去重

适合场景：

- 用户表达不稳定
- 问题可能存在多种表述方式
- 想提升召回覆盖率而不是只依赖单一 query

---

### 2. `chatpdf-faiss-HybridSearch.py`

演示从 PDF 中抽取文本，建立 FAISS 向量索引，并结合 BM25 做混合检索，再基于检索结果进行问答。

核心能力：

- PDF 文本抽取
- 文本切片
- 向量索引构建与落盘
- BM25 + 向量混合检索
- 页码回溯
- 基于检索上下文生成回答

适合场景：

- 中文文档问答
- 需要兼顾关键词匹配和语义检索
- 希望构建比纯向量检索更稳的 baseline

---

### 3. `chatpdf-faiss-HybridSearch-Rerank.py`

在 Hybrid Retrieval 的基础上，进一步加入 Rerank 模型，对候选文档重新打分和排序。

核心能力：

- 先召回候选文档
- 再使用 `bge-reranker` 类模型做重排
- 提升最终进入回答阶段的文档质量

适合场景：

- 检索候选较多但排序不稳定
- 想提升最终回答的相关性
- 希望对召回结果做二次精排

---

## 核心思路

这个目录里的几个脚本体现的是一个逐步增强的检索思路：

### 第 1 层：向量检索

先用 embedding + FAISS 做基本语义检索。

### 第 2 层：Multi-Query

对用户问题生成多个变体，扩大召回面，减少“问法不同导致漏召回”的问题。

### 第 3 层：Hybrid Retrieval

将：

- BM25 的关键词能力
- Vector Search 的语义能力

组合起来，兼顾精准匹配和语义相似性。

### 第 4 层：Rerank

对召回结果重新排序，把更相关的文档排在前面，提高最终回答质量。

这也是很多真实 RAG 系统常见的工程路径：

**召回 → 扩召回 → 混合召回 → 重排 → 生成回答**

---

## 依赖安装

在仓库根目录执行：

```bash
pip install -r efficient_retrieval/requirements.txt
```

当前 `requirements.txt` 已包含：

- `langchain==0.3.25`
- `langchain_community==0.3.23`
- `langchain_openai==0.3.16`
- `PyPDF2==3.0.1`

但从当前脚本实际使用情况看，还依赖以下库：

- `faiss`
- `jieba`
- `rank_bm25`
- `langchain-text-splitters`
- `langchain-core`
- `modelscope`（仅 Rerank 脚本需要）
- `torch`（仅 Rerank 脚本需要）

如果你准备长期维护这个目录，建议后续把这些依赖也补充到 `requirements.txt` 中。

---

## 环境变量

所有脚本都依赖：

```env
DASHSCOPE_API_KEY=your_dashscope_api_key
```

PowerShell 示例：

```powershell
$env:DASHSCOPE_API_KEY="your_dashscope_api_key"
```

如果没有设置，会在运行时报错：

- `请设置环境变量 DASHSCOPE_API_KEY`

---

## 运行方式

### 1. 运行 Multi-Query Retrieval 示例

```bash
python efficient_retrieval/multi-query-retriever.py
```

说明：

- 依赖已有的 `vector_db/` 向量索引
- 会先生成多个 query 变体
- 然后执行检索并输出去重后的结果

---

### 2. 运行 Hybrid Search 示例

```bash
python efficient_retrieval/chatpdf-faiss-HybridSearch.py
```

说明：

- 如果本地已经存在 `vector_db_hybrid/`，会直接加载
- 否则会从 PDF 抽取文本并重新构建向量数据库
- 会执行混合检索并基于上下文生成回答

---

### 3. 运行 Hybrid Search + Rerank 示例

```bash
python efficient_retrieval/chatpdf-faiss-HybridSearch-Rerank.py
```

说明：

- 在 Hybrid Search 基础上再做重排
- 首次运行可能需要下载 rerank 模型
- 相比纯 Hybrid Search，排序更精细，但运行更慢

---

## 向量库与缓存说明

当前目录中已经包含一些本地索引缓存目录：

### `vector_db/`

用于 `multi-query-retriever.py` 加载已有向量库。

### `vector_db_hybrid/`

用于 `chatpdf-faiss-HybridSearch.py` 和 `chatpdf-faiss-HybridSearch-Rerank.py` 的混合检索场景。

通常会保存：

- `index.faiss`：FAISS 向量索引
- `index.pkl`：向量库相关序列化数据
- `page_info.pkl`：文本块与页码映射
- `chunks.pkl`：切片结果（混合检索时需要）

这些缓存文件的作用是：

- 避免重复 embedding
- 避免重复处理 PDF
- 提升后续运行速度
- 降低 token 和计算消耗

---

## 当前脚本的输入数据说明

### `multi-query-retriever.py`

默认直接加载：

- `efficient_retrieval/vector_db/`

并使用固定示例 query：

- `客户经理的考核标准是什么？`

### `chatpdf-faiss-HybridSearch.py`

默认读取：

- `./测试.pdf`

如果目录下没有对应 PDF，脚本会无法重建索引。

### `chatpdf-faiss-HybridSearch-Rerank.py`

同样依赖 PDF 文件和本地缓存，同时还会加载 rerank 模型。

使用这个示例时，建议：

- 在 README 中明确说明哪些文件是示例输入
- 将模型缓存和本地生成的中间产物排除在版本控制之外

---

## 代码实现亮点

### 1. Multi-Query 变体生成

通过 LLM 把一个 query 扩展成多个角度不同但语义相关的查询，从而提升召回覆盖率。

### 2. Hybrid Retriever

通过 `BM25 + Vector` 的融合方式，兼顾：

- 关键词强匹配
- 语义相似度匹配

### 3. 页码映射

在 PDF 文本抽取后，保留 chunk 对应页码信息，方便回答后回溯出处。

### 4. 向量库落盘

把知识库索引和元数据保存到本地，避免每次重复构建。

### 5. Rerank 二次排序

对召回候选做更精细的相关性打分，从而提升最终回答的上下文质量。

---

## 常见问题

### 1. 报错 `请设置环境变量 DASHSCOPE_API_KEY`

说明没有正确配置 DashScope API Key。

请先设置环境变量后再运行。

---

### 2. `vector_db` 或 `vector_db_hybrid` 加载失败

可能原因：

- 本地缓存目录不存在
- 缓存文件不完整
- FAISS / pickle 文件与当前依赖版本不兼容

建议：

- 删除旧缓存后重新构建
- 确保相关依赖版本一致

---

### 3. PDF 文件找不到

`chatpdf-faiss-HybridSearch.py` 默认读取：

- `./测试.pdf`

请确认：

- 文件存在
- 路径正确
- 编码和文件名在当前系统可正常识别

---

### 4. `jieba` / `rank_bm25` / `faiss` 缺失

虽然当前 `requirements.txt` 没有完全列出这些依赖，但脚本实际使用了它们。

如果运行时报缺包，请额外安装对应依赖。

---

### 5. Rerank 模型下载失败

`chatpdf-faiss-HybridSearch-Rerank.py` 使用了 `modelscope` 下载模型。

可能原因：

- 网络不可访问 ModelScope
- 模型缓存目录没有写权限
- 本地环境缺少 `torch`

---

## 适合用于什么场景

这个目录很适合作为以下场景的参考实现：

- 构建中文文档问答系统
- 对比不同 retrieval 增强策略
- 学习从“纯向量检索”升级到“混合检索 + 重排”
- 沉淀团队内部的 RAG baseline
- 作为 Skill / Harness 中 `retrieval pipeline` 的核心参考案例

---

## 后续建议

如果你准备把这个目录进一步沉淀成团队可复用资产，建议继续做这些优化：

1. 补齐 `requirements.txt` 中缺失的依赖
2. 将 PDF 路径和 query 改成配置项
3. 把示例脚本拆成模块：
   - loader
   - chunker
   - embedder
   - retriever
   - reranker
   - qa
4. 提供一个统一入口脚本
5. 增加 `.env.example`
6. 增加真实数据和缓存目录的忽略规则
7. 增加效果对比说明：
   - vector only
   - multi-query
   - hybrid
   - hybrid + rerank

---

## 版本控制建议

建议保留：

- 示例脚本
- `requirements.txt`
- 本 README
- 必要的小型演示数据（如果有）

建议排除：

- 大体积模型缓存
- 本地私有 PDF
- 过大的向量库缓存
- 包含敏感信息的测试数据

如果你后续要把这个目录纳入 Skill / Harness 体系，它最适合作为这些能力的参考案例：

- `build-retrieval-pipeline`
- `build-hybrid-retriever`
- `build-multi-query-retriever`
- `add-reranker-to-rag`
