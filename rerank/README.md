# rerank

使用 `BAAI/bge-reranker-large` 演示 Rerank（重排序）能力的最小示例。

这个目录主要展示：

- 如何加载 `bge-reranker-large`
- 如何对 query / document 对进行相关性打分
- Rerank 在 RAG 召回后重排中的基本用法

---

## 目录结构

```text
rerank/
├─ bge-reranker.py
├─ requirements.txt
└─ README.md
```

---

## 什么是 Rerank

在 RAG 或检索系统中，通常会先做一轮召回，从知识库中找出一批候选文档。

Rerank 用于对这些候选文档再次打分和排序，把更相关的文档排到前面。

和 embedding 检索不同，Reranker 通常采用 cross-encoder 方式：

- query 和 document 一起输入模型
- 模型直接输出相关性分数
- 分数越高，通常表示相关性越强

由于计算成本更高，Rerank 一般用于对少量候选结果做二次精排，而不是替代第一轮召回。

---

## 当前脚本做了什么

`bge-reranker.py` 演示了两种基础用法：

### 1. 单条 query-document 对打分

示例输入：

- Query: `what is panda?`
- Document: `The giant panda is a bear species endemic to China.`

### 2. 多条候选文档打分

示例输入了 3 组 query-document 对，用于观察高相关、中等相关和不相关文本之间的分数差异。

---

## 依赖安装

在仓库根目录执行：

```bash
pip install -r rerank/requirements.txt
```

当前依赖：

- `modelscope==1.25.0`
- `torch==2.7.0`
- `transformers==4.49.0`

---

## 运行方式

在仓库根目录执行：

```bash
python rerank/bge-reranker.py
```

运行后会输出类似的相关性分数张量：

```text
tensor([x.xxxx])
tensor([x.xxxx, x.xxxx, x.xxxx])
```

---

## 模型与环境说明

当前脚本使用 `modelscope.snapshot_download` 自动下载模型：

- 模型名：`BAAI/bge-reranker-large`
- 缓存目录示例：`/root/autodl-tmp/models`

这意味着当前脚本更偏向 Linux / AutoDL 环境。

如果在 Windows 或本地其他路径运行，建议将模型缓存目录改为本机可写路径，例如：

```python
model_dir = snapshot_download('BAAI/bge-reranker-large', cache_dir='D:/models')
```

并同步修改后续模型加载路径。

---

## 实现原理

脚本的核心流程如下：

1. 下载或加载 `BAAI/bge-reranker-large`
2. 使用 `AutoTokenizer` 对 query-document 对进行编码
3. 使用 `AutoModelForSequenceClassification` 做前向计算
4. 从 `logits` 中取出相关性分数
5. 按分数高低比较候选文档相关性

---

## 适用场景

这个示例适合作为以下场景的参考：

- 学习 RAG 中 Rerank 的基本原理
- 验证 `bge-reranker-large` 的打分效果
- 构建“召回后重排”的最小原型
- 对比 embedding-only 检索与 rerank 后排序的差异

典型流程通常是：

1. 先用向量检索 / BM25 / 混合检索召回 Top-K 文档
2. 再用 reranker 对这 K 条候选重新打分
3. 取分数最高的前 N 条进入最终回答环节

---

## 常见问题

### 1. 模型下载失败

可能原因：

- 网络不可访问 ModelScope
- 当前环境没有权限写入缓存目录
- 指定缓存路径不存在或不可写

建议：

- 检查网络连接
- 修改 `cache_dir` 为本机可写目录

### 2. 本地路径报错

当前脚本使用了 Linux 风格的模型缓存路径。

如果不是在 AutoDL 环境运行，建议：

- 使用 `snapshot_download()` 返回的 `model_dir`
- 或改成本机绝对路径

### 3. CUDA / Torch 相关报错

如果本机没有正确的 PyTorch 环境，可能出现：

- GPU 不可用
- CUDA 版本不匹配
- CPU / GPU 依赖冲突

建议优先在干净虚拟环境中安装依赖。

### 4. 为什么分数不是 0 到 1

脚本里直接取的是模型输出的原始 `logits`，它不是概率值，因此不一定在 0 到 1 之间。

实际使用时更关注分数的相对大小，而不是把它当成概率解释。

---

## 后续可优化方向

如果准备把这个目录进一步沉淀成团队可复用资产，可以继续做这些改进：

1. 把硬编码路径改成配置项
2. 把“单条打分”和“批量重排”封装成函数
3. 增加一个真实 RAG 候选重排示例
4. 增加 `top_k rerank` 的完整例子
5. 支持从本地候选文档列表中自动排序输出

---

## 版本控制建议

建议保留：

- 示例脚本
- `requirements.txt`
- 本 README

建议排除：

- 模型缓存目录
- 本地下载的大模型文件
- 与个人环境强绑定的临时文件
- 运行过程中产生的中间产物

这个目录适合作为以下能力的参考案例：

- `add-reranker-to-rag`
- `build-rerank-module`
