# 基于 TF-IDF 的酒店文本推荐（Seattle Hotels）

一个使用 `scikit-learn` 实现的内容推荐小项目：  
通过酒店描述文本提取 TF-IDF 特征，计算酒店之间的余弦相似度，并返回相似酒店 Top-N。

## 项目功能

- 读取酒店数据集 `Seattle_Hotels.csv`
- 对酒店描述文本进行清洗（小写化、符号清理、停用词过滤）
- 使用 `CountVectorizer` 做 n-gram 词频统计并可视化 Top 词组
- 使用 `TfidfVectorizer` 提取文本特征
- 使用余弦相似度（`linear_kernel`）实现相似酒店推荐

## 项目结构

```text
TF-IDF_recommendation/
├─ hotel_rec.py
├─ Seattle_Hotels.csv
└─ README.md
```

## 运行环境

- Python 3.8+
- 依赖库：
  - `pandas`
  - `scikit-learn`
  - `matplotlib`

## 安装依赖

在项目根目录执行：

```bash
pip install pandas scikit-learn matplotlib
```

## 运行方式

```bash
python TF-IDF_recommendation/hotel_rec.py
```

脚本运行后会：

1. 打印数据集基础信息；
2. 输出词频相关调试信息；
3. 计算 TF-IDF 特征与酒店相似度矩阵；
4. 给出示例酒店的 Top10 推荐结果。

## 核心方法说明

### 1) CountVectorizer（词频统计）

- 用于提取 `n-gram` 词频特征；
- 主要用于分析“哪些短语在语料中出现最多”；
- 在本项目中用于生成 Top 词频统计和可视化。

### 2) TfidfVectorizer（TF-IDF 特征）

- 在词频基础上引入逆文档频率（IDF）；
- 降低全局高频常见词权重，提升更具区分度词语的权重；
- 生成用于相似度计算的特征矩阵。

### 3) 余弦相似度（Cosine Similarity）

- 对每个酒店文本向量计算两两相似度；
- 相似度越高，表示文本描述越接近；
- 根据排序返回推荐酒店列表。

## 示例输出（说明）

脚本末尾示例会对以下酒店输出推荐结果：

- `Hilton Seattle Airport & Conference Center`
- `The Bacon Mansion Bed and Breakfast`

你可以在 `hotel_rec.py` 中修改 `recommendations('酒店名')` 的输入，查看其他酒店推荐。

## 已处理的兼容性问题

- 兼容新旧版本 `scikit-learn` 的特征名接口差异：
  - 新版：`get_feature_names_out()`
  - 旧版：`get_feature_names()`
- 针对 Windows 控制台编码问题，已统一标准输出/错误输出为 UTF-8，避免 `UnicodeEncodeError`。

## 方法优缺点（简要）

**优点**
- 简单、可解释、实现成本低；
- 无需用户行为数据即可做内容推荐；
- 适合作为文本推荐基线方案（baseline）。

**局限**
- 偏词面匹配，语义理解能力有限；
- 高维稀疏特征在大规模数据下会带来存储/检索压力；
- 个性化能力有限（未融合用户行为）。

## 后续可优化方向

- 增强文本预处理（词形还原、同义词归一化）；
- 融合更多字段（如价格、地理位置、设施标签）；
- 尝试语义向量模型（如 Sentence-BERT）并与 TF-IDF 结果做融合排序。
