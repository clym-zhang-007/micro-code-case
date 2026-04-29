# HuggingFace 模型使用示例

基于 HuggingFace Transformers 生态的中文 NLP 模型使用示例，涵盖模型加载、微调训练、推理等核心场景。

## 环境安装

```bash
pip install -r requirements.txt
```

## 文件说明

| 文件 | 说明 |
|---|---|
| `model.py` | 展示基座模型与分类头模型的区别（`AutoModel` vs `AutoModelForSequenceClassification`） |
| `spam-classifier-sft.py` | 完整的垃圾邮件分类微调流程：数据准备 → Tokenize → 训练 → 推理 |

## 核心概念

### 模型加载方式

| AutoModel 类 | 加的什么头 | 头是否预训练好 | 用途 |
|---|---|---|---|
| `AutoModel` | 不加头 | — | 提取隐藏层特征 |
| `AutoModelForCausalLM` | LM Head | 已训练好 | 文本生成（GPT、LLaMA） |
| `AutoModelForSequenceClassification` | 分类头 | 随机初始化 | 分类任务，需微调 |
| `AutoModelForMaskedLM` | MLM Head | 已训练好 | 完形填空（BERT） |

### Tokenizer

`AutoTokenizer.from_pretrained()` 首次调用会自动从 HuggingFace Hub 下载 tokenizer 文件并缓存到本地，无需手动下载。

### 推理注意事项

训练完成后应重新加载保存的模型进行推理，并调用 `model.eval()` 关闭 dropout：

```python
model = AutoModelForSequenceClassification.from_pretrained("/path/to/saved/model")
model.eval()
```
