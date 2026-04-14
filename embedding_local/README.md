# Embedding_BGE-M3 示例项目

本目录包含 3 个 embedding 示例脚本，演示：

- 使用 `BAAI/bge-m3` 生成向量并计算相似度
- 使用 `iic/gte_Qwen2-1.5B-instruct` 的两种调用方式（高层封装 / 底层流程）

## 文件说明

```text
embedding_BGE-M3/
├─ bge-m3.py
├─ gte-qwen2-1.5b-direction.py
├─ gte-qwen2-1.5b-workflow.py
└─ requirements.txt
```

## 三个脚本的区别

- `bge-m3.py`
  - 使用 `FlagEmbedding` 的 `BGEM3FlagModel`
  - 更贴合 BGE-M3 官方能力入口，使用简单

- `gte-qwen2-1.5b-direction.py`
  - 使用 `SentenceTransformer` 高层接口
  - 代码更短，适合快速上手

- `gte-qwen2-1.5b-workflow.py`
  - 使用 `AutoTokenizer + AutoModel` 底层方式
  - 手动池化、归一化、相似度计算，可控性更强

## 推荐环境（重要）

建议使用**独立 conda 环境**运行，避免与其它项目依赖冲突：

```bash
conda create -n bge_m3 python=3.11 -y
conda activate bge_m3
python -m pip install --upgrade pip
python -m pip install -r embedding_BGE-M3/requirements.txt
```

## 运行方式

在仓库根目录运行：

```bash
python embedding_BGE-M3/bge-m3.py
python embedding_BGE-M3/gte-qwen2-1.5b-direction.py
python embedding_BGE-M3/gte-qwen2-1.5b-workflow.py
```

## 常见问题

### 1) `ModuleNotFoundError: No module named 'modelscope'`

说明当前 Python 环境未安装依赖，或没有切到 `bge_m3` 环境。  
先执行：

```bash
python -m pip install -r embedding_BGE-M3/requirements.txt
```

并确认 `python` 来自 `bge_m3` 环境。

### 2) `RuntimeError: operator torchvision::nms does not exist`

这是 `torch/torchvision` 版本不匹配导致的典型报错。  
建议不要在大杂烩 base 环境里运行，改用独立 `bge_m3` 环境。

### 3) `conda activate bge_m3` 看起来执行了但不生效

PowerShell 需要先初始化 conda hook：

```bash
conda init powershell
```

然后关闭并重开终端，再 `conda activate bge_m3`。

## 说明

脚本中的模型下载目录（如 `/root/autodl-tmp/models`）是示例路径。  
如果你在 Windows 本地运行，请按实际机器路径修改为可写目录。
