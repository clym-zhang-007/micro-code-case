# Unsloth-LoRA-BLEU 中文医疗微调与评估

基于 Unsloth + LoRA 的中文医疗领域 SFT 微调完整 Pipeline，涵盖数据工程、模型训练、BLEU 自动评估的全流程。

## 项目结构

```
Unsloth-LoRA-BLEU-case/
├── run_sft_pipeline.py              # 主 Pipeline（串联全部 6 个步骤）
├── medical_data_processor.py        # 医疗数据收集、清洗、格式转换
├── data_quality_report.py           # 数据质量评估报告生成
├── data_format_converter.py         # 数据格式转换（Alpaca/ShareGPT）
├── bleu_evaluation.py               # BLEU 自动评估工具（支持中英文）
├── Qwen2_5_(7B)_医疗微调.py         # Qwen2.5-7B 医疗微调完整脚本
├── Qwen3_5_医疗微调.py              # Qwen3.5-0.8B 微调（自动适配 CPU/GPU）
├── Qwen3_5_医疗微调_CPU.py          # Qwen3.5-0.8B CPU 模式微调
├── Qwen3_5_医疗微调_GPU.py          # Qwen3.5-0.8B GPU 模式微调
├── sft_quick_comparison.py          # 清洗前后对比实验（自动适配 CPU/GPU）
├── sft_quick_comparison_CPU.py      # 清洗前后对比实验（CPU 版）
├── sft_quick_comparison_GPU.py      # 清洗前后对比实验（GPU 版）
├── download_model.py                # 模型下载脚本
├── requirements.txt                 # Python 依赖
└── 【数据集】中文医疗数据/            # 原始医疗数据集
```

## 核心 Pipeline

### 一键运行全流程

```bash
python run_sft_pipeline.py
```

### 分步运行（推荐，每步可观察结果）

```bash
python run_sft_pipeline.py --step 1   # 数据收集清洗（CPU）
python run_sft_pipeline.py --step 2   # 数据质量评估（CPU）
python run_sft_pipeline.py --step 3   # 数据格式转换（CPU）
python run_sft_pipeline.py --step 4   # 模型微调（GPU/CPU）
python run_sft_pipeline.py --step 5   # BLEU 效果评估（CPU）
python run_sft_pipeline.py --step 6   # 清洗价值验证（GPU/CPU）
```

## 脚本说明

### 数据工程

| 脚本 | 功能 |
|------|------|
| `medical_data_processor.py` | 多格式医疗数据加载（Excel/CSV/JSON）、去重、去空值、去极值、格式转换 |
| `data_quality_report.py` | 生成 JSON 质量报告：文本长度分布、重复率、字段完整性、问题类型统计 |
| `data_format_converter.py` | 将清洗后数据转换为 Alpaca / ShareGPT 等微调格式 |
| `bleu_evaluation.py` | 中文/英文 BLEU 评估，支持多参考文本、逐样本分析 |

### 模型微调

| 脚本 | 模型 | 环境 | 说明 |
|------|------|------|------|
| `Qwen2_5_(7B)_医疗微调.py` | Qwen2.5-7B | GPU (4bit量化) | 7B 模型垂直领域微调完整示例 |
| `Qwen3_5_医疗微调.py` | Qwen3.5-0.8B | CPU/GPU | 自动适配 CPU/GPU 环境 |
| `Qwen3_5_医疗微调_GPU.py` | Qwen3.5-0.8B | GPU | 小模型 GPU 模式微调，适合低显存环境 |
| `Qwen3_5_医疗微调_CPU.py` | Qwen3.5-0.8B | CPU | 无 GPU 时可运行，自动减少训练步数 |

### 对比实验

| 脚本 | 说明 |
|------|------|
| `sft_quick_comparison.py` | 清洗前 vs 清洗后数据分别微调，对比 Loss 曲线和生成质量（自动适配） |
| `sft_quick_comparison_GPU.py` | 同上，GPU 版本（Unsloth 4bit 量化） |
| `sft_quick_comparison_CPU.py` | 同上，CPU 版本（训练步数减少） |

### 工具脚本

- `download_model.py` — 使用 ModelScope 下载模型到本地

## 环境安装

```bash
pip install -r requirements.txt
```

### 核心依赖

| 依赖 | 用途 |
|------|------|
| `unsloth` | 高效 LLM 微调框架（显存优化 + 加速训练） |
| `transformers` | HuggingFace 模型加载与推理 |
| `trl` | SFTTrainer（监督微调训练器） |
| `peft` | LoRA 低秩适配 |
| `torch` | PyTorch 深度学习框架 |
| `datasets` | 数据集加载与预处理 |
| `pandas` | 数据处理 |
| `Pillow` | 图像处理 |

## 快速开始

### GPU 环境

```bash
# 1. 数据清洗
python run_sft_pipeline.py --step 1

# 2. 质量评估
python run_sft_pipeline.py --step 2

# 3. 格式转换
python run_sft_pipeline.py --step 3

# 4. 模型微调
python run_sft_pipeline.py --step 4

# 5. BLEU 评估
python run_sft_pipeline.py --step 5
```

### 无 GPU（CPU 模式）

```bash
# 使用 CPU 版本的脚本，自动降级为少量训练步数
python "Qwen3_5_医疗微调_CPU.py"
```

## 注意事项

- GPU 训练建议至少 4GB 显存（0.8B 模型），7B 模型建议 16GB+
- CPU 模式训练步数自动减少至 10 步，仅用于验证流程
