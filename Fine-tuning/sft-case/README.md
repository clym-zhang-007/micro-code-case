# LLM 模型蒸馏与微调实操

基于 Unsloth 框架，对 Qwen2.5 系列模型进行 SFT 微调、GRPO 强化学习训练及视觉多模态微调的完整实践。

## 项目结构

```
sft-case/
├── Qwen2_5_(7B)_Alpaca.py          # Qwen2.5-7B SFT 微调（Alpaca 数据集）
├── Qwen2_5_(7B)_R1_GRPO.py         # Qwen2.5-7B GRPO 强化学习训练（R1 推理能力）
├── qwen_vl_car_sft_train.py        # Qwen2.5-VL-3B 视觉模型微调（汽车里程表识别）
├── model_comparison_eval.py        # 微调/蒸馏效果对比评估工具
├── model_comparison_eval.py        # 模型对比评估
├── requirements.txt                # Python 依赖
├── qwen-vl-train.xlsx              # VL 微调训练数据
├── images/                         # 示例图片
├── 【数据集】alpaca-cleaned/       # Alpaca 指令微调数据集
└── 【数据集】gsm8k/                # GSM8K 数学推理数据集
```

## 训练脚本说明

### 1. Qwen2.5-7B SFT 微调 (`Qwen2_5_(7B)_Alpaca.py`)

- **框架**: Unsloth + LoRA
- **数据集**: Alpaca cleaned（52K 指令数据）
- **量化**: 4bit 量化加载，减少显存占用
- **LoRA 配置**: rank=16, alpha=16, dropout=0.1
- **运行环境**: AutoDL GPU（建议 A100/3090）
- **核心特性**: 序列长度 2048，gradient checkpointing，warmup + cosine lr scheduler

### 2. Qwen2.5-7B GRPO 强化学习 (`Qwen2_5_(7B)_R1_GRPO.py`)

- **框架**: Unsloth + GRPO（Group Relative Policy Optimization）
- **数据集**: GSM8K（数学推理题）
- **量化**: 4bit 量化 + vLLM 快速推理
- **LoRA 配置**: rank=32，更大的 rank 以获得更强的推理能力
- **运行环境**: AutoDL A100/A800（40GB+ 显存）
- **核心特性**: 强化学习奖励机制，生成推理链（reasoning trace），多输出采样

### 3. Qwen2.5-VL-3B 视觉微调 (`qwen_vl_car_sft_train.py`)

- **框架**: Unsloth FastVisionModel + LoRA
- **任务**: 汽车保险承保 — 车辆里程表识别
- **数据格式**: Excel 标注数据 + 车辆仪表盘照片
- **LoRA 配置**: 同时微调视觉层、语言层、Attention 和 MLP 模块
- **核心特性**: 多模态指令微调，自定义数据加载 pipeline

### 4. 模型对比评估 (`model_comparison_eval.py`)

- **功能**: 对比基座模型 vs SFT 微调模型 vs GRPO 模型的输出质量
- **评估维度**:
  - 格式遵循能力（XML 标签完整性）
  - 答案准确性
  - 推理过程完整性
  - 响应长度与冗余度

## 环境安装

```bash
pip install -r requirements.txt
```

### 核心依赖

| 依赖 | 用途 |
|------|------|
| `unsloth` | 高效 LLM 微调框架 |
| `transformers` | HuggingFace 模型库 |
| `trl` | SFTTrainer / GRPOTrainer |
| `vllm` | 高性能推理引擎（GRPO 需要） |
| `datasets` | 数据处理 |
| `modelscope` | 模型下载 |
| `torch` | 深度学习框架 |
| `Pillow` | 图像处理（VL 微调需要） |

## 快速开始

### SFT 微调

```bash
python "Qwen2_5_(7B)_Alpaca.py"
```

### GRPO 强化学习训练

```bash
python "Qwen2_5_(7B)_R1_GRPO.py"
```

### 视觉模型微调

```bash
python "qwen_vl_car_sft_train.py"
```

### 模型评估对比

```bash
python "model_comparison_eval.py"
```

## 模型下载（AutoDL 环境）

```python
from modelscope import snapshot_download

# Qwen2.5-7B-Instruct
snapshot_download('Qwen/Qwen2.5-7B-Instruct', cache_dir='/root/autodl-tmp/models')

# Qwen2.5-VL-3B-Instruct
snapshot_download('Qwen/Qwen2.5-VL-3B-Instruct', cache_dir='/root/autodl-tmp/models')
```

## 注意事项

- 建议在 AutoDL GPU 环境运行，不同显卡对 dtype 的支持不同（T4 用 Float16，Ampere+ 用 Bfloat16）
- GRPO 训练显存需求较高，建议 40GB+
- VL 微调需要准备车辆仪表盘照片及对应的 Excel 标注数据
