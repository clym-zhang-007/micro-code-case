# -*- coding: utf-8 -*-
"""
Qwen2.5-7B 中文医疗模型SFT微调
功能：使用中文医疗数据对Qwen2.5-7B进行垂直领域微调（完整实操代码）
环境：AutoDL GPU实例

核心流程：
  1. 加载预训练模型（4bit量化）
  2. 配置LoRA低秩适配器
  3. 加载并格式化医疗数据
  4. 使用Unsloth加速进行SFT训练
  5. 推理测试（验证微调效果）
  6. 保存LoRA权重并验证加载
"""

# ============================================================
# Step 1: 模型加载
# ============================================================
# 从unsloth导入FastLanguageModel，这是unsloth提供的快速模型加载工具
# 相比HuggingFace原生加载，unsloth对显存和速度做了深度优化
from unsloth import FastLanguageModel
import torch

# 最大序列长度：模型能处理的最长token数
# 2048对于医疗问答场景足够覆盖大部分"问题+回答"的组合
max_seq_length = 2048

# dtype=None 表示自动选择最佳精度（unsloth会根据GPU自动选bfloat16或float16）
dtype = None

# 4bit量化：将模型权重从16位压缩到4位，显存占用从约14GB降至约4GB
# 7B模型不量化的话需要约14GB显存，4bit量化后单张消费级显卡即可运行
load_in_4bit = True

# 加载预训练模型和分词器
# model_name路径指向已下载好的Qwen2.5-7B-Instruct模型
# 如果本地没有，可以改为 "unsloth/Qwen2.5-7B-Instruct" 从HuggingFace自动下载
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/root/autodl-tmp/models/Qwen/Qwen2___5-7B-Instruct",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)


# ============================================================
# Step 2: LoRA配置
# ============================================================
# LoRA（Low-Rank Adaptation）：不修改原始模型权重，只插入小型可训练矩阵
# 优点：显存占用少、训练快、可随时切换不同任务的LoRA适配器

# SVD矩阵分解原理：任何矩阵 W 可以分解为 W ≈ A × B
# LoRA的思想是：ΔW = A × B，其中A(shape: d×r) 和 B(shape: r×d) 是小矩阵
# r是秩(rank)，r越小参数量越少、训练越快，但表达能力可能不足

model = FastLanguageModel.get_peft_model(
    model,
    r=16,                           # LoRA秩，16是7B模型的常见选择
                                    # rank越大表达能力越强，但参数量和显存占用也越多
    # target_modules: 指定在哪些层插入LoRA适配器
    # q/k/v/o_proj 是注意力层的投影矩阵，gate/up/down_proj 是FFN层的投影矩阵
    # 覆盖注意力+前馈网络，几乎影响模型所有关键计算路径
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,                  # LoRA缩放系数，通常设为与r相同或2倍
                                    # 实际更新量为: ΔW × (alpha / r)
    lora_dropout=0,                 # LoRA层的dropout率，小数据集可以设为0
    bias="none",                    # 不训练bias参数，减少参数量
    use_gradient_checkpointing="unsloth",  # 梯度检查点：用时间换显存
                                          # "unsloth"是unsloth优化版，比普通版更快
    random_state=3407,              # 随机种子，保证实验可复现
    use_rslora=False,               # 不使用RSLoRA（标准化LoRA），普通LoRA已够用
    loftq_config=None,              # 不使用LOFTQ量化初始化
)


# ============================================================
# Step 3: 医疗数据准备
# ============================================================

import os
import pandas as pd
from datasets import Dataset

# 医疗对话提示模板（Prompt Template）
# 这是模型训练时看到的"指令-回答"拼接格式
# 训练时：模型学会根据"问题"生成对应的"回答"
# 推理时：只填入"问题"部分，让模型续写"回答"部分
medical_prompt = """你是一个专业的医疗助手。请根据患者的问题提供专业、准确的回答。

### 问题：
{}

### 回答：
{}"""

# EOS_TOKEN（End Of Sequence）：序列结束标记
# 训练时在每条数据末尾添加，告诉模型"到这里该停了"
EOS_TOKEN = tokenizer.eos_token


def read_csv_with_encoding(file_path):
    """
    尝试使用不同编码读取CSV文件
    中文数据常见编码优先级：gbk/gb2312/gb18030（简体中文常用） > utf-8
    gb18030是gbk的超集，支持更多生僻汉字
    """
    encodings = ['gbk', 'gb2312', 'gb18030', 'utf-8']
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法使用任何编码读取文件: {file_path}")


def load_medical_data(data_dir):
    """
    从6个科室的CSV目录加载医疗问答数据
    返回: HuggingFace Dataset对象

    数据格式: instruction(指令) + input(问题) => output(回答)
    """
    data = []

    # 科室目录名 -> 中文名的映射
    departments = {
        'IM_内科': '内科',
        'Surgical_外科': '外科',
        'Pediatric_儿科': '儿科',
        'Oncology_肿瘤科': '肿瘤科',
        'OAGD_妇产科': '妇产科',
        'Andriatria_男科': '男科'
    }

    for dept_dir, dept_name in departments.items():
        dept_path = os.path.join(data_dir, dept_dir)
        if not os.path.exists(dept_path):
            print(f"目录不存在: {dept_path}")
            continue

        print(f"\n处理{dept_name}数据...")

        # 遍历科室目录下的所有CSV文件
        csv_files = [f for f in os.listdir(dept_path) if f.endswith('.csv')]

        for csv_file in csv_files:
            file_path = os.path.join(dept_path, csv_file)
            print(f"正在处理文件: {csv_file}")

            try:
                df = read_csv_with_encoding(file_path)
                print(f"文件 {csv_file} 的列名: {df.columns.tolist()}")

                # 逐行提取问答对
                for _, row in df.iterrows():
                    try:
                        question = None
                        answer = None

                        # 支持多种列名格式：'question'或'ask'作为问题列
                        if 'question' in row:
                            question = str(row['question']).strip()
                        elif 'ask' in row:
                            question = str(row['ask']).strip()

                        # 支持多种列名格式：'answer'或'response'作为回答列
                        if 'answer' in row:
                            answer = str(row['answer']).strip()
                        elif 'response' in row:
                            answer = str(row['response']).strip()

                        # 过滤无效数据
                        if not question or not answer:
                            continue
                        # 长度过滤：问题和回答不超过200字
                        # 这是简单的质量控制，避免过长/过短的数据影响训练
                        if len(question) > 200 or len(answer) > 200:
                            continue

                        # 保存为Alpaca格式的三元组
                        data.append({
                            "instruction": "请回答以下医疗相关问题",
                            "input": question,
                            "output": answer
                        })

                    except Exception as e:
                        print(f"处理数据行时出错: {e}")
                        continue

            except Exception as e:
                print(f"处理文件 {csv_file} 时出错: {e}")
                continue

    if not data:
        raise ValueError("没有成功处理任何数据!")

    print(f"\n成功处理 {len(data)} 条数据")
    return Dataset.from_list(data)


def formatting_prompts_func(examples):
    """
    将Alpaca格式的三元组拼接为模型训练所需的纯文本格式
    这是SFT训练的核心：模型看到的是完整的"指令+输入+输出"文本

    输入示例: {"input": ["头晕怎么办"], "output": ["建议休息..."]}
    输出示例: {"text": ["你是一个专业的医疗助手...\n### 问题：\n头晕怎么办\n\n### 回答：\n建议休息...<EOS>"]}

    batched=True 表示examples是一个批次的数据，用zip批量处理
    """
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for input_text, output_text in zip(inputs, outputs):
        # 将问题和回答填入模板，并在末尾添加EOS标记
        text = medical_prompt.format(input_text, output_text) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}


# 加载医疗数据
# "Data_数据" 是数据目录路径（需要根据实际位置修改）
dataset = load_medical_data("Data_数据")
# 将每条数据从 {"instruction", "input", "output"} 转换为 {"text": "..."} 格式
dataset = dataset.map(formatting_prompts_func, batched=True)


# ============================================================
# Step 4: 模型训练
# ============================================================

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

# 训练参数配置
training_args = TrainingArguments(
    # 每个设备上的训练批次大小 = 2
    # 越大训练越稳定，但显存占用越多
    per_device_train_batch_size=2,

    # 梯度累积步数 = 4
    # 每4个batch才更新一次权重，等效batch_size = 2×4 = 8
    # 作用：在显存有限的情况下模拟更大的batch size
    gradient_accumulation_steps=4,

    # 预热步数 = 5
    # 训练前几步的学习率从0线性增加到目标值，避免一开始学习率过大导致训练不稳定
    warmup_steps=5,

    # 最大训练步数 = -1 表示不限制，由num_train_epochs控制总步数
    max_steps=-1,

    # 训练轮数 = 3，即整个数据集过3遍
    # 小数据集可以多一些，大数据集1-2遍即可
    num_train_epochs=3,

    # 学习率 = 2e-4（0.0002）
    # LoRA微调的常见学习率范围：1e-4 ~ 5e-4
    # 比全参数微调的学习率大（全参数通常是1e-5 ~ 5e-5）
    # 因为LoRA只更新少量参数，需要更大的学习率来加速收敛
    learning_rate=2e-4,

    # 精度选择：如果GPU不支持bfloat16则使用float16
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),

    # 每1步打印一次训练日志（loss、学习率等）
    logging_steps=1,

    # 优化器：adamw_8bit 是8bit量化的AdamW，显存占用更少
    optim="adamw_8bit",

    # 权重衰减 = 0.01，防止过拟合
    # L2正则化的变体，对大参数进行惩罚
    weight_decay=0.01,

    # 学习率调度器 = linear（线性衰减）
    # 学习率先从0增加到2e-4，然后线性衰减到接近0
    lr_scheduler_type="linear",

    # 随机种子，保证实验可复现
    seed=3407,

    # 模型输出（checkpoint）保存目录
    output_dir="outputs",

    # 关闭wandb/tensorboard等外部日志报告
    report_to="none",
)

# SFTTrainer（Supervised Fine-Tuning Trainer）
# 来自trl库（Transformer Reinforcement Learning），专门用于指令微调
trainer = SFTTrainer(
    model=model,                          # 加载了LoRA的模型
    tokenizer=tokenizer,                  # 分词器
    train_dataset=dataset,                # 训练数据集
    dataset_text_field="text",            # 数据集中文本字段的名称
    max_seq_length=max_seq_length,        # 最大序列长度
    dataset_num_proc=2,                   # 数据预处理使用2个进程
    packing=False,                        # 不打包多条数据为一条
                                          # True时会把多条短数据拼成max_seq_length，提高训练效率
                                          # False时每条数据独立，更简单直观
    args=training_args,                   # 训练参数
)

# 打印GPU信息
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# 开始训练
trainer_stats = trainer.train()

# 打印训练后的显存使用统计
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


# ============================================================
# Step 5: 模型推理
# ============================================================

def generate_medical_response(question):
    """
    使用微调后的模型生成医疗回答

    流程：
    1. 切换到推理模式（for_inference）
    2. 将问题填入模板并分词
    3. 模型自回归生成回答（逐个token生成）
    """
    # 将模型切换到推理模式（关闭dropout等训练时特有的行为）
    FastLanguageModel.for_inference(model)

    # 将问题填入模板（回答部分留空），进行分词
    inputs = tokenizer(
        [medical_prompt.format(question, "")],
        return_tensors="pt"
    ).to("cuda")

    # TextStreamer：流式输出，每生成一个token就打印出来
    # 类似ChatGPT的打字效果，可以实时看到生成过程
    from transformers import TextStreamer
    text_streamer = TextStreamer(tokenizer)

    # 生成回答
    _ = model.generate(
        **inputs,                              # 输入token
        streamer=text_streamer,                # 流式输出
        max_new_tokens=256,                    # 最多生成256个新token
        temperature=0.7,                       # 温度参数：越低越确定，越高越随机
                                               # 0.7是中等创造性，医疗场景偏保守
        top_p=0.9,                             # 核采样：只从累积概率90%的词中选择
        repetition_penalty=1.1                 # 重复惩罚：>1表示降低重复生成相同内容的概率
    )


# 测试问题列表
test_questions = [
    "我最近总是感觉头晕，应该怎么办？",    # 一般症状咨询
    "感冒发烧应该吃什么药？",              # 用药建议
    "高血压患者需要注意什么？"             # 健康管理
]

# 逐一测试
for question in test_questions:
    print("\n" + "=" * 50)
    print(f"问题：{question}")
    print("回答：")
    generate_medical_response(question)


# ============================================================
# Step 6: 模型保存与加载
# ============================================================

# 保存LoRA微调权重（只保存了低秩适配器，不保存整个模型）
# LoRA权重通常只有几十MB，远小于完整模型的几GB
model.save_pretrained("lora_model_medical")
tokenizer.save_pretrained("lora_model_medical")

# 验证：加载保存的LoRA权重，重新进行推理
# 实际使用中，可以只在训练时运行一次，后续直接用加载保存的权重
if True:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="lora_model_medical",      # 加载之前保存的LoRA权重
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)

# 再次测试，验证加载后的模型能正常工作
question = "我最近总是感觉头晕，应该怎么办？"
generate_medical_response(question)
