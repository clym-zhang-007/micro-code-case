#!/usr/bin/env python
# coding: utf-8

import os
# 必须在导入 pipeline 之前设置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/root/autodl-tmp/models"  # 同时控制 model 和 tokenizer 缓存

from transformers import AutoModel, AutoModelForSequenceClassification

# 1. 纯净版模型 -- 基座模型，只做了预训练，没有微调与强化学习 
# 如果传入的是带头的模型，也会只加载基础模型
base_model = AutoModel.from_pretrained("bert-base-chinese")
base_model
# 输出形状：(Batch_Size, Sequence_Length, Hidden_Size) -> (2, 10, 768)


# 2. 带分类头的模型
# 由于 bert-base-chinese 本身没有分类头，所以会随机初始化一个全新的分类头（维度为 768 → 2）
# 这也意味着：需要先做微调训练，更新分类头的权重，才能用于实际的分类任务。直接用这个模型做推理，输出的结果是随机的。
cls_model = AutoModelForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=2)
# 输出形状：(Batch_Size, Num_Labels) -> (2, 2)
cls_model

