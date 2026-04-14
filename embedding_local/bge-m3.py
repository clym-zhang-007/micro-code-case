#!/usr/bin/env python
# coding: utf-8

#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('BAAI/bge-m3', cache_dir='/root/autodl-tmp/models')


from FlagEmbedding import BGEM3FlagModel
# Setting use_fp16 to True speeds up computation with a slight performance degradation
# 用 FP16（半精度） 来做编码/推理 -- 通常会 更快、更省显存，代价是可能有轻微精度下降（相似度数值可能会有很小差异
# 如果你在 CPU 上跑或遇到精度/兼容问题，可以把它改成 use_fp16=False
model = BGEM3FlagModel('/root/autodl-tmp/models/BAAI/bge-m3', use_fp16=True) 

sentences_1 = ["What is BGE M3?", "Defination of BM25"]
sentences_2 = ["BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.", 
               "BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document"]

embeddings_1 = model.encode(sentences_1, 
                            # batch_size：每个 batch 同时送入模型编码的句子数量。
                            # - 越大通常吞吐更高，但会占用更多显存/内存；显存不足会 OOM。
                            batch_size=8, 
                            # max_length：单条输入允许的最大 token/长度上限（超出会被截断）。
                            # - 设置得越大越“能装长文本”，但计算更慢、显存占用更高。
                            # - 若文本不长，建议调小以提升编码速度。
                            max_length=1024, 
                            )['dense_vecs']
embeddings_2 = model.encode(sentences_2, batch_size=12, max_length=8192)['dense_vecs']
similarity = embeddings_1 @ embeddings_2.T
print(similarity)
# [[0.6265, 0.3477], [0.3499, 0.678 ]]

