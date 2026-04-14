# Query Rewriter（查询改写与联网搜索识别）

本目录提供一组查询改写脚本，面向中文问答/RAG 场景，主要用于：

- 识别用户问题类型（上下文依赖、对比、模糊指代、多意图、反问）
- 将原始 Query 改写为更适合检索的表达
- 判断是否需要联网搜索，并生成搜索策略

## 目录结构

```text
query_rewriter/
├─ intent-detection.py
├─ need-web-search.py
├─ requirements.txt
└─ .env (本地配置，不建议提交)
```

## 脚本说明

- `intent-detection.py`
  - 多类型 Query 改写示例（上下文依赖、对比、指代、多意图、反问）
  - 提供自动识别 + 自动改写流程

- `need-web-search.py`
  - 判断查询是否需要联网搜索
  - 若需要，生成改写后的搜索查询与搜索策略
  - 内置 JSON 解析容错（支持 ```json 包裹文本）

## 依赖安装

在仓库根目录执行：

```bash
pip install -r query_rewriter/requirements.txt
```

## 环境变量

在 `query_rewriter/.env` 中配置（示例）：

```env
# 二选一即可
DASHSCOPE_API_KEY=
OPENAI_API_KEY=

# OpenAI 兼容网关（默认 Coding）
OPENAI_BASE_URL=https://coding.dashscope.aliyuncs.com/v1

# 默认模型
CHAT_MODEL=kimi-k2.5

# 是否开启思考模式（建议 false 以提升速度）
ENABLE_THINKING=false

# 输出 token 上限（调小可提速）
MAX_COMPLETION_TOKENS=512
```

## 运行方式

```bash
python query_rewriter/intent-detection.py
python query_rewriter/need-web-search.py
```

## 常见问题

- **401 invalid_api_key**
  - Key 无效/过期，或与 `OPENAI_BASE_URL` 不匹配
  - 更新 `.env` 中 Key，并确认网关地址与 Key 来源一致

- **返回慢**
  - 关闭思考模式：`ENABLE_THINKING=false`
  - 减少输出长度：调小 `MAX_COMPLETION_TOKENS`
  - 选择更快模型（按网关支持列表）

- **JSON 解析失败导致误判**
  - `need-web-search.py` 已加代码块剥离与兜底提取逻辑
  - 若仍异常，建议在 prompt 中进一步约束“仅输出 JSON 对象”
