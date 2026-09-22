# RecSys — AGENTS.md

> 给 AI agent 的协作约定与索引。最后更新：2026-09-22。
> **项目档案不在这里**：进度与决策在 `memory.md`，管线与口径在 `architecture.md`，专题结论在 `docs/`（见下面的索引）。
> 判据：**"怎么干"（规则、环境、约定）留本文件；"干了什么、为什么这么设计"进 `memory.md` / `architecture.md` / `docs/`。**

## 项目是什么

用 Yandex **Yambda-5B** 数据集学习并搭建一个推荐系统，一步一步来。

项目的参考都在 `/Users/yukuanzou/workspace/reference/` 下（仓库外；目前有 `fun-rec`：Datawhale《深度推荐算法实践》，TensorFlow 2.13，许可 CC BY-NC-SA 4.0）——**只当读物，不要复制其代码进本仓库**。

## 索引

| 文件 | 内容 |
|---|---|
| `README.md` | 仓库门面：这是什么、三个基线复现到什么程度、怎么跑 |
| `memory.md` | 进度（已完成 / 待办）、关键决策与理由、数据与产物状态、杂项 |
| `architecture.md` | 数据流、口径（切分 / id 空间 / 评测）、目录约定、为什么这么设计 |
| `docs/dataset_notes.md` | 数据集事实：字段含义、实测统计、时间戳、显式反馈语义、已知坑（含出处） |
| `docs/benchmark_repro.md` | 官方 benchmark 在本机的落地情况 + 官方基线数字 |
| `docs/eda.md` | 切分后数据的分布 / 长尾 / 时间结构（图在 `artifacts/eda/`） |
| `docs/baselines.md` | 各 baseline 的原理与数字 |
| `scripts/` | `01` 看数据、`02` 切分、`03` EDA、`04` popularity、`05` itemknn、`06` sasrec；另有 `rec_eval.py`（评测库，各模型共用） |

## 协作方式（重要）

- **一步一步来**：一次只做一件事，做完停下确认，不要连着往下做。
- **先讲方案再动手**；不擅自扩大范围。
- 输出**朴素**：不要装饰性打印、emoji、花哨分隔线。
- **不要过度交付**：只做被要求的事。一次性的验证不要固化成脚本（曾经写过 `02_dataset_cross_checks.py`，被要求删除）。
- 交流用中文。
- 结论要有出处；不确定的事情标注"未解释"，不要编。

## 工作要求（2026-09-19 追加）

- **专注推荐系统**：本项目专注推荐系统的搭建；每一步都要能回答"为什么必须这么做"。
- **不确定就问**：不确定或模糊的地方一定要先向用户询问，避免过度交付或考虑不全。
- **最小代码量**：以最小代码量完成任务，保证代码的可解释性和逻辑连贯。
- **大任务先拆解**：遇到大规模构建任务，先思考框架和计划，把大任务拆成小任务，然后一步步执行。
- **注意长耗时任务**：时刻注意运行或思考时间过长的任务；如果时间过长，把当前的思考和想法告诉用户，由用户决定继续还是放弃。
- **进度报告模板**：用户要求报告进度时，按"任务开始状态 → 设计和规划 → 实际工作 → 达成的效果 → 自我评估"这个模板来报告。

## 环境

- 用 **uv** 管理（不是手工 venv+pip）：`uv run python ...` 即可，无需 activate。
- **Python 3.12.14**（uv 下载的独立版本，不是系统自带的 3.9.6），锁定在 `.python-version`。
- 依赖见 `pyproject.toml` / `uv.lock`（增删用 `uv add` / `uv remove`）；当前装了 pandas、pyarrow、numpy、matplotlib、scipy、torch。
- **torch 2.14.0**（与 `vendor/` 那个独立 env 同版本）：只给 `scripts/06_sasrec.py` 用。训练在 MPS 上跑，但注意力那层的**推理必须用 CPU**（MPS 上会崩，见 `docs/benchmark_repro.md`）；注意力现在是自写 block（不再是 `nn.TransformerEncoder`，原因见 `docs/baselines.md` 的缺陷一节）。
- 项目根：`/Users/yukuanzou/workspace/RecSys`，git remote = `Jormungarr/RecSys`。

## 约定

- 脚本放 `scripts/`，按序号命名（`01_`、`02_`…），朴素 `print`；专题结论放 `docs/`。**库文件**（不是可执行脚本）不编序号，如 `scripts/rec_eval.py`。
- **档案分层**：本文件只放 agent 相关（协作方式、工作要求、环境、约定、索引）；进度与决策 → `memory.md`；管线与口径 → `architecture.md`；口径的代码级唯一来源写在脚本 docstring 里。
- 代码的记忆用 `codebase-memory-mcp`（本机：`~/.local/bin/codebase-memory-mcp`，单次调用 `cli <tool>`，如 `search_code`、`query_graph`；本仓库已建索引，项目名 `Users-yukuanzou-workspace-RecSys`，`scripts/`、`docs/` 靠仓库根 `.cbmignore` 的 `!` 规则放回索引）。
- sasrec 的 checkpoint 有 **8 处守卫**（`MAX_SEQ_LEN` / `EMB` / `LAYERS` / `USE_TIME_FEATURE` / `USE_TIME_BIAS` / `POS_MODE` / `USE_DECAY` / `USE_ORGANIC`，都在 `scripts/06_sasrec.py` 顶部）：改这些常量或开关会让旧 checkpoint 装不回去，脚本会给可读提示而不是 traceback。后三个（`POS_MODE` / `USE_DECAY` / `USE_ORGANIC`）可用环境变量覆盖（如 `POS_MODE=time_t2v uv run ...`），方便一条命令里连跑多臂；`NEG_MODE` / `WEIGHT_MODE` 只改训练、不改推理，所以没有守卫。**换配置重训前先把当前最好那份 `cp` 成 `state_l*.pt`**——`state.pt` 每次训练都会被覆写。
- `data/`、`artifacts/` 一律 gitignore。
- 量纲/协议类问题**先看官方常量怎么用**，再看数据（这是踩过的坑：绕了四条判据才确认时间戳单位是秒，而官方 `GAP_SIZE=1800` ↔ "Gap: 30 minutes" 一眼就能确认）。
