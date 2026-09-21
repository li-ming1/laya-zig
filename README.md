# laya-zig — Laya 决策模型的纯 Zig CPU 推理运行时

> **English TL;DR** — A from-scratch, dependency-free Zig runtime for the
> [Laya](https://github.com/NandhaKishorM/laya) System-1 decision model
> (`convaiinnovations/laya-multilingual`, 322M, mmBERT-base + decision head).
> No Python, no torch, no BLAS: it reads `model.safetensors` directly and runs the
> entire forward pass — tokenizer included — on the CPU, in one process. Ships
> with a CLI, a local browser UI, and a Snake probe that measures what a zero-shot
> decision model actually does on a task it was never trained for.

把 Laya 决策模型的推理端从 Python + torch 里搬出来，做成一个**零第三方依赖的单体 Zig 程序**：
分词、22 层 mmBERT 编码器、决策头、softmax 全部自己实现，权重直接从 `model.safetensors` 读。
附带一个本地网页界面和一个贪吃蛇探针，用来看它在**没训练过的任务**上到底会干什么。

## 它是什么

Laya 是一个 **System-1 决策模型**：一次前向、非自回归、不生成文本。你给它一个 state
（文本 / JSON / 对话）和若干「带类型的题目」，它直接返回选项概率 —— 没有东西需要 parse，
也没有东西可以幻觉。官方实现是 Python（`pip install laya`，模型与训练由
[Convai Innovations](https://github.com/NandhaKishorM/laya) 提供）。

本项目是**同一套权重的 Zig 重实现**，目标是：单二进制、无依赖、可离线、启动即用。

## 亮点

- **零依赖**：没有 torch / numpy / BLAS，`zig build` 出来就是一个 exe。
- **自带分词器**：256k 词表 BPE + Metaspace 预分词 + byte fallback + added token，
  与 HF `tokenizers` 逐 token 对齐。
- **完整模型**：RoPE、全局/滑窗交替注意力、GLU 前馈、选项 marker 打分、act/escalate 头，
  按上游参考实现逐层移植。
- **8 宽 SIMD + 多线程** GEMM：按输出列切分，每个权重行只从内存流过一次
  （这一步是每问题 ~2 s 和 ~200 ms 的差别）。
- **可验证**：三个校验工具，不是「看起来能跑」，是逐层对齐过。见「正确性验证」。

## 实测

测试机：i5-1240P（12C/16T）/ 16 GB / Windows，`-mcpu=native`，线程数自动取 8。

| 项目 | 实测 |
|---|---|
| 权重加载（614 MB f16） | **528 ms** |
| 分词器加载（256k 词表 / 580,604 merges / 249 added） | **122 ms** |
| 内置 demo（2 个问题，63 + 64 token） | forward **615–640 ms** |
| 蛇探针（单问题 166–169 token） | **618–741 ms / 步** |

纯 CPU 的代价换来零依赖。作为参照，官方实现在 T4 上单问题约 33 ms。
线程数默认 `min(核数, 8)`：实测超过 8 条线程后 matmul 变成内存延迟瓶颈，反而更慢。

## 快速开始

### 1. 拿模型

权重不在仓库里（614 MB）。用官方渠道下到仓库根目录：

```bash
huggingface-cli download convaiinnovations/laya-multilingual --local-dir .
```

实际会用到其中三样：

```
model.safetensors          614 MB f16 权重        必需
tokenizer/tokenizer.json   分词器                 必需
rl_agent_config.json       温度等配置              可选，缺省时用内置默认值
encoder/config.json        不读取——结构常量已硬编码，换 checkpoint 才需要改
```

### 2. 构建

需要 **Zig 0.17.0-dev.1737** 或同代开发版：代码用到了 `std.Io`、`std.process.Init`、
`std.array_list.Managed`，稳定版编译不过。

```bash
zig build                  # 默认 fast + 本机 CPU
zig build -Doptimize=debug
zig build check            # 只做类型检查，不产出二进制
zig build run              # 构建并直接跑内置 demo
```

### 3. 跑

```bash
zig-out/bin/laya.exe                       # 内置 demo（印地语退款邮件，两个问题）
zig-out/bin/laya.exe --json q.json         # 自己的 state + 题目
zig-out/bin/laya.exe --serve --port 8080   # 本地网页界面
```

`q.json`：

```json
{
  "state": "मुझसे इनवॉइस 4411 के लिए दो बार शुल्क लिया गया। कृपया आज ही धनवापसी करें।",
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle `body`?",
      "criteria": {
        "billing": "invoices, payments, refunds",
        "technical": "bugs and outages",
        "sales": "pricing"
      }
    },
    "refund_requested": {
      "type": "noul",
      "instructions": "Does the sender ask for money back?"
    }
  }
}
```

三种题型：`choice`（选一个）、`score`（有序打分）、`noul`（是 / 否判断）。

`state` 是字符串时按原样送入；写成对象 / 数组时会序列化成 JSON 再送入 ——
上面这个例子里，官方 Python 版传的是 `{"body": "..."}`，序列化后带 `{"body": "...", ...}`
的 JSON 壳。注意本实现输出的是**紧凑 JSON**，而 Python 的 `json.dumps` 默认分隔符是
`", "` / `": "`，会多两个空格；需要与官方逐字节一致时，请把 `state` 直接写成字符串。

## 命令行

| 参数 | 说明 |
|---|---|
| `--json FILE` | 用自己的 state + 题目 |
| `--dir DIR` | 模型目录（默认 `.`） |
| `--threads N` | 线程数（默认 `min(核数, 8)`） |
| `--serve [--port N]` | 起本地网页界面（默认 8080） |
| `--snake` | 让模型玩贪吃蛇，逐帧渲染 |
| `--snake --games N` | N 局汇总，不逐帧渲染 |
| `--snake --policy greedy\|random` | 换成基线策略做对照 |
| `--snake --prompt` | 打印喂给模型的盘面文本 |
| `--size N --seed N --max-steps N --delay MS` | 蛇的参数（默认 10 / 12345 / 400 / 0） |
| `--selftest` | SIMD 内核与 f64 对照 |
| `--dumpstats` | 打印逐层隐状态指纹，给 `tools/refcheck.py` 用 |
| `--tokcheck FILE` | 用黄金样本校验分词器 |
| `--debug` | 打印 marker 位置与 logits |

## 模型结构

权重里有什么就实现什么。下列细节全部照上游 `laya/common.py` 与
`transformers/models/modernbert/modeling_modernbert.py`：

```
输入序列   [CLS] <type> question: <instructions> [SEP]
           [MASK]opt0 [MASK]opt1 ... [SEP] <state> [SEP]
              ↑ 每个选项在自己的 [MASK] 位置上被打分

encoder    mmBERT-base：22 层 / hidden 768 / 12 头 / head_dim 64 / 256k 词表
           · RoPE，theta=160000，rotate_half 配对（第 i 维与第 i+32 维）
           · 第 0/3/6/…/21 层全局注意力，其余层滑窗 ±64；始终双向，无因果掩码
           · 第 0 层的 attn_norm 是 Identity（embeddings 已经归一化过）
           · 前馈是 GLU：Wi 输出 2304 切成两半，gelu(前半) * 后半
           · LayerNorm 无 bias；attention / mlp 均无 bias

决策头      h = encoder(x) → final_norm → + type_emb[qtype]
           → 2 层 TransformerEncoderLayer（pre-norm，ReLU 前馈 3072，双向）
           → 在每个选项的 [MASK] 位置取一行
           → scorer(LayerNorm → Linear → GELU → Linear) 得该选项分数
           → 对该题所有选项 softmax
           → act/escalate 头：h[:,0] 拼上 [top1, top1-top2, 归一化熵, k/255] 出 2 个动作概率
```

预算：`max_len = 1024`（整条序列），`head_max_len = 256`（题面 + 所有选项共享）。
选项数上限经验值是 ~20：选项越多，题面能分到的 token 越少。

## 正确性验证

前向的正确性是量出来的，不是猜的。三个命令：

```bash
# 1) SIMD 点积内核与 f64 对照（阈值 1e-4，实测 5.0e-6）
#    这个 zig 版本曾把 4 路累加器版本静默编译错（结果偏 40%），
#    所以动过 dotv / matmul 之后必须重跑。
zig-out/bin/laya.exe --selftest

# 2) 整网对照：纯 Python（只用标准库，不需要 numpy / torch）独立复算
#    22 层编码器 + final_norm + type_emb + 2 层决策头 + scorer + 最终 logits，
#    与本实现的逐层指纹比对。约 4 分钟。实测 ≈1e-5，softmax 4 位小数一致。
zig-out/bin/laya.exe --dumpstats --json q.json 2> _dump.txt
python -X utf8 tools/refcheck.py _dump.txt

# 3) 页面脚本冒烟测试：桩 DOM / fetch 跑 src/web/index.html 的脚本并点按钮（13 项检查）
node tools/web-smoke.js
```

第 2 项值得单独说：因为它，能确定「模型输出奇怪」是模型的行为而不是端口的 bug。
比如 README 里那个官方 demo，模型给出的答案是 `sales`，不是注释里写的 `billing` ——
独立参考实现给出同样的 `sales`（0.7481 / 0.7480），所以这不是移植错误。
模型卡自己也写着：零样本 typed-decisions 0.342，随机 0.318。

## 蛇探针：这个模型零样本会干什么

`--serve` 打开界面，或者跑三项对照。同一批种子（8×8，种子 1–5，各 5 局）：

| 决策者 | 均值分 |
|---|---|
| **model** | **0.00** |
| greedy（一步贪心） | 16.20 |
| random | 0.40 |

**模型低于随机基线** —— 这正是模型卡自报的水平（0.342 vs 0.318），只是更极端。
模型没在蛇上训练过，所以「它玩得不好」是这次实验的**结论**，不是故障。

更值得记的一条性质：**它可能永远不结束。** 决策者没有记忆，看到同一个盘面必然给出
同一个答案，所以状态一旦重复，这一局就**可证明**不会终止。同一批种子里，
模型 4 局在 4–5 步内撞死、1 局落进 11 步循环；贪心也有 4/5 局出现盘面重复
（它同样是确定性的）。网页界面和对照都在检测到重复时中止该局并标注出来，
而不是让它在步数上限里空转。

界面会把这件事量化：选了会立刻死掉的方向的比例、其中置信度还 > 0.5 的次数、
平均置信度、循环周期与重复次数，以及三个策略的对照表。

## 已知限制

- **没有 GPU 后端**，只做 CPU。
- **没有 mmap**：614 MB 权重全量读进内存，加激活常驻约 700 MB。
- **f32 计算**（上游在 GPU 上用 bf16/fp16）。
- **没有 batching**：一次前向只处理一个题目，多个题目串行跑。上游可以 batch。
- 未实现上游的 `temperature_by_options` 分桶（当前 checkpoint 里是空表，不影响结果）、
  `option_order`、`truncate_left`。
- 结构常量（22 层 / 768 / 1152 / 滑窗 64）按这个 checkpoint 硬编码；换 checkpoint 要改
  `src/main.zig` 顶部的常量。
- **这个模型不保证可用性**：零样本 typed-decisions 接近随机。要用在生产上请按官方建议
  在自有数据上微调并对概率做温度校准。

## 目录

```
src/main.zig          分词器 + 模型 + 前向 + CLI + 网页服务
src/snake.zig         贪吃蛇环境与基线策略
src/web/index.html    本地界面（单文件，构建时编译进二进制）
tools/refcheck.py     纯 Python 参考实现（整网逐层对照）
tools/web-smoke.js    页面脚本冒烟测试
```

## 致谢

模型、架构与训练全部来自 **Convai Innovations** 的 Laya 项目：

- https://github.com/NandhaKishorM/laya
- https://huggingface.co/convaiinnovations/laya-multilingual

本仓库只包含 Zig 侧的推理实现与实验代码，**不含模型权重**。
上游以 Apache-2.0 发布，本项目沿用同一许可证，详见 [`NOTICE`](NOTICE)。

## 许可证

[Apache License 2.0](LICENSE)。
