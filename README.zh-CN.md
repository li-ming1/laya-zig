# laya-zig — Laya 决策模型的纯 Zig CPU 推理运行时

[English](README.md) | **简体中文**

> ### 状态：半成品
>
> 从"这个模型能不能脱离 Python 跑"到能跑通，前后大约**两个小时**。端到端能跑，
> 前向也与独立参考实现对齐过，性能也认真优化过两轮（见「实测」：笔记本 CPU 上
> 一步决策 ~170 ms，占用 ~330 MB），但请把它当作一个**半成品**：边角粗糙、功能不全，
> 也还没有在真实场景里用过。
>
> 欢迎提 issue、指正和 PR。

把 Laya 决策模型的推理端从 Python + torch 里搬出来，做成一个**零第三方依赖的单体 Zig 程序**：
分词、22 层 mmBERT 编码器、决策头、softmax 全部自己实现，权重直接从 `model.safetensors` 读。
附带一个本地网页界面和一个贪吃蛇探针，用来看它在**没训练过的任务**上到底会干什么。

## 它是什么

Laya 是一个 **System-1 决策模型**：一次前向、非自回归、不生成文本。你给它一个 state
（文本 / JSON / 对话）和若干「带类型的题目」，它直接返回选项概率 —— 没有东西需要 parse，
也没有东西可以幻觉。官方实现是 Python（`pip install laya`），模型与训练由
[Convai Innovations](https://github.com/NandhaKishorM/laya) 提供。

本项目是**同一套权重的独立 Zig 重实现**，目标是：单二进制、无依赖、可离线、映射 614 MB
权重后约三分之一秒就能开始答题。

## 亮点

- **零依赖**：没有 torch / numpy / BLAS，`zig build` 出来就是一个 exe。
- **自带分词器**：256k 词表 BPE + Metaspace 预分词 + byte fallback + added token，
  与 HF `tokenizers` 逐 token 对齐。
- **完整模型**：RoPE、全局/滑窗交替注意力、GLU 前馈、选项 marker 打分、act/escalate 头，
  按上游参考实现逐层移植。
- **8 宽 SIMD + 多线程** GEMM：按输出列切分，每个权重行只从内存流过一次。
  softmax、RoPE、GLU、norm、QKV 拆分也全部向量化并并行 —— 这才是笔记本 CPU 上
  一步决策 ~170 ms 的来处。
- **可验证**：三个校验工具，不是「看起来能跑」，是逐层对齐过。见「正确性验证」。

## 实测

测试机：i5-1240P（4 大核 + 8 小核，12C/16T）/ 16 GB / Windows，`-mcpu=native`。
「优化前」是上一个 commit（`62bc212`），同一台机器同样编译；「现在」是当前代码，
8 线程（自动）。下面每一对数字都是**交替**测的 —— 前、后、后、前 —— 因为持续加载
几分钟之后稳定频率比头几次慢 ~20%，先测一批「前」再测一批「后」量到的是温度计，
不是代码。

### 延迟

| 项目 | 优化前 | 现在 |
|---|---|---|
| 内置 demo 端到端（2 个问题，63 + 64 token） | 659–667 ms | **329–382 ms** |
| 内置 demo，只算两次前向 | 106–143 ms | 146–161 ms |
| 蛇探针一步决策（175–180 token），第 2–6 步 | 155–194 ms | 157–189 ms |
| 蛇探针**第一步** | 155–186 ms | 183–258 ms |
| 权重加载（614 MB） | 376–406 ms | **3–7 ms** |
| 分词器加载（256k 词表 / 580,604 merges / 249 added） | 137–153 ms | 137–153 ms |

这几行做的交易是真的，得说清楚：启动时不再把权重展开成一份 f32 拷贝，启动时间从
~0.4 s 变成 ~0.005 s；代价是第一次前向要替操作系统把只读映射的页从文件缓存里读进
来，大概 30~90 ms，每个进程只付一次。第一次之后的速度和优化前没有区别（在这台机器
的噪声范围内），而冷进程从启动到答完 demo 第一个问题快了将近一倍。

### 开销

| 项目 | 优化前 | 现在 |
|---|---|---|
| demo，CPU 时间 | 2906 ms | **1328–1562 ms** |
| 一局蛇（6 次决策），CPU 时间 | 16.0–16.8 s | **9.0–10.9 s** |
| 峰值工作集 | 1171–1177 MB | **322–328 MB** |
| 峰值私有内存 | 1433–1486 MB | **169–222 MB** |
| `--serve` 空转 10 s | 0 ms CPU，工作集 1167 MB | 0 ms CPU，工作集 **81 MB** |

私有内存是操作系统收不回去的那部分；其余就是权重文件的只读映射，本质上是系统文件
缓存，可随时被换出 —— 这就是空转时只剩 81 MB 的原因，也是内存吃紧时它不可能再变成
614 MB 的原因。

提速没有拿精度去换：两边打印的选择与概率逐位相同。`--selftest` 里 GEMM 相对 f64 的
最大误差 7.98e-6，`--dumpstats` 的逐层转储逐字节一致，`tools/refcheck.py` 用纯
Python 独立复算 22 层加两个头，逐层偏差 2.6e-5。

`--profile` 给出的耗时分布：matmul 91%、attention 4%、norm 1.5%、RoPE / GLU /
QKV 拆分 / serial 各 ~0.6%、residual 0.4%、embedding 0.1%。matmul 每次前向读的是
239 MB f16，不再是原来的 441 MB f32 —— 同一个核，用五分之一的内存跑出同样的速度。
前向 ~150 ms 里 matmul 之外的部分一共 ~14 ms。

线程数默认 `min(核数, 8)`，这个默认值是按 CPU 开销选的，不是按延迟：

| | 8 线程 | 12 线程 | 16 线程 |
|---|---|---|---|
| 一局蛇，墙上时间 | 1.31–1.56 s | 1.42–1.45 s | 2.24–2.31 s |
| 一局蛇，CPU | 9.0–10.9 s | 14.4 s | 28.2 s |
| demo，墙上时间 | 368–375 ms | 312–337 ms | 497–512 ms |
| demo，CPU | 1.41–1.47 s | 1.56–1.58 s | 4.42–4.45 s |

12 线程只在 demo 那 2 个问题上快 ~10%，蛇的一步决策并不更快，CPU 时间却多三到六成16 线程（每个逻辑核一条，听着最直观）既慢又是三倍 CPU：屏障要等最慢的那条 SMT
兄弟线程。

作为参照，官方实现在 T4 上单问题约 33 ms。纯 CPU 的代价换来零依赖。

## 快速开始

### 1. 克隆

```bash
git clone https://github.com/li-ming1/laya-zig
cd laya-zig
```

### 2. 拿模型

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

### 3. 构建

需要 **Zig 0.17.0-dev.1737** 或同代开发版：代码用到了 `std.Io`、`std.process.Init`、
`std.array_list.Managed`，稳定版编译不过。

```bash
zig build                  # 默认 fast + 本机 CPU
zig build -Doptimize=debug
zig build check            # 只做类型检查，不产出二进制
zig build run              # 构建并直接跑内置 demo
```

### 4. 跑

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
上面这个例子里，官方 Python 版传的是 `{"body": "..."}`，模型看到的是带 JSON 壳的文本。
注意本实现输出的是**紧凑 JSON**，而 Python 的 `json.dumps` 默认分隔符是 `", "` / `": "`，
会多两个空格；需要与官方逐字节一致时，请把 `state` 直接写成字符串。

## 命令行

| 参数 | 说明 |
|---|---|
| `--json FILE` | 用自己的 state + 题目 |
| `--dir DIR` | 模型目录（默认 `.`） |
| `--threads N` | 线程数（默认 `min(核数, 12)`） |
| `--serve [--port N]` | 起本地网页界面（默认 8080） |
| `--snake` | 让模型玩贪吃蛇，逐帧渲染 |
| `--snake --games N` | N 局汇总，不逐帧渲染 |
| `--snake --policy greedy\|random` | 换成基线策略做对照 |
| `--snake --prompt` | 打印喂给模型的盘面文本 |
| `--size N --seed N --max-steps N --delay MS` | 蛇的参数（默认 10 / 12345 / 400 / 0） |
| `--selftest` | SIMD 内核与 f64 对照 |
| `--profile` | 打印每次前向在各阶段上的耗时分布 |
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
比如上游 README 里那个官方 demo，模型给出的答案是 `sales`，不是注释里写的 `billing` ——
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
- 权重走只读内存映射，所以 `model.safetensors` 在进程存活期间一直被打开了 ——
  Windows 上这意味着服务跑着的时候不能删除或替换该文件。映射失败会退回普通读取
  （~0.4 s、~1.4 GB）。
- **算的是 f32、存的是 f16**（上游在 GPU 上用 bf16/fp16）：权重不在内存里展开成 f32，
  GEMM 核内一次宽化 4 列。
- **没有 batching**：一次前向只处理一个题目，多个题目串行跑。上游可以 batch。
- 未实现上游的 `temperature_by_options` 分桶（当前 checkpoint 里是空表，不影响结果）、
  `option_order`、`truncate_left`。
- 结构常量（22 层 / 768 / 1152 / 滑窗 64）按这个 checkpoint 硬编码；换 checkpoint 要改
  `src/main.zig` 顶部的常量。
- **这个模型本身不保证可用性**：零样本 typed-decisions 接近随机。要用在生产上请按官方建议
  在自有数据上微调并对概率做温度校准。

## 目录

```
src/main.zig          分词器 + 模型 + 前向 + CLI
src/snake.zig         贪吃蛇环境与基线策略
src/server.zig        --serve 用的极简阻塞式 HTTP/1.1 服务
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
