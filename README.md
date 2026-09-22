# laya-zig — a pure-Zig CPU runtime for the Laya decision model

**English** | [简体中文](README.zh-CN.md)

> ### Status: work in progress
>
> This is about **two hours of work**, from "can this model run outside Python?"
> to a working port. It runs end to end and the forward pass is checked against an
> independent reference implementation, and it has had one serious performance
> pass (see *Measured* — a Snake decision costs ~200 ms on a laptop CPU), but
> treat it as a **half-finished project**: expect rough edges and missing
> features. It has not been used in anything real.
>
> Issues, corrections and pull requests are welcome.

A from-scratch Zig runtime for the [Laya](https://github.com/NandhaKishorM/laya)
System-1 decision model (`convaiinnovations/laya-multilingual`, 322M). No Python,
no torch, no BLAS: it reads `model.safetensors` directly and runs the whole forward
pass — tokenizer included — on the CPU, in a single process.

It ships with a CLI, a local browser UI, and a Snake probe that measures what a
zero-shot decision model actually does on a task it was never trained for.

## What this is

Laya is a **System-1 decision model**: one forward pass, non-autoregressive, no text
generation. You give it a state (text, JSON, or a conversation) and typed questions,
and it returns option probabilities directly — nothing to parse, nothing to
hallucinate. The reference implementation is Python (`pip install laya`); the model
and its training come from
[Convai Innovations](https://github.com/NandhaKishorM/laya).

This repository is an independent **Zig re-implementation of the same weights**,
aiming at a single dependency-free binary that reads 614 MB of weights and starts
answering in about half a second, offline.

## Highlights

- **No dependencies** — no torch, numpy or BLAS. `zig build` produces one exe.
- **Tokenizer included** — 256k BPE with Metaspace pre-tokenization, byte fallback
  and added tokens, matching HF `tokenizers` token for token.
- **The whole model** — RoPE, alternating global/sliding attention, GLU
  feed-forward, option-marker scoring, act/escalate head, ported layer by layer
  from the upstream reference implementation.
- **8-wide SIMD, multi-threaded GEMM**, split over output columns so each weight row
  is streamed from memory exactly once. Every other stage — softmax, RoPE, GLU,
  the norms, the QKV split — is vectorized and threaded too, which is what brings
  a Snake decision down to ~200 ms on a laptop CPU.
- **Verifiable** — three checks, listed below. Not "it seems to run": the forward
  pass is aligned layer by layer.

## Measured

i5-1240P (4 P-cores + 8 E-cores, 12C/16T), 16 GB, Windows, `-mcpu=native`,
12 threads (auto). Each column is a median over repeated runs, and the width of
the range *is* the machine: sustained clocks after a few minutes of load are ~20 %
slower than the first runs, so before/after were measured in the same state —
cooled down first, then the same pair again hot. "Before" is the previous commit,
built the same way on the same box.

| | before | now | speedup |
|---|---|---|---|
| built-in demo, 2 questions (63 + 64 tokens) | 844–1017 ms | **124–143 ms** | ~7× |
| Snake, one decision (175–180 tokens) | 1909–2160 ms | **202–224 ms** | ~9.5× |
| weights load (614 MB f16 → f32) | 593–708 ms | **376–406 ms** | ~1.7× |
| tokenizer load (256k vocab / 580,604 merges / 249 added) | 136–151 ms | 137–153 ms | — |
| resident set | 1177 MB | 1177 MB | — |

The Snake row is what decides whether the browser UI feels live: at ~200 ms per
move the game gets 5 decisions a second. That median is taken over a 6-move game;
the fastest single move was 901 ms before and 174 ms now, and the old code got
*worse* as the game went on (spawn-per-call plus thermal drift) while the new one
holds roughly flat.

Neither column trades accuracy for speed — both print bit-identical probabilities
and choices. `--selftest` puts the worst GEMM relative error at 7.98e-6 against
f64, and `tools/refcheck.py` recomputes all 22 layers plus both heads in
independent Python and agrees to 2.6e-5.

`--profile` says where that time goes: matmul 84 %, attention 6 %, norm 3 %,
residual 3 %, RoPE 1 %, GLU 1 %, QKV split 1 %. The matmul streams 441 MB of f32
weights per forward at ~250 GFLOP/s; the plumbing under it is now single-digit
milliseconds total, so what is left is in the GEMM kernel itself — or in keeping
the weights as f16 and converting inside the kernel, which would halve both the
streaming and the 1177 MB resident set, at a cost this chip may not pay back.

Thread count defaults to `min(cores, 12)`. Measured on this CPU for the demo:
253 ms at 4 threads, 170 at 8, 153 at 12, **640 at 16**. Past one thread per
physical core, more threads is not more parallel — it is oversubscribed SMT, and
every barrier waits for the slowest straggler.

For reference, the upstream implementation needs ~33 ms per question on a T4.
This is what a plain CPU costs with zero dependencies.

## Getting started

### 1. Clone

```bash
git clone https://github.com/li-ming1/laya-zig
cd laya-zig
```

### 2. Get the model

The weights are not in this repository (614 MB). Download them from the official
Hub repo into the project root:

```bash
huggingface-cli download convaiinnovations/laya-multilingual --local-dir .
```

Only three of those files are used:

```
model.safetensors          614 MB f16 weights      required
tokenizer/tokenizer.json   tokenizer                required
rl_agent_config.json       temperature etc.         optional, defaults are built in
encoder/config.json        NOT read — architecture constants are hardcoded
```

### 3. Build

Requires **Zig 0.17.0-dev.1737** or an equivalent dev build: the code uses
`std.Io`, `std.process.Init` and `std.array_list.Managed`, so a stable release
will not compile it.

```bash
zig build                  # fast + host CPU by default
zig build -Doptimize=debug
zig build check            # type-check only, no binary
zig build run              # build and run the built-in demo
```

### 4. Run

```bash
zig-out/bin/laya.exe                       # built-in demo (Hindi refund email, 2 questions)
zig-out/bin/laya.exe --json q.json         # your own state and questions
zig-out/bin/laya.exe --serve --port 8080   # local browser UI
```

`q.json`:

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

Three question types: `choice` (pick one), `score` (ordinal rating), `noul`
(yes/no).

A string `state` is passed through as-is. An object or array `state` is serialized
to JSON first — in the example above the upstream Python version passes
`{"body": "..."}` and the model sees the JSON wrapper. Note that this
implementation emits **compact JSON**, while Python's `json.dumps` uses `", "` and
`": "` separators, so two spaces differ; write the state as a string if you need
byte-identical input to upstream.

## CLI

| flag | what it does |
|---|---|
| `--json FILE` | use your own state and questions |
| `--dir DIR` | model directory (default `.`) |
| `--threads N` | thread count (default `min(cores, 12)`) |
| `--serve [--port N]` | local browser UI (default 8080) |
| `--snake` | let the model play Snake, one frame per move |
| `--snake --games N` | N games, summary only |
| `--snake --policy greedy\|random` | swap in a baseline for comparison |
| `--snake --prompt` | print the board text handed to the model |
| `--size N --seed N --max-steps N --delay MS` | Snake options (default 10 / 12345 / 400 / 0) |
| `--selftest` | SIMD kernels vs f64 |
| `--profile` | print where each forward spends its time, by stage |
| `--dumpstats` | per-layer hidden-state fingerprints, for `tools/refcheck.py` |
| `--tokcheck FILE` | validate the tokenizer against golden samples |
| `--debug` | print marker positions and logits |

## Architecture

Everything in the weights is implemented. The details below follow upstream
`laya/common.py` and `transformers/models/modernbert/modeling_modernbert.py`:

```
sequence   [CLS] <type> question: <instructions> [SEP]
           [MASK]opt0 [MASK]opt1 ... [SEP] <state> [SEP]
              ^ each option is scored at its own [MASK] position

encoder    mmBERT-base: 22 layers / hidden 768 / 12 heads / head_dim 64 / 256k vocab
           - RoPE, theta=160000, rotate_half pairing (dim i with dim i+32)
           - layers 0/3/6/.../21 are global attention, the rest sliding +-64;
             always bidirectional, never causal
           - layer 0's attn_norm is Identity (the embeddings are already normalized)
           - feed-forward is a GLU: Wi outputs 2304, split in half, gelu(first) * second
           - LayerNorm without bias; no attention or MLP bias anywhere

head       h = encoder(x) -> final_norm -> + type_emb[qtype]
           -> 2 TransformerEncoderLayer (pre-norm, ReLU feed-forward 3072, bidirectional)
           -> gather the row at each option's [MASK] position
           -> scorer(LayerNorm -> Linear -> GELU -> Linear) gives that option's score
           -> softmax over that question's options
           -> act/escalate head: h[:,0] plus [top1, top1-top2, normalized entropy, k/255]
```

Budgets: `max_len = 1024` for the whole sequence, `head_max_len = 256` shared by the
instructions and all options. Keep `choice` questions under ~20 options: the more
options, the fewer tokens each one gets.

## Verification

Correctness is measured, not assumed. Three commands:

```bash
# 1) SIMD dot-product kernels vs f64 (threshold 1e-4, measured 5.0e-6).
#    This Zig build once silently miscompiled the 4-accumulator version of this
#    kernel (results ~40% off), so re-run this after touching dotv/matmul.
zig-out/bin/laya.exe --selftest

# 2) Whole-network check: a pure-Python reference (stdlib only, no numpy, no torch)
#    recomputes 22 encoder layers + final_norm + type_emb + 2 head layers + scorer +
#    final logits and diffs them against this implementation's per-layer
#    fingerprints. ~4 minutes. Measured ~1e-5, softmax identical to 4 decimals.
zig-out/bin/laya.exe --dumpstats --json q.json 2> _dump.txt
python -X utf8 tools/refcheck.py _dump.txt

# 3) UI smoke test: runs src/web/index.html's script against a stubbed DOM/fetch
#    and drives its buttons (13 checks).
node tools/web-smoke.js
```

Check 2 is worth calling out, because it is what separates "the model behaves
oddly" from "the port is broken". The official demo in the upstream README, for
instance, prints `sales` rather than the `billing` its comment claims — and the
independent reference agrees (`0.7481` vs `0.7480`), so that is the model, not the
port. The model card says as much: 0.342 on zero-shot typed decisions against
0.318 random.

## The Snake probe: what this model does zero-shot

Open the UI with `--serve`, or run the three-way comparison. Same seeds each time
(8x8 board, seeds 1–5, 5 games per policy):

| policy | mean score |
|---|---|
| **model** | **0.00** |
| greedy (one-step) | 16.20 |
| random | 0.40 |

**The model scores below the random baseline.** That matches the model card's own
number (0.342 vs 0.318), just more extreme. The model was never trained on Snake,
so "it plays badly" is the *result of this experiment*, not a malfunction.

There is a sharper property worth remembering: **it may never finish.** A policy
with no memory that sees the same board twice will answer the same way twice, so
once a state repeats, the episode provably cannot terminate. On those same seeds
the model died within 4–5 moves in four games and fell into an 11-step loop in the
fifth; greedy also repeated a board in 4 of 5 games (it is deterministic too). The
UI and the comparison both stop a game when a board repeats and say so, instead of
spinning until a step cap.

The UI quantifies all of it: what fraction of moves picked a direction that ends
the game immediately, how many of those were made with confidence > 0.5, mean
confidence, cycle length and repeat count, and the three-way comparison table.

## Limitations

- **CPU only**, no GPU backend.
- **No mmap**: the 614 MB of weights are read into memory, ~700 MB resident with
  activations.
- **f32 math** (upstream uses bf16/fp16 on GPU).
- **No batching**: one question per forward pass, several questions run
  sequentially. Upstream batches them.
- Upstream's `temperature_by_options` buckets are not implemented (the table is
  empty in this checkpoint, so results are unaffected), nor `option_order` or
  `truncate_left`.
- Architecture constants (22 layers / 768 / 1152 / window 64) are hardcoded for
  this checkpoint; another checkpoint means editing the constants at the top of
  `src/main.zig`.
- **The model itself is not production-ready**: zero-shot typed decisions are close
  to random. Upstream's advice applies — fine-tune on your own data and recalibrate
  the probabilities before trusting them.

## Files

```
src/main.zig          tokenizer + model + forward pass + CLI
src/snake.zig         Snake environment and baseline policies
src/server.zig        a minimal blocking HTTP/1.1 server for --serve
src/web/index.html    local UI (one file, embedded into the binary at build time)
tools/refcheck.py     pure-Python reference implementation (whole-network diff)
tools/web-smoke.js    UI script smoke test
```

## Credits

The model, its architecture and its training are all from Convai Innovations'
Laya project:

- https://github.com/NandhaKishorM/laya
- https://huggingface.co/convaiinnovations/laya-multilingual

This repository contains only the Zig inference implementation and experiment
code, and **no model weights**. Upstream is Apache-2.0 licensed; this project uses
the same license — see [`NOTICE`](NOTICE).

## License

[Apache License 2.0](LICENSE).
