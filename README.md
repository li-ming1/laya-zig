---
license: apache-2.0
library_name: transformers
pipeline_tag: text-classification
language: [multilingual, en, de, fr, es, pt, it, nl, sv, da, nb, ru, pl, tr, ar, he, fa, ur, hi, bn, ta, te, kn, ml, th, vi, id, ms, tl, ja, ko, zh, el, hu, fi, ro, sq, sl, sw, af, cy, am, hy, ka, km, my, mn, lv, is, az, jv]
tags: [laya, multilingual, mmbert, system-one, calibrated-decisions, rlcd, classification, routing, guardrails, moderation, commercial-use]
---

<p align="center">
  <img src="https://huggingface.co/convaiinnovations/laya/resolve/main/assets/logo-mark.png" alt="" width="72" />
</p>

# Laya Multilingual

Non-autoregressive **System 1 decision model** covering 100+ languages. Give it a **state**
(text, email, ticket, or JSON) and **typed questions**; it returns typed answers with
probabilities in a single forward pass. No text generation, so nothing to parse and nothing to
hallucinate.

Part of the [Laya family](https://huggingface.co/convaiinnovations/laya) — **use this checkpoint
for anything that is not English.**

| checkpoint | encoder | params | context | use it for |
|---|---|---|---|---|
| [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya) | ModernBERT-large | 421M | 512 | English |
| **`convaiinnovations/laya-multilingual`** (this repo) | mmBERT-base | 322M | 1024 | 100+ languages, ~2x faster |
| [`convaiinnovations/laya-typed-decisions`](https://huggingface.co/convaiinnovations/laya-typed-decisions) | ModernBERT-large | 421M | 1024 | the typed-decisions workflows |

## Quickstart

```bash
pip install laya
```

```python
import laya

agent = laya.load("convaiinnovations/laya-multilingual")
result = agent.predict(
    {"body": "मुझसे इनवॉइस 4411 के लिए दो बार शुल्क लिया गया। कृपया आज ही धनवापसी करें।"},
    {"department": {"type": "choice", "instructions": "Which team should handle `body`?",
                    "criteria": {"billing": "invoices, payments, refunds",
                                 "technical": "bugs and outages", "sales": "pricing"}},
     "refund_requested": {"type": "noul", "instructions": "Does the sender ask for money back?"}},
)
print(result["answers"]["department"]["choice"])      # billing
```

### Let the Router choose

```python
from laya import Router

router = Router()
router.predict({"body": "I was charged twice"}, questions)          # -> laya
router.predict({"body": "二重に請求されました"}, questions)            # -> laya-multilingual
```

Routing is per request, so a mixed workload otherwise pays a checkpoint swap on every language
change. Load the ones you expect once, up front, and every later call is just a forward pass:

```python
router = Router()
router.preload(["english", "multilingual"])      # both resident; no swap at request time
```

`router.attach("multilingual", agent)` registers an `Agent` you already built, so a process that
loaded this checkpoint directly can hand it to the router instead of loading it twice.

Routing is decided from the script of the input, **before** the forward pass — because the
model's confidence gives no warning when a checkpoint cannot read its input (see below).

> **If `laya.load()` hangs:** `transformers` probes for TensorFlow at import, and when TF is
> installed its abseil runtime can deadlock model construction. Run with `USE_TF=0`.

## Why this checkpoint exists

Measured across **all 51 MASSIVE languages**, intent classification with 20 options
(random = 0.050), both checkpoints answering byte-identical questions:

| | `laya` (English) | **`laya-multilingual`** |
|---|---|---|
| macro accuracy | 0.227 | **0.366** |
| macro ECE | 0.733 | **0.387** |
| languages clearing 3x random | 23 / 51 | **45 / 51** |

The English checkpoint does not degrade gracefully outside English — it collapses, and stays
confident while doing so. Khmer: **0.000 accuracy at 0.952 confidence**. Hebrew 0.060, Armenian
0.050 (exactly random), Bengali 0.080 — all reported with 0.89–0.96 confidence. Its mean
confidence never drops below 0.885 at any accuracy level, so confidence gating cannot catch it.

Per-language, this checkpoint turns near-random into usable: Arabic 0.110 → **0.400**, Bengali
0.080 → **0.290**, Azerbaijani 0.100 → **0.300**, Hindi 0.100 → **0.387**, Korean 0.110 →
**0.490**, Turkish 0.140 → **0.437**.

### XNLI (15 languages)

| | `laya` | **`laya-multilingual`** |
|---|---|---|
| English | **0.860** | 0.843 |
| 14 other languages | 0.521 | **0.731** |

### Speed — it is also the faster checkpoint

| questions per call | `laya` | **`laya-multilingual`** |
|---|---|---|
| 1 | 39.5 ms | **32.8 ms** |
| 10 | 158.6 ms (15.9 ms/q) | **72.3 ms (7.2 ms/q)** |
| 50 | 771 ms | **337 ms (6.8 ms/q)** |

103–332 questions/sec batched on one T4, despite a 256k vocabulary — the 768-dim / 22-layer
encoder is cheaper per token than 1024-dim / 28-layer, and the gap widens with batch size.

## Architecture

- **Backbone** mmBERT-base (307M, bidirectional, 22 layers, hidden 768, 256k vocab) + a decision
  head trained from scratch: 2 transformer layers, an option-marker scorer, and an act/escalate
  head. 322M total.
- **Option markers** every option is scored at its own `[MASK]` token, then softmaxed over that
  question's options — so the answer space is defined per request, with no retraining.
- **Budget** 1024 tokens per question, of which 256 go to the question and its options.
- Trained from scratch with RLCD: 15,987 updates, 4 epochs, ~4.97 h.

## Limits

- **Ships uncalibrated.** `temperature = [1.0, 1.0, 1.0]` with no per-option-count buckets. It is
  systematically over-confident (mean confidence 0.75–0.83 against much lower accuracy). Refitting
  one temperature per (question type, option count) on held-out data moves mean ECE
  **0.314 → 0.106**. Do this on your own data before trusting the probabilities.
- **Weaker on English** than the English checkpoint: 0.619 vs 0.684 macro across English suites.
  Route rather than replace.
- **Near chance on typed-decisions zero-shot** — 0.342, against a 0.318 random and 0.461
  majority-class baseline. Fine-tune for a specific workflow; that is where the capability comes
  from.
- **Keep `choice` questions under ~20 options.** Options share the fixed 256-token head budget, so
  a very large label space leaves only a few tokens per label and accuracy falls off sharply.
- Low-resource languages are weak, not fixed: Swahili 0.210, Tamil 0.250, Amharic 0.110.
- Ordinal `score` questions are the weakest primitive (SST-5 0.282).

## Links

- **Hub / family** https://huggingface.co/convaiinnovations/laya
- **GitHub** https://github.com/NandhaKishorM/laya · full benchmark data on the `research` branch
- **PyPI** https://pypi.org/project/laya/
- **Demo** https://huggingface.co/spaces/convaiinnovations/laya-demo

Apache 2.0 · Convai Innovations
