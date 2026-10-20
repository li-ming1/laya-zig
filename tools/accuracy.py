#!/usr/bin/env python3
"""Accuracy probe for the decision model, run through the CLI.

Laya is a zero-shot *typed decision* model: upstream's own model card puts it
near chance (0.342 vs 0.318 random) on tasks it was not tuned for. This script
exists to measure that honestly on this port, and to catch behaviour changes:
every case states its answer in plain words, so a wrong answer is the model's,
not the harness's.

Usage:
    python -X utf8 tools/accuracy.py [rounds] [--plain] [--json]

Cases run in `rounds` passes (default 3) so that instability is visible; the
model is deterministic (argmax over one forward pass), so any case that flips
between rounds is a bug worth knowing about.

The default state form is `{"body": "..."}`, which is what upstream's reference
implementation passes; `--plain` sends the bare string instead. The wrapper is
worth ~4 points here (87% vs 83%): it moves the ordinal scores, not the
yes/no answers.
"""

import json
import os
import re
import subprocess
import sys
import tempfile

EXE = os.path.join("zig-out", "bin", "laya.exe")
TMP = os.path.join(".tmp", "accuracy")

# ---------------------------------------------------------------- case list
# kind: "noul" (answer true/false), "choice" (pick a key), "score" (ordinal
# 0..4, judged only as low <= 1.5 / high >= 2.5 so the criterion stays obvious).
CASES = [
    # ---- noul, Chinese: the request is stated or clearly absent
    ("zh-refund-explicit", "客户要求把订单金额全额退回。", "noul",
     "用户是否要求退款？", "true", True),
    ("zh-refund-absent", "用户只询问了门店的营业时间，没有提出其他诉求。", "noul",
     "用户是否要求退款？", "true", False),
    ("zh-phone-blackline", "用户昨天购买了一台手机，今天发现屏幕出现黑线，要求退货退款。", "noul",
     "用户是否要求退款？", "true", True),
    ("zh-thanks", "感谢你们的快速处理，问题已经解决了！", "noul",
     "用户是否表达不满？", "true", False),
    ("zh-delivered", "订单已按时送达，客户发来感谢信。", "noul",
     "订单是否出现延迟？", "true", False),
    ("zh-compensation-absent", "用户投诉快递员态度恶劣，而且包裹外箱破损。", "noul",
     "用户是否明确要求赔偿？", "true", False),
    ("zh-delay-claimed", "用户说包裹比预计时间晚了五天，要求解释。", "noul",
     "订单是否出现延迟？", "true", True),
    # ---- noul, English
    ("en-cancel-explicit", "The user asks to cancel the subscription immediately.", "noul",
     "Does the user ask to cancel?", "true", True),
    ("en-complaint-absent", "I just wanted to say the new update is great.", "noul",
     "Is the customer complaining?", "true", False),
    ("en-money-back", "The invoice was charged twice. I want my money back.", "noul",
     "Does the customer ask for money back?", "true", True),
    ("en-defect", "The screen cracked after one day of normal use.", "noul",
     "Is this a product defect report?", "true", True),
    ("en-invoice-only", "Please send the invoice for order 4411.", "noul",
     "Is the customer asking for a refund?", "true", False),
    ("en-late-crushed", "The package arrived three weeks late and the box was crushed.", "noul",
     "Is the customer complaining?", "true", True),
    ("en-bug", "The app crashes every time I open the settings screen after the update.", "noul",
     "Does the customer report a bug?", "true", True),
    # ---- noul, Hindi (the language of upstream's own example)
    ("hi-refund", "इनवॉइस 4411 के लिए मुझसे दो बार शुल्क लिया गया। कृपया धनवापसी करें।", "noul",
     "Does the sender ask for money back?", "true", True),
    # ---- choice: the body names its own topic
    ("zh-payment-spin", "付款时页面一直转圈，钱被扣了但订单没有生成。", "choice",
     "Which team should handle `body`?",
     {"billing": "invoices, payments, refunds, money that moved",
      "technical": "bugs and outages, pages that do not load",
      "sales": "pricing, discounts, plans"}, "billing"),
    ("en-http-500", "The app returns a 500 error on the checkout page after the update.", "choice",
     "Which team should handle `body`?",
     {"billing": "invoices, payments, refunds",
      "technical": "bugs and outages, failed requests",
      "sales": "pricing, discounts, plans"}, "technical"),
    ("zh-plan-quote", "请发一份企业版的报价单，我们想比较一下不同档位的价格。", "choice",
     "Which team should handle `body`?",
     {"billing": "invoices, payments, refunds",
      "technical": "bugs and outages, failed requests",
      "sales": "pricing, discounts, plans"}, "sales"),
    ("en-plan-price", "How much does the pro plan cost, and is there a discount if we pay for a year upfront?", "choice",
     "Which team should handle `body`?",
     {"billing": "invoices, payments, refunds",
      "technical": "bugs and outages, failed requests",
      "sales": "pricing, discounts, plans"}, "sales"),
    # ---- score: severity 0 (none) .. 4 (critical), judged low/high only
    ("zh-severity-critical", "整个系统无法登录，所有业务中断，客户在大量投诉。", "score",
     "How severe is the incident?",
     ["no impact", "cosmetic only", "minor and local", "major, some customers affected",
      "critical, business stopped"], "high"),
    ("zh-severity-mild", "包装盒上有一道很浅的划痕，产品本身完好。", "score",
     "How severe is the incident?",
     ["no impact", "cosmetic only", "minor and local", "major, some customers affected",
      "critical, business stopped"], "low"),
    ("en-severity-mild", "A minor cosmetic scratch on the outer box; nothing inside was damaged.", "score",
     "How severe is the incident?",
     ["no impact", "cosmetic only", "minor and local", "major, some customers affected",
      "critical, business stopped"], "low"),
    ("en-severity-critical", "The whole data center is down and every customer is affected.", "score",
     "How severe is the incident?",
     ["no impact", "cosmetic only", "minor and local", "major, some customers affected",
      "critical, business stopped"], "high"),
]


def build_question(case, wrap):
    cid, state, kind, ins, third, _expect = case
    q = {"type": kind, "instructions": ins}
    if kind == "noul":
        q["criteria"] = {"true": "yes", "false": "no"}
    elif kind == "choice":
        q["criteria"] = third
    else:
        q["criteria"] = third
    # upstream passes the state as {"body": "..."}; a plain string is the same
    # tokens minus the JSON wrapper, so the two are worth comparing
    st = {"body": state} if wrap else state
    return {"state": st, "questions": {cid: q}}


def run_case(case, rounds, wrap):
    cid = case[0]
    path = os.path.join(TMP, cid + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(build_question(case, wrap), f, ensure_ascii=False)
    got = []
    for _ in range(rounds):
        out = subprocess.run([EXE, "--json", path], capture_output=True, text=True,
                             encoding="utf-8", errors="replace")
        text = out.stdout + out.stderr
        m = re.search(r"noul\s+([0-9.]+)", text)
        if m:
            got.append("true" if float(m.group(1)) > 0.5 else "false")
            continue
        m = re.search(r"choice\s+(\S+)", text)
        if m:
            got.append(m.group(1))
            continue
        m = re.search(r"score\s+([0-9.]+)", text)
        if m:
            v = float(m.group(1))
            got.append("high" if v >= 2.5 else ("low" if v <= 1.5 else "mid(%.2f)" % v))
            continue
        got.append("?")
    return got


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    rounds = int(argv[0]) if argv else 3
    as_json = "--json" in sys.argv
    wrap = "--plain" not in sys.argv
    os.makedirs(TMP, exist_ok=True)

    per_case = []
    for case in CASES:
        got = run_case(case, rounds, wrap)
        expect = case[5]
        # noul expectations are written as Python booleans, the parsed answers as
        # the strings the CLI prints
        want = {True: "true", False: "false"}.get(expect, expect)
        ok = [g == want for g in got]
        per_case.append((case[0], case[2], want, got, ok))

    total = sum(len(ok) for *_, ok in per_case)
    hits = sum(sum(ok) for *_, ok in per_case)
    stable = all(all(ok) or not any(ok) for *_, ok in per_case)

    if as_json:
        print(json.dumps({
            "rounds": rounds,
            "cases": [{"id": cid, "kind": k, "expect": e, "got": g} for cid, k, e, g, _ in per_case],
            "hits": hits, "total": total,
        }, ensure_ascii=False, indent=1))
        return

    print("accuracy probe: %d cases x %d rounds = %d forwards (state: %s)" %
          (len(CASES), rounds, total, "{\"body\": ...}" if wrap else "plain string"))
    kind_hits, kind_total = {}, {}
    for cid, kind, expect, got, ok in per_case:
        kind_hits[kind] = kind_hits.get(kind, 0) + sum(ok)
        kind_total[kind] = kind_total.get(kind, 0) + len(ok)
        mark = "ok  " if all(ok) else ("MIX " if any(ok) else "MISS")
        uniq = "" if len(set(got)) == 1 else "  <-- unstable: %s" % ",".join(got)
        print("  %s %-22s %-6s expect %-10s got %s%s" % (mark, cid, kind, expect, ",".join(got), uniq))
    print("")
    for kind in sorted(kind_total):
        print("  %-8s %d/%d" % (kind, kind_hits[kind], kind_total[kind]))
    lang_hits, lang_total = {}, {}
    for cid, _kind, _e, _g, ok in per_case:
        lang = cid.split("-")[0]
        lang_hits[lang] = lang_hits.get(lang, 0) + sum(ok)
        lang_total[lang] = lang_total.get(lang, 0) + len(ok)
    for lang in sorted(lang_total):
        print("  lang %-4s %d/%d" % (lang, lang_hits[lang], lang_total[lang]))
    print("  overall  %d/%d = %.1f%%   determinism: %s" %
          (hits, total, 100.0 * hits / total, "all rounds agree per case" if stable else "some case flips"))


if __name__ == "__main__":
    main()