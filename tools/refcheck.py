#!/usr/bin/env python
"""Independent reference check for the Zig ModernBERT + decision-head port.

Recomputes the whole forward pass in plain Python -- stdlib only (struct/json/
math), no numpy, no torch -- and diffs it against the per-layer numbers that
`laya.exe --dumpstats` prints. The point is to catch a structural mistake that
would otherwise only show up as "the model behaves oddly": wrong GLU split,
wrong RoPE pairing, wrong head layout, a missing norm.

    zig-out/bin/laya.exe --dumpstats --json q.json 2> _dump.txt
    python -X utf8 tools/refcheck.py _dump.txt

Runs the full 22 encoder layers, both decision-head layers, the option-marker
scorer and the final logits. Takes a couple of minutes: it is ~10M dot products
of length 768 in interpreted Python.
"""
import json
import math
import re
import struct
import sys
from operator import mul

D = 768
NL = 22
HD = 64
NH = 12
INTER = 1152
EPS = 1e-5
ROPE_THETA = 160000.0
WINDOW = 64  # local_attention // 2, the half-window of sliding_attention layers

INV_FREQ = [1.0 / (ROPE_THETA ** (d / (HD / 2))) for d in range(HD // 2)]


# --------------------------------------------------------------------- weights
class Safetensors:
    def __init__(self, path):
        self.f = open(path, "rb")
        n = struct.unpack("<Q", self.f.read(8))[0]
        self.hdr = json.loads(self.f.read(n))
        self.base = 8 + n
        self.cache = {}

    def _decode(self, e, raw):
        if e["dtype"] == "F16":
            return struct.unpack("<%de" % (len(raw) // 2), raw)
        if e["dtype"] == "F32":
            return struct.unpack("<%df" % (len(raw) // 4), raw)
        raise ValueError("dtype " + e["dtype"])

    def get(self, name):
        if name not in self.cache:
            e = self.hdr[name]
            lo, hi = e["data_offsets"]
            self.f.seek(self.base + lo)
            self.cache[name] = self._decode(e, self.f.read(hi - lo))
        return self.cache[name]

    def rows(self, name, row_len, idxs):
        """Read only a few rows -- the embedding table is 393 MB of f16."""
        e = self.hdr[name]
        lo = e["data_offsets"][0]
        nbytes = row_len * 2
        out = []
        for i in idxs:
            self.f.seek(self.base + lo + i * nbytes)
            out.append(self._decode(e, self.f.read(nbytes)))
        return out

    def drop(self, prefix):
        for k in [k for k in self.cache if k.startswith(prefix)]:
            del self.cache[k]


def matvec(st, name, x):
    """y = W x for a [n, k] row-major weight (torch Linear layout)."""
    w = st.get(name)
    k = len(x)
    return [sum(map(mul, w[j * k:(j + 1) * k], x)) for j in range(len(w) // k)]


# ----------------------------------------------------------------------- math
def layernorm(v, w, b=None):
    n = len(v)
    mean = sum(v) / n
    var = sum((t - mean) * (t - mean) for t in v) / n
    inv = 1.0 / math.sqrt(var + EPS)
    if b is None:
        return [(t - mean) * inv * w[i] for i, t in enumerate(v)]
    return [(t - mean) * inv * w[i] + b[i] for i, t in enumerate(v)]


def gelu(x):
    return 0.5 * x * (1.0 + math.erf(x * 0.7071067811865476))


def relu(x):
    return x if x > 0.0 else 0.0


def softmax(v):
    m = max(v)
    e = [math.exp(t - m) for t in v]
    s = sum(e)
    return [t / s for t in e]


def rope(v, pos):
    """HF rotate_half pairing: dimension i rotates with i + head_dim/2."""
    out = [0.0] * HD
    for d in range(HD // 2):
        f = INV_FREQ[d] * pos
        c, s = math.cos(f), math.sin(f)
        x0, x1 = v[d], v[d + HD // 2]
        out[d] = x0 * c - x1 * s
        out[d + HD // 2] = x1 * c + x0 * s
    return out


def split_heads(rows):
    """[L][3*D] -> [L][3*NH][HD]; q is [:NH], k is [NH:2NH], v is [2NH:]."""
    return [[row[i * HD:(i + 1) * HD] for i in range(3 * NH)] for row in rows]


def merge_heads(rows):
    return [[d for hv in row for d in hv] for row in rows]


def attention(q, k, v, sliding):
    """q/k/v are [L][D] with heads laid out head-major. Returns [L][D]."""
    L = len(q)
    scale = 1.0 / math.sqrt(HD)
    out = [[0.0] * D for _ in range(L)]
    for i in range(L):
        klo = max(0, i - WINDOW) if sliding else 0
        khi = min(L, i + WINDOW + 1) if sliding else L
        row = out[i]
        for h in range(NH):
            o = h * HD
            qi = q[i][o:o + HD]
            p = softmax([sum(map(mul, qi, k[j][o:o + HD])) * scale
                         for j in range(klo, khi)])
            for t, j in enumerate(range(klo, khi)):
                pj = p[t]
                if pj == 0.0:
                    continue
                vj = v[j][o:o + HD]
                for d in range(HD):
                    row[o + d] += pj * vj[d]
    return out


# ------------------------------------------------------------------- encoder
def encoder_layer(st, x, l):
    prefix = "encoder.layers.%d." % l
    L = len(x)
    sliding = (l % 3) != 0
    if l == 0:
        src = x          # layer 0's attn_norm is nn.Identity (embeddings already normed)
    else:
        src = [layernorm(row, st.get(prefix + "attn_norm.weight")) for row in x]

    qkv = [matvec(st, prefix + "attn.Wqkv.weight", row) for row in src]
    heads = split_heads(qkv)  # [L][3*NH][HD]
    q = merge_heads([[rope(hv, i) for hv in row[:NH]] for i, row in enumerate(heads)])
    k = merge_heads([[rope(hv, i) for hv in row[NH:2 * NH]] for i, row in enumerate(heads)])
    v = merge_heads([row[2 * NH:] for row in heads])

    ctx = attention(q, k, v, sliding)
    x = [[a + b for a, b in zip(row, matvec(st, prefix + "attn.Wo.weight", cr))]
         for row, cr in zip(x, ctx)]

    n2 = [layernorm(row, st.get(prefix + "mlp_norm.weight")) for row in x]
    glu = []
    for row in n2:
        wi = matvec(st, prefix + "mlp.Wi.weight", row)
        glu.append([gelu(wi[j]) * wi[INTER + j] for j in range(INTER)])
    x = [[a + b for a, b in zip(row, matvec(st, prefix + "mlp.Wo.weight", g))]
         for row, g in zip(x, glu)]
    return x


# ---------------------------------------------------------------------- head
def head_layer(st, x, l):
    """nn.TransformerEncoderLayer(d, 12, 4*d, norm_first=True), bidirectional."""
    prefix = "head.layers.%d." % l
    n1 = [layernorm(row, st.get(prefix + "norm1.weight"), st.get(prefix + "norm1.bias")) for row in x]
    ib = st.get(prefix + "self_attn.in_proj_bias")
    qkv = [[v + ib[i] for i, v in enumerate(matvec(st, prefix + "self_attn.in_proj_weight", row))]
           for row in n1]
    heads = split_heads(qkv)  # [L][3*NH][HD]
    q = merge_heads([row[:NH] for row in heads])
    k = merge_heads([row[NH:2 * NH] for row in heads])
    v = merge_heads([row[2 * NH:] for row in heads])

    ctx = attention(q, k, v, False)
    ob = st.get(prefix + "self_attn.out_proj.bias")
    x = [[a + v2 + ob[i] for i, (a, v2) in
          enumerate(zip(row, matvec(st, prefix + "self_attn.out_proj.weight", cr)))]
         for row, cr in zip(x, ctx)]

    n2 = [layernorm(row, st.get(prefix + "norm2.weight"), st.get(prefix + "norm2.bias")) for row in x]
    ff = []
    for row in n2:
        h = matvec(st, prefix + "linear1.weight", row)
        b = st.get(prefix + "linear1.bias")
        ff.append([relu(t + b[i]) for i, t in enumerate(h)])
    b2 = st.get(prefix + "linear2.bias")
    return [[a + t + b2[i] for i, (a, t) in
             enumerate(zip(row, matvec(st, prefix + "linear2.weight", f)))]
            for row, f in zip(x, ff)]


def scorer(st, hv):
    v = layernorm(hv, st.get("scorer.0.weight"), st.get("scorer.0.bias"))
    h = matvec(st, "scorer.1.weight", v)
    b = st.get("scorer.1.bias")
    h = [gelu(t + b[i]) for i, t in enumerate(h)]
    w = st.get("scorer.3.weight")
    return sum(map(mul, w, h)) + st.get("scorer.3.bias")[0]


# ---------------------------------------------------------------------- main
def read_text(path):
    """PowerShell's `2>` writes UTF-16 and prefixes native stderr with the exe path."""
    raw = open(path, "rb").read()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff") or raw.count(b"\x00") > len(raw) // 8:
        return raw.decode("utf-16", errors="replace")
    return raw.decode("utf-8", errors="replace")


def parse_dump(path):
    ids, marks, logits = [], [], []
    layers = {}
    meta = None
    for line in read_text(path).splitlines():
        m = re.search(r"\[dump\] meta L=(\d+) qtype=(\d+)", line)
        if m:
            meta = (int(m.group(1)), int(m.group(2)))
            continue
        m = re.search(r"\[dump\] ids((?: \d+)+)", line)
        if m:
            ids.extend(int(t) for t in m.group(1).split())
            continue
        m = re.search(r"\[dump\] mark((?: \d+)+)", line)
        if m:
            marks.extend(int(t) for t in m.group(1).split())
            continue
        m = re.search(r"\[dump\] logit(.*)", line)
        if m:
            logits.extend(float(t) for t in m.group(1).split())
            continue
        # tags are space-padded to 5 chars, so separate with \s+ not a literal space
        m = re.search(r"\[dump\] (\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)", line)
        if m:
            v = [float(m.group(i)) for i in range(2, 7)]
            layers[m.group(1)] = (v[:3], v[3], v[4])
    return meta, ids, marks, logits, layers


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    meta, ids, marks, logits, got = parse_dump(sys.argv[1])
    if meta is None:
        print("no [dump] output found in " + sys.argv[1])
        return 1
    L, qtype = meta
    print("reference check: L=%d qtype=%d markers=%s" % (L, qtype, marks))

    st = Safetensors("model.safetensors")
    x = [list(r) for r in st.rows("encoder.embeddings.tok_embeddings.weight", D, ids)]
    wnorm = st.get("encoder.embeddings.norm.weight")
    x = [layernorm(row, wnorm) for row in x]
    st.drop("encoder.embeddings.tok_embeddings")

    bad = []

    def compare(tag, x, tol=5e-3):
        if tag not in got:
            return
        flat = [t for row in x for t in row]
        mine = (flat[0:3], sum(flat), sum(abs(t) for t in flat))
        ref, rs, ra = got[tag]
        dmax = max(abs(p - q) for p, q in zip(mine[0], ref))
        serr = abs(mine[1] - rs) / max(1.0, abs(rs))
        aerr = abs(mine[2] - ra) / max(1.0, abs(ra))
        ok = dmax < tol and serr < 1e-3 and aerr < 1e-3
        if not ok:
            bad.append(tag)
        print("  %-6s %s  |d|=%.1e  sum=%.1e  asum=%.1e"
              % (tag, "ok  " if ok else "DIFF", dmax, serr, aerr))
        if not ok:
            print("        zig %s" % ["%.6f" % t for t in ref])
            print("        ref %s" % ["%.6f" % t for t in mine[0]])

    compare("emb", x)
    for l in range(NL):
        x = encoder_layer(st, x, l)
        compare("l%d" % l, x)
        st.drop("encoder.layers.%d." % l)
    x = [layernorm(row, st.get("encoder.final_norm.weight")) for row in x]
    compare("final", x)

    te = st.rows("type_emb.weight", D, [qtype])[0]
    x = [[t + te[d] for d, t in enumerate(row)] for row in x]
    compare("temb", x)

    for l in range(2):
        x = head_layer(st, x, l)
        compare("h%d" % l, x)

    if marks and logits:
        mine = [scorer(st, x[m]) for m in marks]
        dmax = max(abs(p - q) for p, q in zip(mine, logits))
        ok = dmax < 5e-2
        if not ok:
            bad.append("logits")
        print("  %-6s %s  |d|=%.1e" % ("logit", "ok  " if ok else "DIFF", dmax))
        print("        zig %s" % ["%.5f" % t for t in logits])
        print("        ref %s" % ["%.5f" % t for t in mine])
        if len(logits) >= 2:
            pz = softmax(logits)
            pr = softmax(mine)
            print("        softmax zig %s  ref %s"
                  % (["%.4f" % t for t in pz], ["%.4f" % t for t in pr]))
    else:
        print("  (no markers/logits in dump)")

    if bad:
        print("FAIL: %s differ" % ", ".join(bad))
        return 1
    print("PASS: the Zig port reproduces the reference forward pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
