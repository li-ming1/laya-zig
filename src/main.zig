//! Inference runtime for `convaiinnovations/laya-multilingual`.
//!
//! mmBERT-base encoder (22 layers, hidden 768, 12 heads, 256k vocab, RoPE, sliding window)
//! + the Laya decision head (2 transformer layers, option-marker scorer, act/escalate head).
//! The whole model runs locally: no torch, no Python, weights straight from model.safetensors.
//!
//! build:  zig build                 (fast, native CPU; -Doptimize=Debug to override)
//! run:    zig-out\bin\laya.exe                    # built-in demo (Hindi billing email)
//!         zig-out\bin\laya.exe --json q.json      # your own state + questions (UTF-8)
//!         zig-out\bin\laya.exe --threads 8 --dir .
//!         zig-out\bin\laya.exe --tokcheck golden.jsonl   # validate the tokenizer
//!
//! q.json: {"state": "...", "questions": {"id": {"type": "choice"|"score"|"noul",
//!         "instructions": "...", "criteria": {...}|[...]}}}
//!
//! browser: --serve [--port 8080]  start the local UI (see src/web/index.html,
//!                                 smoke-tested by tools/web-smoke.js)
//! misc:   --selftest               check the SIMD kernels against f64 (run after
//!                                 touching dotv/matmul — they have been miscompiled
//!                                 by this zig build before)
//!
//! snake:  --snake                let the model play Snake (one game, animated board)
//!         --snake --games 10     summary over N games, no animation
//!         --snake --policy greedy|random   baselines to compare against
//!         --snake --prompt       print the board text the model is given
//!         --size N --max-steps N --delay MS --seed N
//!
//! Validated against an independent pure-Python reference implementation
//! (tools/refcheck.py): layer-by-layer activations, logits and probabilities agree
//! to ~1e-5. README.md documents that check and the two others (--selftest,
//! tools/web-smoke.js).

const std = @import("std");
const Allocator = std.mem.Allocator;
const AList = std.array_list.Managed;

// --------------------------------------------------------------------------- architecture
const D: usize = 768;
const H: usize = 12;
const HD: usize = 64;
const NL: usize = 22;
const INTER: usize = 1152;
const EPS: f32 = 1e-5;
const ROPE_THETA: f32 = 160000.0;
const WINDOW: usize = 128 / 2; // sliding layers: |i-j| <= 64
const HEAD_LAYERS: usize = 2;
const HEAD_FF: usize = 4 * D;
const MAX_LEN: usize = 1024;
const HEAD_MAX_LEN: usize = 256;

const CLS_ID: u32 = 2; // <bos>
const SEP_ID: u32 = 1; // <eos>
const MASK_ID: u32 = 4; // <mask>
const MASK_TOKEN: []const u8 = "<mask>";
const SEP_CHAR: []const u8 = "▁"; // U+2581

// --------------------------------------------------------------------------- math
fn erf(x: f32) f32 {
    const s: f32 = if (x < 0) -1.0 else 1.0;
    const a = @abs(x);
    const t: f32 = 1.0 / (1.0 + 0.3275911 * a);
    const poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
    return s * (1.0 - poly * @exp(-a * a));
}

fn gelu(x: f32) f32 {
    return 0.5 * x * (1.0 + erf(x * 0.70710678118654752440));
}

fn softmax(x: []f32) void {
    var mx: f32 = -std.math.inf(f32);
    for (x) |v| mx = @max(mx, v);
    var sum: f32 = 0;
    for (x) |*v| {
        v.* = @exp(v.* - mx);
        sum += v.*;
    }
    for (x) |*v| v.* /= sum;
}

fn layerNorm(x: []f32, w: []const f32, b: ?[]const f32) void {
    const n: f32 = @floatFromInt(x.len);
    var mean: f32 = 0;
    for (x) |v| mean += v;
    mean /= n;
    var varr: f32 = 0;
    for (x) |v| {
        const d = v - mean;
        varr += d * d;
    }
    varr = 1.0 / @sqrt(varr / n + EPS);
    for (x, 0..) |*v, i| v.* = (v.* - mean) * varr * w[i] + if (b) |bb| bb[i] else 0.0;
}

// --------------------------------------------------------------------------- threading
var g_threads: usize = 4;
var g_debug: bool = false;

/// `--dumpstats`: after every encoder layer, print a few numbers describing the
/// hidden state, so a reference implementation can be diffed against this one
/// layer by layer (see tools/refcheck.py). Only the first forward pass dumps.
var g_dump_stats: bool = false;
var g_fwd_count: usize = 0;

/// Kept to short lines on purpose: tools/refcheck.py reads this after PowerShell
/// has captured stderr, and PowerShell hard-wraps long lines at the console width.
fn dumpStats(tag: []const u8, x: []const f32, L: usize) void {
    var sum: f64 = 0;
    var asum: f64 = 0;
    for (x[0 .. L * D]) |v| {
        sum += v;
        asum += @abs(v);
    }
    std.debug.print("[dump] {s:<5} {d:.5} {d:.5} {d:.5} {d:.3} {d:.3}\n", .{ tag, x[0], x[1], x[2], sum, asum });
}

fn dumpIds(ids: []const u32) void {
    var i: usize = 0;
    while (i < ids.len) : (i += 8) {
        std.debug.print("[dump] ids", .{});
        for (ids[i..@min(i + 8, ids.len)]) |v| std.debug.print(" {d}", .{v});
        std.debug.print("\n", .{});
    }
}

const RangeJob = struct {
    ctx: *anyopaque,
    f: *const fn (*anyopaque, usize, usize) void,
    lo: usize,
    hi: usize,
};

fn rangeWorker(job: RangeJob) void {
    if (job.hi > job.lo) job.f(job.ctx, job.lo, job.hi);
}

fn parallel(alloc: Allocator, n: usize, ctx: *anyopaque, f: *const fn (*anyopaque, usize, usize) void) void {
    if (n == 0) return;
    var nt = g_threads;
    if (n < 64) nt = 1;
    if (nt <= 1) {
        f(ctx, 0, n);
        return;
    }
    const chunk = (n + nt - 1) / nt;
    const threads = alloc.alloc(std.Thread, nt - 1) catch {
        f(ctx, 0, n);
        return;
    };
    defer alloc.free(threads);
    var spawned: usize = 0;
    for (0..nt - 1) |t| {
        const lo = @min(t * chunk, n);
        const hi = @min(lo + chunk, n);
        threads[t] = std.Thread.spawn(.{}, rangeWorker, .{RangeJob{
            .ctx = ctx,
            .f = f,
            .lo = lo,
            .hi = hi,
        }}) catch break;
        spawned = t + 1;
    }
    for (spawned..nt) |t| {
        const lo = @min(t * chunk, n);
        const hi = @min(lo + chunk, n);
        if (hi > lo) f(ctx, lo, hi);
    }
    for (threads[0..spawned]) |t| t.join();
}

// --------------------------------------------------------------------------- matmul
const MM = struct {
    out: []f32,
    x: []const f32,
    w: []const f32,
    n: usize,
    k: usize,
    rows: usize,
};

/// out[j] = dot(x, w[j*k .. (j+1)*k])   (w is [n][k] row-major)
/// NOTE: kept as a single 8-wide accumulator on purpose. A 4-accumulator / 32-wide
/// unrolled version is miscompiled by this zig build (0.17.0-dev.1737) and silently
/// returns wrong results, so do not "optimise" it back without re-validating.
/// dot product of two equal-length slices.
///
/// NOTE: an earlier 4-accumulator / 32-wide-unrolled variant of this function was
/// silently miscompiled by zig 0.17.0-dev.1737 (results ~40% off). Any change here
/// must be re-validated with `laya.exe --selftest`, which checks this kernel against
/// an f64 reference for every shape the model uses.
fn dotv(a: []const f32, b: []const f32) f32 {
    const n = @min(a.len, b.len);
    var a0: @Vector(8, f32) = @splat(0.0);
    var a1: @Vector(8, f32) = @splat(0.0);
    var i: usize = 0;
    while (i + 16 <= n) : (i += 16) {
        const x0: @Vector(8, f32) = @bitCast(a[i..][0..8].*);
        const x1: @Vector(8, f32) = @bitCast(a[i + 8 ..][0..8].*);
        const y0: @Vector(8, f32) = @bitCast(b[i..][0..8].*);
        const y1: @Vector(8, f32) = @bitCast(b[i + 8 ..][0..8].*);
        a0 += x0 * y0;
        a1 += x1 * y1;
    }
    while (i + 8 <= n) : (i += 8) {
        const x0: @Vector(8, f32) = @bitCast(a[i..][0..8].*);
        const y0: @Vector(8, f32) = @bitCast(b[i..][0..8].*);
        a0 += x0 * y0;
    }
    var s: f32 = 0;
    while (i < n) : (i += 1) s += a[i] * b[i];
    return s + @reduce(.Add, a0 + a1);
}

/// `--selftest`: check the SIMD kernels against f64 for the shapes the model uses.
fn selfTest(gpa: Allocator, w: *std.Io.Writer) !void {
    const Shape = struct { rows: usize, n: usize, k: usize, what: []const u8 };
    const shapes = [_]Shape{
        .{ .rows = 1, .n = 768, .k = 768, .what = "scorer / act head" },
        .{ .rows = 63, .n = 2304, .k = 768, .what = "qkv + mlp in" },
        .{ .rows = 63, .n = 768, .k = 1152, .what = "mlp out" },
        .{ .rows = 63, .n = 3072, .k = 768, .what = "head ff" },
        .{ .rows = 63, .n = 768, .k = 768, .what = "attn / head proj" },
    };
    var rng = std.Random.DefaultPrng.init(0x5eed);
    var worst: f64 = 0;
    for (shapes) |s| {
        const x = try gpa.alloc(f32, s.rows * s.k);
        defer gpa.free(x);
        const wm = try gpa.alloc(f32, s.n * s.k);
        defer gpa.free(wm);
        const out = try gpa.alloc(f32, s.rows * s.n);
        defer gpa.free(out);
        for (x) |*v| v.* = rng.random().float(f32) * 2.0 - 1.0;
        for (wm) |*v| v.* = rng.random().float(f32) * 2.0 - 1.0;

        matmul(gpa, out, x, wm, s.rows, s.n, s.k);

        var local_worst: f64 = 0;
        for (0..s.rows) |i| {
            var j: usize = 0;
            while (j < s.n) : (j += 29) {
                var ref: f64 = 0;
                for (0..s.k) |t| ref += @as(f64, x[i * s.k + t]) * @as(f64, wm[j * s.k + t]);
                const got: f64 = out[i * s.n + j];
                const rel = @abs(got - ref) / @max(1.0, @abs(ref));
                if (rel > local_worst) local_worst = rel;
            }
        }
        worst = @max(worst, local_worst);
        try w.print("  {d:>3}x{d:<5} x {d:>4}x{d:<5} {s:<18} max rel err {e:.2}\n", .{ s.rows, s.k, s.k, s.n, s.what, local_worst });
    }
    try w.print("\nworst relative error vs f64: {e:.3}\n", .{worst});
    if (worst > 1e-4) {
        try w.print("FAIL: the GEMM kernel is wrong (expected < 1e-4)\n", .{});
    } else {
        try w.print("PASS\n", .{});
    }
    try w.flush();
}

/// single input row against n weight rows
fn rowDot(out: []f32, x: []const f32, w: []const f32) void {
    for (0..out.len) |j| out[j] = dotv(x, w[j * x.len ..][0..x.len]);
}

/// out[i][j] = dot(x[i], w[j]) -- parallelised over output columns so that each
/// weight row is streamed exactly once (this is the difference between ~2 s and
/// ~200 ms per question: the other order re-reads the whole matrix per token).
fn mmJob(ctx: *anyopaque, jlo: usize, jhi: usize) void {
    const m: *MM = @ptrCast(@alignCast(ctx));
    for (jlo..jhi) |j| {
        const wrow = m.w[j * m.k ..][0..m.k];
        for (0..m.rows) |i| {
            m.out[i * m.n + j] = dotv(m.x[i * m.k ..][0..m.k], wrow);
        }
    }
}

fn biasJob(ctx: *anyopaque, lo: usize, hi: usize) void {
    const m: *MM = @ptrCast(@alignCast(ctx));
    for (lo..hi) |i| {
        const row = m.out[i * m.n ..][0..m.n];
        for (row, 0..) |*v, j| v.* += m.w[j];
    }
}

fn matmul(alloc: Allocator, out: []f32, x: []const f32, w: []const f32, rows: usize, n: usize, k: usize) void {
    var m = MM{ .out = out, .x = x, .w = w, .n = n, .k = k, .rows = rows };
    parallel(alloc, n, &m, mmJob);
}

fn linear(alloc: Allocator, out: []f32, x: []const f32, w: []const f32, bias: ?[]const f32, rows: usize, n: usize, k: usize) void {
    matmul(alloc, out, x, w, rows, n, k);
    if (bias) |b| {
        var m = MM{ .out = out, .x = &[0]f32{}, .w = b, .n = n, .k = 0, .rows = rows };
        parallel(alloc, rows, &m, biasJob);
    }
}

fn rowDot1(a: []const f32, b: []const f32) f32 {
    var o: [1]f32 = undefined;
    rowDot(&o, a, b);
    return o[0];
}

// --------------------------------------------------------------------------- safetensors
const TensorInfo = struct {
    dtype: []const u8,
    offset: usize,
    len: usize,
};

fn f16at(raw: []const u8, i: usize) f32 {
    const u = std.mem.readInt(u16, raw[i * 2 ..][0..2], .little);
    return @floatCast(@as(f16, @bitCast(u)));
}

fn toF32(alloc: Allocator, raw: []const u8, dtype: []const u8) ![]f32 {
    if (std.mem.eql(u8, dtype, "F32")) {
        const n = raw.len / 4;
        const out = try alloc.alloc(f32, n);
        for (0..n) |i| out[i] = @bitCast(std.mem.readInt(u32, raw[i * 4 ..][0..4], .little));
        return out;
    }
    if (!std.mem.eql(u8, dtype, "F16")) return error.UnsupportedDtype;
    const n = raw.len / 2;
    const out = try alloc.alloc(f32, n);
    for (0..n) |i| out[i] = f16at(raw, i);
    return out;
}

fn tensorRaw(infos: *const std.StringHashMap(TensorInfo), data: []const u8, name: []const u8) ![]const u8 {
    const t = infos.get(name) orelse {
        std.debug.print("missing tensor: {s}\n", .{name});
        return error.MissingTensor;
    };
    return data[t.offset .. t.offset + t.len];
}

fn tensorF32(alloc: Allocator, infos: *const std.StringHashMap(TensorInfo), data: []const u8, name: []const u8) ![]const f32 {
    const t = infos.get(name) orelse {
        std.debug.print("missing tensor: {s}\n", .{name});
        return error.MissingTensor;
    };
    return try toF32(alloc, data[t.offset .. t.offset + t.len], t.dtype);
}

// --------------------------------------------------------------------------- tokenizer
const Tokenizer = struct {
    vocab: std.StringHashMap(u32),
    merges: std.StringHashMap(u32),
    added: std.StringHashMap(u32),
    added_lens: []usize, // distinct lengths, descending
    start_byte: [256]bool,
    byte_ids: [256]u32,

    fn findAdded(self: *const Tokenizer, s: []const u8) ?struct { usize, u32 } {
        for (self.added_lens) |len| {
            if (len <= s.len) {
                if (self.added.get(s[0..len])) |id| return .{ len, id };
            }
        }
        return null;
    }
};

const JScan = struct {
    s: []const u8,
    i: usize,

    fn ws(self: *JScan) void {
        while (self.i < self.s.len) : (self.i += 1) {
            switch (self.s[self.i]) {
                ' ', '\n', '\r', '\t' => {},
                else => return,
            }
        }
    }
    fn peek(self: *JScan) !u8 {
        if (self.i >= self.s.len) return error.TruncatedJson;
        return self.s[self.i];
    }
    fn byte(self: *JScan) !u8 {
        if (self.i >= self.s.len) return error.TruncatedJson;
        const c = self.s[self.i];
        self.i += 1;
        return c;
    }
    fn expect(self: *JScan, c: u8) !void {
        self.ws();
        if (try self.byte() != c) return error.BadJson;
    }
    fn skipLit(self: *JScan) !void {
        self.ws();
        while (self.i < self.s.len) : (self.i += 1) {
            switch (self.s[self.i]) {
                ',', '}', ']', ' ', '\n', '\r', '\t' => return,
                else => {},
            }
        }
    }
    fn skipString(self: *JScan) !void {
        try self.expect('"');
        while (true) {
            const c = try self.byte();
            if (c == '"') return;
            if (c == '\\') _ = try self.byte();
        }
    }
    /// raw (still escaped) length of the string starting at self.i (which must be '"')
    fn rawStringLen(self: *const JScan) !usize {
        var i = self.i + 1;
        var n: usize = 0;
        while (i < self.s.len) {
            const c = self.s[i];
            if (c == '"') return n;
            if (c == '\\') {
                i += 2;
                n += 2;
                continue;
            }
            i += 1;
            n += 1;
        }
        return error.TruncatedJson;
    }
    fn hex4(self: *JScan) !u32 {
        var v: u32 = 0;
        for (0..4) |_| {
            const c = try self.byte();
            const d: u32 = switch (c) {
                '0'...'9' => c - '0',
                'a'...'f' => c - 'a' + 10,
                'A'...'F' => c - 'A' + 10,
                else => return error.BadJson,
            };
            v = (v << 4) | d;
        }
        return v;
    }
    fn stringInto(self: *JScan, buf: []u8) ![]const u8 {
        try self.expect('"');
        var o: usize = 0;
        while (true) {
            const c = try self.byte();
            if (c == '"') break;
            if (c != '\\') {
                if (o >= buf.len) return error.BufferTooSmall;
                buf[o] = c;
                o += 1;
                continue;
            }
            const e = try self.byte();
            switch (e) {
                'n', 't', 'r', 'b', 'f' => {
                    if (o + 1 > buf.len) return error.BufferTooSmall;
                    buf[o] = switch (e) {
                        'n' => '\n',
                        't' => '\t',
                        'r' => '\r',
                        'b' => 0x08,
                        else => 0x0c,
                    };
                    o += 1;
                },
                '/', '"', '\\' => {
                    if (o + 1 > buf.len) return error.BufferTooSmall;
                    buf[o] = e;
                    o += 1;
                },
                'u' => {
                    var cp: u32 = try self.hex4();
                    if (cp >= 0xD800 and cp < 0xDC00) {
                        try self.expect('\\');
                        if (try self.byte() != 'u') return error.BadJson;
                        const lo = try self.hex4();
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    }
                    o = try encodeUtf8(cp, buf, o);
                },
                else => return error.BadJson,
            }
        }
        return buf[0..o];
    }
    fn uint(self: *JScan) !u32 {
        self.ws();
        var v: u32 = 0;
        var any = false;
        while (self.i < self.s.len) {
            const c = self.s[self.i];
            if (c >= '0' and c <= '9') {
                v = v *% 10 + @as(u32, c - '0');
                any = true;
                self.i += 1;
            } else break;
        }
        if (!any) return error.BadJson;
        return v;
    }
};

fn encodeUtf8(cp: u32, buf: []u8, o: usize) !usize {
    var out: [4]u8 = undefined;
    const n: usize = if (cp < 0x80) 1 else if (cp < 0x800) 2 else if (cp < 0x10000) 3 else 4;
    switch (n) {
        1 => out[0] = @intCast(cp),
        2 => {
            out[0] = @intCast(0xC0 | (cp >> 6));
            out[1] = @intCast(0x80 | (cp & 0x3F));
        },
        3 => {
            out[0] = @intCast(0xE0 | (cp >> 12));
            out[1] = @intCast(0x80 | ((cp >> 6) & 0x3F));
            out[2] = @intCast(0x80 | (cp & 0x3F));
        },
        4 => {
            out[0] = @intCast(0xF0 | (cp >> 18));
            out[1] = @intCast(0x80 | ((cp >> 12) & 0x3F));
            out[2] = @intCast(0x80 | ((cp >> 6) & 0x3F));
            out[3] = @intCast(0x80 | (cp & 0x3F));
        },
        else => unreachable,
    }
    if (o + n > buf.len) return error.BufferTooSmall;
    @memcpy(buf[o .. o + n], out[0..n]);
    return o + n;
}

const TokSink = struct {
    tok: *Tokenizer,
    alloc: Allocator,
};

fn walkValue(sc: *JScan, sink: *TokSink) !void {
    sc.ws();
    switch (try sc.peek()) {
        '{' => {
            _ = try sc.byte();
            while (true) {
                sc.ws();
                if (try sc.peek() == '}') {
                    _ = try sc.byte();
                    return;
                }
                if (try sc.peek() == ',') {
                    _ = try sc.byte();
                    continue;
                }
                var kbuf: [128]u8 = undefined;
                const rawlen = try sc.rawStringLen();
                if (rawlen > kbuf.len) {
                    try sc.skipString();
                    try sc.expect(':');
                    try walkValue(sc, sink);
                    continue;
                }
                const key = try sc.stringInto(&kbuf);
                try sc.expect(':');
                if (std.mem.eql(u8, key, "vocab")) {
                    try parseVocab(sc, sink);
                } else if (std.mem.eql(u8, key, "merges")) {
                    try parseMerges(sc, sink);
                } else if (std.mem.eql(u8, key, "added_tokens")) {
                    try parseAdded(sc, sink);
                } else {
                    try walkValue(sc, sink);
                }
            }
        },
        '[' => {
            _ = try sc.byte();
            while (true) {
                sc.ws();
                if (try sc.peek() == ']') {
                    _ = try sc.byte();
                    return;
                }
                if (try sc.peek() == ',') {
                    _ = try sc.byte();
                    continue;
                }
                try walkValue(sc, sink);
            }
        },
        '"' => try sc.skipString(),
        else => try sc.skipLit(),
    }
}

fn parseVocab(sc: *JScan, sink: *TokSink) !void {
    try sc.expect('{');
    const buf = try sink.alloc.alloc(u8, 1 << 20);
    defer sink.alloc.free(buf);
    while (true) {
        sc.ws();
        if (try sc.peek() == '}') {
            _ = try sc.byte();
            return;
        }
        if (try sc.peek() == ',') {
            _ = try sc.byte();
            continue;
        }
        const s = try sc.stringInto(buf);
        try sc.expect(':');
        const id = try sc.uint();
        try sink.tok.vocab.put(try sink.alloc.dupe(u8, s), id);
    }
}

fn parseMerges(sc: *JScan, sink: *TokSink) !void {
    try sc.expect('[');
    var buf = try sink.alloc.alloc(u8, 1 << 18);
    defer sink.alloc.free(buf);
    var rank: u32 = 0;
    while (true) {
        sc.ws();
        if (try sc.peek() == ']') {
            _ = try sc.byte();
            return;
        }
        if (try sc.peek() == ',') {
            _ = try sc.byte();
            continue;
        }
        try sc.expect('[');
        const a = try sc.stringInto(buf);
        sc.ws();
        _ = try sc.byte(); // ','
        const b = try sc.stringInto(buf[a.len..]);
        try sc.expect(']');
        const key = try std.fmt.allocPrint(sink.alloc, "{s}\x00{s}", .{ a, b });
        try sink.tok.merges.put(key, rank);
        rank += 1;
    }
}

fn parseAdded(sc: *JScan, sink: *TokSink) !void {
    try sc.expect('[');
    const buf = try sink.alloc.alloc(u8, 1 << 16);
    defer sink.alloc.free(buf);
    while (true) {
        sc.ws();
        if (try sc.peek() == ']') {
            _ = try sc.byte();
            return;
        }
        if (try sc.peek() == ',') {
            _ = try sc.byte();
            continue;
        }
        try sc.expect('{');
        var id: u32 = 0;
        var content: ?[]const u8 = null;
        while (true) {
            sc.ws();
            if (try sc.peek() == '}') {
                _ = try sc.byte();
                break;
            }
            if (try sc.peek() == ',') {
                _ = try sc.byte();
                continue;
            }
            var kbuf: [64]u8 = undefined;
            const key = try sc.stringInto(&kbuf);
            try sc.expect(':');
            if (std.mem.eql(u8, key, "id")) {
                id = try sc.uint();
            } else if (std.mem.eql(u8, key, "content")) {
                content = try sc.stringInto(buf);
            } else if (try sc.peek() == '"') {
                try sc.skipString();
            } else {
                try sc.skipLit();
            }
        }
        if (content) |c| try sink.tok.added.put(try sink.alloc.dupe(u8, c), id);
    }
}

fn loadTokenizer(alloc: Allocator, io: std.Io, path: []const u8) !Tokenizer {
    const data = try readFile(alloc, io, path);
    defer alloc.free(data);

    var tok = Tokenizer{
        .vocab = std.StringHashMap(u32).init(alloc),
        .merges = std.StringHashMap(u32).init(alloc),
        .added = std.StringHashMap(u32).init(alloc),
        .added_lens = &.{},
        .start_byte = undefined,
        .byte_ids = undefined,
    };
    @memset(tok.start_byte[0..], false);
    @memset(tok.byte_ids[0..], 0);
    try tok.vocab.ensureTotalCapacity(300000);
    try tok.merges.ensureTotalCapacity(700000);

    var sc = JScan{ .s = data, .i = 0 };
    var sink = TokSink{ .tok = &tok, .alloc = alloc };
    walkValue(&sc, &sink) catch |e| {
        const lo = if (sc.i > 120) sc.i - 120 else 0;
        const hi = @min(data.len, sc.i + 120);
        std.debug.print("tokenizer.json parse error {} at offset {d}\n...{s}...\n", .{ e, sc.i, data[lo..hi] });
        return e;
    };

    var lens = AList(usize).init(alloc);
    var it = tok.added.keyIterator();
    while (it.next()) |k| {
        const l = k.*.len;
        var seen = false;
        for (lens.items) |x| {
            if (x == l) seen = true;
        }
        if (!seen) try lens.append(l);
        tok.start_byte[k.*[0]] = true;
    }
    std.mem.sort(usize, lens.items, {}, struct {
        fn lt(_: void, a: usize, b: usize) bool {
            return a > b;
        }
    }.lt);
    tok.added_lens = try lens.toOwnedSlice();

    const hexdig = "0123456789ABCDEF";
    var hb: [6]u8 = undefined;
    for (0..256) |b| {
        hb[0] = '<';
        hb[1] = '0';
        hb[2] = 'x';
        hb[3] = hexdig[b >> 4];
        hb[4] = hexdig[b & 15];
        hb[5] = '>';
        tok.byte_ids[b] = tok.vocab.get(&hb) orelse 3;
    }
    return tok;
}

fn bpePiece(tok: *const Tokenizer, alloc: Allocator, piece: []const u8, out: *AList(u32)) !void {
    if (piece.len == 0) return;
    var syms = AList([2]usize).init(alloc);
    defer syms.deinit();
    var i: usize = 0;
    while (i < piece.len) {
        const n = std.unicode.utf8ByteSequenceLength(piece[i]) catch 1;
        const len = @min(n, piece.len - i);
        try syms.append(.{ i, i + len });
        i += len;
    }
    var keybuf: [1024]u8 = undefined;
    while (syms.items.len > 1) {
        var best: u32 = std.math.maxInt(u32);
        var bi: usize = 0;
        var found = false;
        for (0..syms.items.len - 1) |s| {
            const a = syms.items[s];
            const b = syms.items[s + 1];
            const la = a[1] - a[0];
            const lb = b[1] - b[0];
            if (la + lb + 1 > keybuf.len) continue;
            @memcpy(keybuf[0..la], piece[a[0]..a[1]]);
            keybuf[la] = 0;
            @memcpy(keybuf[la + 1 ..][0..lb], piece[b[0]..b[1]]);
            if (tok.merges.get(keybuf[0 .. la + lb + 1])) |r| {
                if (!found or r < best) {
                    best = r;
                    bi = s;
                    found = true;
                }
            }
        }
        if (!found) break;
        syms.items[bi][1] = syms.items[bi + 1][1];
        _ = syms.orderedRemove(bi + 1);
    }
    for (syms.items) |s| {
        const str = piece[s[0]..s[1]];
        if (tok.vocab.get(str)) |id| {
            try out.append(id);
        } else {
            for (str) |b| try out.append(tok.byte_ids[b]);
        }
    }
}

fn isSepAt(t: []const u8, i: usize) bool {
    return i + 3 <= t.len and std.mem.startsWith(u8, t[i..], SEP_CHAR);
}

fn encodeSegment(tok: *const Tokenizer, alloc: Allocator, s: []const u8, out: *AList(u32)) !void {
    var norm = AList(u8).init(alloc);
    defer norm.deinit();
    try norm.ensureTotalCapacity(s.len + 3);
    for (s) |c| {
        if (c == ' ') try norm.appendSlice(SEP_CHAR) else try norm.append(c);
    }
    const t = norm.items;
    if (t.len == 0) return;

    var scratch: ?[]u8 = null;
    defer if (scratch) |b| alloc.free(b);

    var j: usize = 0;
    if (!isSepAt(t, 0)) {
        while (j < t.len and !isSepAt(t, j)) j += 1;
        scratch = try alloc.alloc(u8, j + 3);
        @memcpy(scratch.?[0..3], SEP_CHAR);
        @memcpy(scratch.?[3..], t[0..j]);
        try bpePiece(tok, alloc, scratch.?, out);
    }
    while (j < t.len) {
        var e = j + 3;
        while (e < t.len and !isSepAt(t, e)) e += 1;
        try bpePiece(tok, alloc, t[j..e], out);
        j = e;
    }
}

const EncCtx = struct {
    tok: *const Tokenizer,
    alloc: Allocator,
    out: *AList(u32),
};

fn encodeAll(tok: *const Tokenizer, alloc: Allocator, text: []const u8) ![]u32 {
    var out = AList(u32).init(alloc);
    var ctx = EncCtx{ .tok = tok, .alloc = alloc, .out = &out };
    var i: usize = 0;
    var plain_start: usize = 0;
    while (i < text.len) {
        if (ctx.tok.start_byte[text[i]]) {
            if (ctx.tok.findAdded(text[i..])) |m| {
                const len: usize = m[0];
                if (i > plain_start) try encodeSegment(ctx.tok, ctx.alloc, text[plain_start..i], ctx.out);
                try ctx.out.append(m[1]);
                i += len;
                plain_start = i;
                continue;
            }
        }
        i += 1;
    }
    if (text.len > plain_start) try encodeSegment(ctx.tok, ctx.alloc, text[plain_start..], ctx.out);
    return out.toOwnedSlice();
}

// --------------------------------------------------------------------------- weights
const EncLayer = struct {
    wqkv: []const f32,
    wo: []const f32,
    wi: []const f32,
    wo2: []const f32,
    attn_norm: ?[]const f32,
    mlp_norm: []const f32,
    sliding: bool,
};

const HeadLayer = struct {
    norm1_w: []const f32,
    norm1_b: []const f32,
    norm2_w: []const f32,
    norm2_b: []const f32,
    in_proj_w: []const f32,
    in_proj_b: []const f32,
    out_proj_w: []const f32,
    out_proj_b: []const f32,
    lin1_w: []const f32,
    lin1_b: []const f32,
    lin2_w: []const f32,
    lin2_b: []const f32,
};

const Model = struct {
    emb_raw: []const u8, // F16 [256000, 768]
    emb_norm: []const f32,
    layers: [NL]EncLayer,
    final_norm: []const f32,
    type_emb: []const f32, // [3, 768]
    head: [HEAD_LAYERS]HeadLayer,
    scorer0_w: []const f32,
    scorer0_b: []const f32,
    scorer1_w: []const f32,
    scorer1_b: []const f32,
    scorer3_w: []const f32,
    scorer3_b: []const f32,
    act0_w: []const f32,
    act0_b: []const f32,
    act2_w: []const f32,
    act2_b: []const f32,
    inv_freq: [32]f32,
    temperature: [3]f32,
};

const Scratch = struct {
    x: []f32,
    xn: []f32,
    q: []f32,
    k: []f32,
    v: []f32,
    qkv: []f32,
    ctx: []f32,
    mlp: []f32,
    mlp2: []f32,
    att: []f32,
    ff: []f32,
    feat: []f32,
    act_in: []f32,
};

const Cfg = struct {
    temperature: [3]f32 = .{ 1.0, 1.0, 1.0 },
    max_len: usize = MAX_LEN,
    head_max_len: usize = HEAD_MAX_LEN,
};

fn loadModel(alloc: Allocator, io: std.Io, dir: []const u8, cfg: *const Cfg) !struct { Model, []u8 } {
    const path = try std.fs.path.join(alloc, &.{ dir, "model.safetensors" });
    defer alloc.free(path);
    const data = try readFile(alloc, io, path);
    errdefer alloc.free(data);

    if (data.len < 8) return error.BadSafetensors;
    const hdr_len = std.mem.readInt(u64, data[0..8], .little);
    if (hdr_len + 8 > data.len) return error.BadSafetensors;
    const base: usize = 8 + hdr_len;

    var parsed = try std.json.parseFromSlice(std.json.Value, alloc, data[8 .. 8 + hdr_len], .{});
    defer parsed.deinit();

    var infos = std.StringHashMap(TensorInfo).init(alloc);
    defer infos.deinit();
    var it = parsed.value.object.iterator();
    while (it.next()) |e| {
        if (std.mem.eql(u8, e.key_ptr.*, "__metadata__")) continue;
        const o = e.value_ptr.object;
        const offs = o.get("data_offsets").?.array.items;
        const s: usize = @intCast(offs[0].integer);
        const en: usize = @intCast(offs[1].integer);
        try infos.put(e.key_ptr.*, .{ .dtype = o.get("dtype").?.string, .offset = base + s, .len = en - s });
    }

    var m: Model = undefined;
    m.emb_raw = try tensorRaw(&infos, data, "encoder.embeddings.tok_embeddings.weight");
    m.emb_norm = try tensorF32(alloc, &infos, data, "encoder.embeddings.norm.weight");
    m.final_norm = try tensorF32(alloc, &infos, data, "encoder.final_norm.weight");
    m.type_emb = try tensorF32(alloc, &infos, data, "type_emb.weight");

    var nm: [160]u8 = undefined;
    for (0..NL) |l| {
        const pre = try std.fmt.bufPrint(&nm, "encoder.layers.{d}.", .{l});
        var an: ?[]const f32 = null;
        if (l > 0) an = try tensorF32(alloc, &infos, data, try std.fmt.bufPrint(&nm, "{s}attn_norm.weight", .{pre}));
        m.layers[l] = .{
            .wqkv = try tensorF32(alloc, &infos, data, try std.fmt.bufPrint(&nm, "{s}attn.Wqkv.weight", .{pre})),
            .wo = try tensorF32(alloc, &infos, data, try std.fmt.bufPrint(&nm, "{s}attn.Wo.weight", .{pre})),
            .wi = try tensorF32(alloc, &infos, data, try std.fmt.bufPrint(&nm, "{s}mlp.Wi.weight", .{pre})),
            .wo2 = try tensorF32(alloc, &infos, data, try std.fmt.bufPrint(&nm, "{s}mlp.Wo.weight", .{pre})),
            .attn_norm = an,
            .mlp_norm = try tensorF32(alloc, &infos, data, try std.fmt.bufPrint(&nm, "{s}mlp_norm.weight", .{pre})),
            // config: layer_types[i] == "full_attention" iff i % global_attn_every_n_layers == 0 (3)
            .sliding = (l % 3) != 0,
        };
    }

    for (0..HEAD_LAYERS) |l| {
        const pre = try std.fmt.bufPrint(&nm, "head.layers.{d}.", .{l});
        m.head[l] = .{
            .norm1_w = try tensorF32(alloc, &infos, data, try std.fmt.bufPrint(&nm, "{s}norm1.weight", .{pre})),
            .norm1_b = try tensorF32(alloc, &infos, data, try std.fmt.bufPrint(&nm, "{s}norm1.bias", .{pre})),
            .norm2_w = try tensorF32(alloc, &infos, data, try std.fmt.bufPrint(&nm, "{s}norm2.weight", .{pre})),
            .norm2_b = try tensorF32(alloc, &infos, data, try std.fmt.bufPrint(&nm, "{s}norm2.bias", .{pre})),
            .in_proj_w = try tensorF32(alloc, &infos, data, try std.fmt.bufPrint(&nm, "{s}self_attn.in_proj_weight", .{pre})),
            .in_proj_b = try tensorF32(alloc, &infos, data, try std.fmt.bufPrint(&nm, "{s}self_attn.in_proj_bias", .{pre})),
            .out_proj_w = try tensorF32(alloc, &infos, data, try std.fmt.bufPrint(&nm, "{s}self_attn.out_proj.weight", .{pre})),
            .out_proj_b = try tensorF32(alloc, &infos, data, try std.fmt.bufPrint(&nm, "{s}self_attn.out_proj.bias", .{pre})),
            .lin1_w = try tensorF32(alloc, &infos, data, try std.fmt.bufPrint(&nm, "{s}linear1.weight", .{pre})),
            .lin1_b = try tensorF32(alloc, &infos, data, try std.fmt.bufPrint(&nm, "{s}linear1.bias", .{pre})),
            .lin2_w = try tensorF32(alloc, &infos, data, try std.fmt.bufPrint(&nm, "{s}linear2.weight", .{pre})),
            .lin2_b = try tensorF32(alloc, &infos, data, try std.fmt.bufPrint(&nm, "{s}linear2.bias", .{pre})),
        };
    }

    m.scorer0_w = try tensorF32(alloc, &infos, data, "scorer.0.weight");
    m.scorer0_b = try tensorF32(alloc, &infos, data, "scorer.0.bias");
    m.scorer1_w = try tensorF32(alloc, &infos, data, "scorer.1.weight");
    m.scorer1_b = try tensorF32(alloc, &infos, data, "scorer.1.bias");
    m.scorer3_w = try tensorF32(alloc, &infos, data, "scorer.3.weight");
    m.scorer3_b = try tensorF32(alloc, &infos, data, "scorer.3.bias");
    m.act0_w = try tensorF32(alloc, &infos, data, "act_head.0.weight");
    m.act0_b = try tensorF32(alloc, &infos, data, "act_head.0.bias");
    m.act2_w = try tensorF32(alloc, &infos, data, "act_head.2.weight");
    m.act2_b = try tensorF32(alloc, &infos, data, "act_head.2.bias");

    for (0..32) |j| {
        m.inv_freq[j] = 1.0 / std.math.pow(f32, ROPE_THETA, @as(f32, @floatFromInt(j)) / 32.0);
    }
    m.temperature = cfg.temperature;
    return .{ m, data };
}

// --------------------------------------------------------------------------- attention
const AttJob = struct {
    q: []const f32,
    k: []const f32,
    v: []const f32,
    out: []f32, // [L][D]
    att: []f32, // [L][stride]
    L: usize,
    stride: usize,
    win: ?usize,
};

fn attWorker(ctx: *anyopaque, lo: usize, hi: usize) void {
    const a: *AttJob = @ptrCast(@alignCast(ctx));
    const L = a.L;
    const scale: f32 = 1.0 / @sqrt(@as(f32, HD));
    for (lo..hi) |i| {
        const row = a.out[i * D ..][0..D];
        @memset(row, 0);
        var klo: usize = 0;
        var khi: usize = L;
        var off: usize = 0;
        if (a.win) |w| {
            klo = if (i >= w) i - w else 0;
            khi = @min(L, i + w + 1);
            off = if (i >= w) 0 else w - i;
        }
        const rowp = a.att[i * a.stride + off ..];
        for (0..H) |h| {
            const qo = i * D + h * HD;
            var mx: f32 = -std.math.inf(f32);
            for (klo..khi) |j| {
                const ko = j * D + h * HD;
                var s: f32 = 0;
                for (0..HD) |d| s += a.q[qo + d] * a.k[ko + d];
                s *= scale;
                rowp[j - klo] = s;
                mx = @max(mx, s);
            }
            var sum: f32 = 0;
            const cnt = khi - klo;
            for (0..cnt) |t| {
                rowp[t] = @exp(rowp[t] - mx);
                sum += rowp[t];
            }
            for (0..cnt) |t| rowp[t] /= sum;
            for (0..cnt) |t| {
                const p = rowp[t];
                if (p == 0) continue;
                const vo = (klo + t) * D + h * HD;
                const pv: @Vector(8, f32) = @splat(p);
                const base = h * HD;
                var c: usize = 0;
                while (c < HD) : (c += 8) {
                    var rv: @Vector(8, f32) = @bitCast(row[base + c ..][0..8].*);
                    rv += pv * @as(@Vector(8, f32), @bitCast(a.v[vo + c ..][0..8].*));
                    row[base + c ..][0..8].* = @as([8]f32, @bitCast(rv));
                }
            }
        }
    }
}

const RopeCtx = struct {
    x: []f32,
    inv: *const [32]f32,
};

fn ropeJob(ctx: *anyopaque, lo: usize, hi: usize) void {
    const m: *RopeCtx = @ptrCast(@alignCast(ctx));
    const inv = m.inv;
    for (lo..hi) |p| {
        const pos: f32 = @floatFromInt(p);
        var cs: [64]f32 = undefined;
        var sn: [64]f32 = undefined;
        for (0..32) |j| {
            const f = inv[j] * pos;
            cs[j] = @cos(f);
            sn[j] = @sin(f);
            cs[j + 32] = cs[j];
            sn[j + 32] = sn[j];
        }
        const o = p * D;
        for (0..H) |h| {
            const b = o + h * HD;
            for (0..32) |d| {
                const x0 = m.x[b + d];
                const x1 = m.x[b + d + 32];
                m.x[b + d] = x0 * cs[d] - x1 * sn[d];
                m.x[b + d + 32] = x1 * cs[d + 32] + x0 * sn[d + 32];
            }
        }
    }
}

fn ropeApply(alloc: Allocator, x: []f32, L: usize, inv: *const [32]f32) void {
    var m = RopeCtx{ .x = x, .inv = inv };
    parallel(alloc, L, &m, ropeJob);
}

// --------------------------------------------------------------------------- forward
const FwdCtx = struct {
    m: *const Model,
    s: *Scratch,
    ids: []const u32,
    L: usize,
    qtype: u32,
    markers: []const u32,
    logits: []f32,
    act: [2]f32,
    pooled: []f32,
    feat: []f32,
    act_in: []f32,
};

fn embedJob(ctx: *anyopaque, lo: usize, hi: usize) void {
    const f: *FwdCtx = @ptrCast(@alignCast(ctx));
    for (lo..hi) |i| {
        const off: usize = @as(usize, f.ids[i]) * D;
        const dst = f.s.x[i * D ..][0..D];
        for (0..D) |d| dst[d] = f16at(f.m.emb_raw, off + d);
    }
}

fn normJob(ctx: *anyopaque, lo: usize, hi: usize) void {
    const f: *FwdCtx = @ptrCast(@alignCast(ctx));
    for (lo..hi) |i| layerNorm(f.s.x[i * D ..][0..D], f.m.emb_norm, null);
}

fn mlpJob(ctx: *anyopaque, lo: usize, hi: usize) void {
    const f: *FwdCtx = @ptrCast(@alignCast(ctx));
    for (lo..hi) |i| {
        for (0..INTER) |j| {
            const a = f.s.mlp[i * 2 * INTER + j];
            const g = f.s.mlp[i * 2 * INTER + INTER + j];
            f.s.mlp2[i * INTER + j] = gelu(a) * g;
        }
    }
}

fn addResJob(ctx: *anyopaque, lo: usize, hi: usize) void {
    const f: *FwdCtx = @ptrCast(@alignCast(ctx));
    for (lo..hi) |i| {
        const dst = f.s.x[i * D ..][0..D];
        const src = f.s.xn[i * D ..][0..D];
        for (0..D) |d| dst[d] += src[d];
    }
}

fn reluJob(ctx: *anyopaque, lo: usize, hi: usize) void {
    const f: *FwdCtx = @ptrCast(@alignCast(ctx));
    for (lo..hi) |i| {
        const row = f.s.ff[i * HEAD_FF ..][0..HEAD_FF];
        for (row) |*v| v.* = if (v.* > 0) v.* else 0;
    }
}

const NormCtx = struct {
    x: []const f32,
    xn: []f32,
    w: []const f32,
    b: ?[]const f32,
};

/// xn[i] = LayerNorm(x[i])  -- x is left untouched (pre-norm, residual must survive)
fn normIntoJob(ctx: *anyopaque, lo: usize, hi: usize) void {
    const c: *NormCtx = @ptrCast(@alignCast(ctx));
    for (lo..hi) |i| {
        const dst = c.xn[i * D ..][0..D];
        @memcpy(dst, c.x[i * D ..][0..D]);
        layerNorm(dst, c.w, c.b);
    }
}

fn normInto(alloc: Allocator, x: []const f32, xn: []f32, L: usize, w: []const f32, b: ?[]const f32) void {
    var c = NormCtx{ .x = x, .xn = xn, .w = w, .b = b };
    parallel(alloc, L, &c, normIntoJob);
}

fn typeEmbJob(ctx: *anyopaque, lo: usize, hi: usize) void {
    const f: *FwdCtx = @ptrCast(@alignCast(ctx));
    const te = f.m.type_emb[@as(usize, f.qtype) * D ..][0..D];
    for (lo..hi) |i| {
        const row = f.s.x[i * D ..][0..D];
        for (0..D) |d| row[d] += te[d];
    }
}

fn splitQkv(q: []f32, k: []f32, v: []f32, qkv: []const f32, L: usize) void {
    for (0..L) |i| {
        const src = qkv[i * 2304 ..];
        @memcpy(q[i * D ..][0..D], src[0..D]);
        @memcpy(k[i * D ..][0..D], src[D .. 2 * D]);
        @memcpy(v[i * D ..][0..D], src[2 * D .. 3 * D]);
    }
}

fn forward(alloc: Allocator, f: *FwdCtx) void {
    const m = f.m;
    const s = f.s;
    const L = f.L;

    // ---- embeddings + norm (layer 0 has no attn_norm; embeddings.norm is folded in)
    parallel(alloc, L, f, embedJob);
    parallel(alloc, L, f, normJob);
    const dumping = g_dump_stats and g_fwd_count == 0;
    if (dumping) {
        std.debug.print("[dump] meta L={d} qtype={d}\n", .{ L, f.qtype });
        dumpIds(f.ids);
        dumpStats("emb", s.x, L);
    }
    for (0..NL) |l| {
        const lay = &m.layers[l];
        var attn_in: []const f32 = s.x;
        if (lay.attn_norm) |an| {
            normInto(alloc, s.x, s.xn, L, an, null);
            attn_in = s.xn;
        }
        linear(alloc, s.qkv, attn_in, lay.wqkv, null, L, 3 * D, D);
        splitQkv(s.q, s.k, s.v, s.qkv, L);
        ropeApply(alloc, s.q, L, &m.inv_freq);
        ropeApply(alloc, s.k, L, &m.inv_freq);

        var aj = AttJob{
            .q = s.q,
            .k = s.k,
            .v = s.v,
            .out = s.ctx,
            .att = s.att,
            .L = L,
            .stride = if (lay.sliding) 2 * WINDOW + 1 else L,
            .win = if (lay.sliding) WINDOW else null,
        };
        parallel(alloc, L, &aj, attWorker);
        linear(alloc, s.xn, s.ctx, lay.wo, null, L, D, D);
        parallel(alloc, L, f, addResJob);

        normInto(alloc, s.x, s.xn, L, lay.mlp_norm, null);
        linear(alloc, s.mlp, s.xn, lay.wi, null, L, 2 * INTER, D);
        parallel(alloc, L, f, mlpJob);
        linear(alloc, s.xn, s.mlp2, lay.wo2, null, L, D, INTER);
        parallel(alloc, L, f, addResJob);
        if (dumping) {
            var tag: [8]u8 = undefined;
            const t = std.fmt.bufPrint(&tag, "l{d}", .{l}) catch "l?";
            dumpStats(t, s.x, L);
        }
    }

    for (0..L) |i| layerNorm(s.x[i * D ..][0..D], m.final_norm, null);
    if (dumping) dumpStats("final", s.x, L);
    parallel(alloc, L, f, typeEmbJob);
    if (dumping) dumpStats("temb", s.x, L);

    // ---- decision head: 2 pre-norm transformer layers, bidirectional
    for (0..HEAD_LAYERS) |l| {
        const hl = &m.head[l];
        normInto(alloc, s.x, s.xn, L, hl.norm1_w, hl.norm1_b);
        linear(alloc, s.qkv, s.xn, hl.in_proj_w, hl.in_proj_b, L, 3 * D, D);
        splitQkv(s.q, s.k, s.v, s.qkv, L);
        var aj = AttJob{
            .q = s.q,
            .k = s.k,
            .v = s.v,
            .out = s.ctx,
            .att = s.att,
            .L = L,
            .stride = L,
            .win = null,
        };
        parallel(alloc, L, &aj, attWorker);
        linear(alloc, s.xn, s.ctx, hl.out_proj_w, hl.out_proj_b, L, D, D);
        parallel(alloc, L, f, addResJob);

        normInto(alloc, s.x, s.xn, L, hl.norm2_w, hl.norm2_b);
        linear(alloc, s.ff, s.xn, hl.lin1_w, hl.lin1_b, L, HEAD_FF, D);
        parallel(alloc, L, f, reluJob);
        linear(alloc, s.xn, s.ff, hl.lin2_w, hl.lin2_b, L, D, HEAD_FF);
        parallel(alloc, L, f, addResJob);
        if (dumping) {
            var tag: [8]u8 = undefined;
            const t = std.fmt.bufPrint(&tag, "h{d}", .{l}) catch "h?";
            dumpStats(t, s.x, L);
        }
    }

    @memcpy(f.pooled, s.x[0..D]);

    // ---- option markers -> scorer
    const K = f.markers.len;
    for (0..K) |r| {
        var mvec: [D]f32 = undefined;
        @memcpy(&mvec, s.x[@as(usize, f.markers[r]) * D ..][0..D]);
        layerNorm(&mvec, m.scorer0_w, m.scorer0_b);
        var h1: [D]f32 = undefined;
        rowDot(&h1, &mvec, m.scorer1_w);
        for (0..D) |d| h1[d] = gelu(h1[d] + m.scorer1_b[d]);
        f.logits[r] = rowDot1(&h1, m.scorer3_w) + m.scorer3_b[0];
    }

    if (dumping) {
        std.debug.print("[dump] mark", .{});
        for (f.markers) |v| std.debug.print(" {d}", .{v});
        std.debug.print("\n[dump] logit", .{});
        for (f.logits[0..K]) |v| std.debug.print(" {d:.5}", .{v});
        std.debug.print("\n", .{});
    }

    if (g_debug) {
        std.debug.print("[dbg] markers=", .{});
        for (f.markers) |mm| std.debug.print("{d} ", .{mm});
        std.debug.print("\n[dbg] logits=", .{});
        for (f.logits[0..K]) |v| std.debug.print("{d:.5} ", .{v});
        std.debug.print("\n", .{});
    }

    // ---- act / escalate head
    const p = f.feat[0..K];
    @memcpy(p, f.logits[0..K]);
    softmax(p);
    const kk: f32 = @floatFromInt(@max(K, @as(usize, 2)));
    var ent: f32 = 0;
    for (p) |v| {
        if (v > 1e-12) ent -= v * @log(v);
    }
    ent /= @log(kk);
    var t0: f32 = 0;
    var t1: f32 = 0;
    for (p) |v| {
        if (v > t0) {
            t1 = t0;
            t0 = v;
        } else if (v > t1) t1 = v;
    }
    const feats = [4]f32{ t0, t0 - t1, ent, kk / 255.0 };
    @memcpy(f.act_in[0..D], f.pooled);
    @memcpy(f.act_in[D .. D + 4], &feats);
    var h256: [256]f32 = undefined;
    rowDot(&h256, f.act_in, m.act0_w);
    for (0..256) |i| h256[i] = gelu(h256[i] + m.act0_b[i]);
    var a2: [2]f32 = undefined;
    for (0..2) |i| a2[i] = rowDot1(&h256, m.act2_w[i * 256 ..][0..256]) + m.act2_b[i];
    softmax(&a2);
    f.act = a2;
    g_fwd_count += 1;
}

// --------------------------------------------------------------------------- sequence
fn replaceMask(alloc: Allocator, s: []const u8) ![]const u8 {
    if (std.mem.indexOf(u8, s, MASK_TOKEN) == null) return s;
    var out = AList(u8).init(alloc);
    var i: usize = 0;
    while (i < s.len) {
        if (std.mem.startsWith(u8, s[i..], MASK_TOKEN)) {
            try out.append(' ');
            i += MASK_TOKEN.len;
        } else {
            try out.append(s[i]);
            i += 1;
        }
    }
    return out.toOwnedSlice();
}

const Seq = struct {
    ids: []u32,
    markers: []u32,
    qtype: u32,
};

fn buildSequence(alloc: Allocator, tok: *const Tokenizer, state: []const u8, qtype_name: []const u8, ins: []const u8, opts: []const []const u8, max_len: usize, head_max_len: usize) !Seq {
    const ins2 = try replaceMask(alloc, ins);
    const head_text = try std.fmt.allocPrint(alloc, "{s} question: {s}", .{ qtype_name, ins2 });
    const head_ids = try encodeAll(tok, alloc, head_text);

    var opt_ids = AList([]u32).init(alloc);
    for (opts) |o| {
        const o2 = try replaceMask(alloc, o);
        const t = try std.fmt.allocPrint(alloc, " {s}", .{o2});
        var e = try encodeAll(tok, alloc, t);
        if (e.len > 48) e = e[0..48];
        const full = try alloc.alloc(u32, e.len + 1);
        full[0] = MASK_ID;
        @memcpy(full[1..], e);
        try opt_ids.append(full);
    }

    var total: usize = 0;
    for (opt_ids.items) |o| total += o.len;
    var opt_budget: i64 = @as(i64, @intCast(head_max_len)) - @as(i64, @intCast(total));
    if (opt_budget < 16) {
        const per: usize = @max(4, (head_max_len - 16) / @max(1, opt_ids.items.len));
        for (opt_ids.items) |*o| {
            if (o.*.len > per) o.* = o.*[0..per];
        }
        total = 0;
        for (opt_ids.items) |o| total += o.len;
        opt_budget = @as(i64, @intCast(head_max_len)) - @as(i64, @intCast(total));
    }
    const head_keep: usize = @max(8, @as(usize, @intCast(@max(opt_budget, 0))));
    var head_ids2 = head_ids;
    if (head_ids2.len > head_keep) head_ids2 = head_ids2[0..head_keep];

    var ids = AList(u32).init(alloc);
    try ids.append(CLS_ID);
    try ids.appendSlice(head_ids2);
    try ids.append(SEP_ID);
    var markers = AList(u32).init(alloc);
    for (opt_ids.items) |o| {
        try markers.append(@intCast(ids.items.len));
        try ids.appendSlice(o);
    }
    try ids.append(SEP_ID);

    const room: i64 = @as(i64, @intCast(max_len)) - @as(i64, @intCast(ids.items.len)) - 1;
    if (room > 0) {
        const st = try replaceMask(alloc, state);
        var st_ids = try encodeAll(tok, alloc, st);
        if (st_ids.len > room) st_ids = st_ids[0..@intCast(room)];
        try ids.appendSlice(st_ids);
    }
    try ids.append(SEP_ID);

    var final_ids = ids.items;
    if (final_ids.len > max_len) final_ids = final_ids[0..max_len];

    var cnt: usize = 0;
    for (markers.items) |v| {
        if (v < max_len) cnt += 1;
    }
    const mk2 = try alloc.alloc(u32, cnt);
    var c: usize = 0;
    for (markers.items) |v| {
        if (v < max_len) {
            mk2[c] = v;
            c += 1;
        }
    }
    return .{ .ids = final_ids, .markers = mk2, .qtype = 0 };
}

// --------------------------------------------------------------------------- questions
fn qtypeId(name: []const u8) !u32 {
    if (std.mem.eql(u8, name, "choice")) return 0;
    if (std.mem.eql(u8, name, "score")) return 1;
    if (std.mem.eql(u8, name, "noul")) return 2;
    return error.BadQuestionType;
}

const Question = struct {
    name: []const u8,
    qtype: u32,
    type_name: []const u8,
    ins: []const u8,
    opts: [][]const u8,
    keys: [][]const u8,
};

fn renderCriterion(alloc: Allocator, v: std.json.Value) ![]const u8 {
    return switch (v) {
        .string => |s| s,
        .null => "",
        else => try std.json.Stringify.valueAlloc(alloc, v, .{}),
    };
}

fn loadQuestions(alloc: Allocator, root: std.json.Value) ![]Question {
    const qs = root.object.get("questions").?.object;
    var list = AList(Question).init(alloc);
    var it = qs.iterator();
    while (it.next()) |e| {
        const q = e.value_ptr.object;
        const tn = q.get("type").?.string;
        const qt = try qtypeId(tn);
        const ins_v = q.get("instructions").?;
        const ins: []const u8 = if (ins_v == .string) ins_v.string else try std.json.Stringify.valueAlloc(alloc, ins_v, .{});

        var opts = AList([]const u8).init(alloc);
        var keys = AList([]const u8).init(alloc);
        const crit = q.get("criteria");
        if (qt == 0) {
            if (crit) |c| {
                if (c == .object) {
                    var ci = c.object.iterator();
                    while (ci.next()) |ce| {
                        const k = ce.key_ptr.*;
                        const v = ce.value_ptr.*;
                        const plain = (v == .null) or (v == .string and v.string.len == 0);
                        try keys.append(k);
                        try opts.append(if (plain) k else try std.fmt.allocPrint(alloc, "{s}: {s}", .{ k, try renderCriterion(alloc, v) }));
                    }
                } else if (c == .array) {
                    for (c.array.items) |v| {
                        try keys.append(v.string);
                        try opts.append(v.string);
                    }
                }
            }
        } else if (qt == 1) {
            if (crit) |c| {
                for (c.array.items, 0..) |v, idx| {
                    try keys.append(v.string);
                    try opts.append(try std.fmt.allocPrint(alloc, "level {d}: {s}", .{ idx, try renderCriterion(alloc, v) }));
                }
            }
        } else {
            var fc: []const u8 = "no, the statement does not hold";
            var tc: []const u8 = "yes, the statement holds";
            if (crit) |c| {
                if (c == .object) {
                    if (c.object.get("false")) |v| {
                        if (!(v == .null) and !(v == .string and v.string.len == 0)) fc = try renderCriterion(alloc, v);
                    }
                    if (c.object.get("true")) |v| {
                        if (!(v == .null) and !(v == .string and v.string.len == 0)) tc = try renderCriterion(alloc, v);
                    }
                }
            }
            try keys.append("false");
            try keys.append("true");
            try opts.append(try std.fmt.allocPrint(alloc, "false: {s}", .{fc}));
            try opts.append(try std.fmt.allocPrint(alloc, "true: {s}", .{tc}));
        }
        try list.append(.{
            .name = e.key_ptr.*,
            .qtype = qt,
            .type_name = tn,
            .ins = ins,
            .opts = try opts.toOwnedSlice(),
            .keys = try keys.toOwnedSlice(),
        });
    }
    return list.toOwnedSlice();
}

fn confidenceFromProbs(p: []const f32, k: usize) f32 {
    if (k < 2) return 1.0;
    var ent: f32 = 0;
    for (p[0..k]) |v| {
        if (v > 1e-12) ent -= v * @log(v);
    }
    return @min(1.0, @max(0.0, 1.0 - ent / @log(@as(f32, @floatFromInt(k)))));
}

// --------------------------------------------------------------------------- snake glue
//
// Wires the model into src/snake.zig: the game asks for a direction, we hand it the
// board text as the `state` of a 4-way `choice` question and take the argmax.
// `--policy greedy|random` uses the baselines built into snake.zig instead.

const snake = @import("snake.zig");

const Policy = enum { model, greedy, random };

const SNAKE_INS = "Which move keeps the snake alive and reaches the food?";

/// Model plus scratch buffers, kept alive across moves so a game never reloads 614 MB.
const ModelChooser = struct {
    gpa: Allocator,
    m: *const Model,
    tok: *const Tokenizer,
    cfg: *const Cfg,
    s: Scratch,
    logits: []f32,
    pooled: []f32,
    ms: i64 = 0,
    last: WebSession.LastMove = .{ .dir = .up, .probs = .{ 0.25, 0.25, 0.25, 0.25 }, .conf = 0, .act = 0, .tokens = 0, .ms = 0 },

    fn init(gpa: Allocator, m: *const Model, tok: *const Tokenizer, cfg: *const Cfg) !ModelChooser {
        const L = cfg.max_len;
        return .{
            .gpa = gpa,
            .m = m,
            .tok = tok,
            .cfg = cfg,
            .s = .{
                .x = try gpa.alloc(f32, L * D),
                .xn = try gpa.alloc(f32, L * D),
                .q = try gpa.alloc(f32, L * D),
                .k = try gpa.alloc(f32, L * D),
                .v = try gpa.alloc(f32, L * D),
                .qkv = try gpa.alloc(f32, L * 3 * D),
                .ctx = try gpa.alloc(f32, L * D),
                .mlp = try gpa.alloc(f32, L * 2 * INTER),
                .mlp2 = try gpa.alloc(f32, L * INTER),
                .att = try gpa.alloc(f32, L * @max(L, 2 * WINDOW + 1)),
                .ff = try gpa.alloc(f32, L * HEAD_FF),
                .feat = try gpa.alloc(f32, 8),
                .act_in = try gpa.alloc(f32, D + 4),
            },
            .logits = try gpa.alloc(f32, 8),
            .pooled = try gpa.alloc(f32, D),
        };
    }

    fn moveFn(ctx: *anyopaque, io: std.Io, g: *snake.Snake, status: *std.Io.Writer) anyerror!snake.Dir {
        const self: *ModelChooser = @ptrCast(@alignCast(ctx));

        var arena = std.heap.ArenaAllocator.init(self.gpa);
        defer arena.deinit();
        const a = arena.allocator();

        const text = try g.stateText(a);
        var opts: [4][]const u8 = undefined;
        try g.optionTexts(a, &opts);

        const sq = try buildSequence(a, self.tok, text, "choice", SNAKE_INS, &opts, self.cfg.max_len, self.cfg.head_max_len);
        const t0 = nowMs(io);
        var f = FwdCtx{
            .m = self.m,
            .s = &self.s,
            .ids = sq.ids,
            .L = sq.ids.len,
            .qtype = 0,
            .markers = sq.markers,
            .logits = self.logits,
            .act = .{ 0, 0 },
            .pooled = self.pooled,
            .feat = self.s.feat,
            .act_in = self.s.act_in,
        };
        forward(self.gpa, &f);
        self.ms = nowMs(io) - t0;

        const k = @min(sq.markers.len, 4);
        const temp = @max(1e-3, self.m.temperature[0]);
        var p: [4]f32 = .{ 0, 0, 0, 0 };
        var mx: f32 = -std.math.inf(f32);
        for (0..k) |r| {
            p[r] = self.logits[r] / temp;
            mx = @max(mx, p[r]);
        }
        var sum: f32 = 0;
        for (0..k) |r| {
            p[r] = @exp(p[r] - mx);
            sum += p[r];
        }
        for (0..k) |r| p[r] /= sum;

        var best: usize = 0;
        for (0..k) |r| {
            if (p[r] > p[best]) best = r;
        }
        const conf = confidenceFromProbs(p[0..k], k);
        self.last = .{ .dir = snake.DIRS[best], .probs = p, .conf = conf, .act = f.act[0], .tokens = sq.ids.len, .ms = self.ms };
        try status.print("model: chose {s}  (up {d:.3} / down {d:.3} / left {d:.3} / right {d:.3})  conf {d:.3}  act {d:.3}  {d} tok  {d} ms\n", .{
            snake.DIRS[best].label(), p[0], p[1], p[2], p[3], conf, f.act[0], sq.ids.len, self.ms,
        });
        return snake.DIRS[best];
    }
};

// --------------------------------------------------------------------------- web ui
//
// `zig-out/bin/laya.exe --serve` starts a tiny HTTP server; the page in
// src/web/index.html draws the board and calls /api/step once per move. The game
// still lives here in Zig, so the browser shows exactly what the CLI shows.

const server = @import("server.zig");
const INDEX_HTML = @embedFile("web/index.html");

const WebSession = struct {
    gpa: Allocator,
    m: *const Model,
    tok: *const Tokenizer,
    cfg: *const Cfg,
    mc: *ModelChooser,
    game: snake.Snake = undefined,
    has_game: bool = false,
    policy: Policy = .model,
    base: snake.Chooser = undefined,
    last: ?LastMove = null,
    fatal: [4]bool = .{ false, false, false, false },
    prompt: AList(u8),

    const LastMove = struct {
        dir: snake.Dir,
        probs: [4]f32,
        conf: f32,
        act: f32,
        tokens: usize,
        ms: i64,
    };

    fn newGame(self: *WebSession, size: usize, seed: u64) !void {
        if (self.has_game) self.game.deinit();
        self.game = try snake.Snake.init(self.gpa, size, seed);
        self.has_game = true;
        self.last = null;
        try self.capture(&self.game);
    }

    fn chooser(self: *WebSession) snake.Chooser {
        return .{ .ctx = self, .moveFn = moveFn };
    }

    /// Record what the board looked like before the move (for the UI): the exact
    /// prompt text and which directions would have ended the game.
    fn capture(self: *WebSession, g: *snake.Snake) !void {
        self.prompt.clearRetainingCapacity();
        const txt = try g.stateText(self.gpa);
        defer self.gpa.free(txt);
        try self.prompt.appendSlice(txt);
        const h = g.head();
        for (snake.DIRS, 0..) |d, i| {
            const nb = g.neighbour(h, d);
            self.fatal[i] = nb == null or g.occupied[nb.?[0] * g.size + nb.?[1]];
        }
    }

    fn moveFn(ctx: *anyopaque, io: std.Io, g: *snake.Snake, status: *std.Io.Writer) anyerror!snake.Dir {
        const self: *WebSession = @ptrCast(@alignCast(ctx));
        try self.capture(g);
        switch (self.policy) {
            .model => {
                const d = try ModelChooser.moveFn(self.mc, io, g, status);
                self.last = self.mc.last;
                return d;
            },
            .greedy, .random => {
                const d = try self.base.moveFn(self.base.ctx, io, g, status);
                // baselines are deterministic about the direction they pick
                var p: [4]f32 = .{ 0, 0, 0, 0 };
                p[@intFromEnum(d)] = 1;
                self.last = .{ .dir = d, .probs = p, .conf = 1, .act = 1, .tokens = 0, .ms = 0 };
                return d;
            },
        }
    }

    fn writeState(self: *WebSession, out: *std.Io.Writer, with_prompt: bool) !void {
        const g = &self.game;
        try out.print("{{\"size\":{d},\"score\":{d},\"step\":{d},\"alive\":{s},\"illegal\":{d},\"death\":\"{s}\",\"dir\":\"{s}\",\"food\":[{d},{d}],\"snake\":[", .{
            g.size,
            g.score,
            g.steps,
            if (g.alive) "true" else "false",
            g.illegal,
            if (g.alive) "" else if (g.hit_wall) "wall" else "self",
            g.dir.label(),
            g.food[0],
            g.food[1],
        });
        for (g.body.items, 0..) |c, i| {
            if (i > 0) try out.writeByte(',');
            try out.print("[{d},{d}]", .{ c[0], c[1] });
        }
        try out.writeAll("],\"fatal\":[");
        for (self.fatal, 0..) |f, i| {
            if (i > 0) try out.writeByte(',');
            try out.writeAll(if (f) "true" else "false");
        }
        try out.writeAll("],\"last\":");
        if (self.last) |l| {
            try out.print("{{\"dir\":\"{s}\",\"probs\":[{d:.5},{d:.5},{d:.5},{d:.5}],\"conf\":{d:.5},\"act\":{d:.5},\"tokens\":{d},\"ms\":{d}}}", .{
                l.dir.label(), l.probs[0], l.probs[1], l.probs[2], l.probs[3], l.conf, l.act, l.tokens, l.ms,
            });
        } else {
            try out.writeAll("null");
        }
        if (with_prompt) {
            try out.writeAll(",\"prompt\":");
            try server.jsonString(out, self.prompt.items);
        }
        try out.writeAll("}");
    }
};

fn webHandle(ctx: *anyopaque, io: std.Io, req: server.Request, out: *std.Io.Writer) anyerror!server.Response {
    const self: *WebSession = @ptrCast(@alignCast(ctx));

    if (std.mem.eql(u8, req.path, "/") or std.mem.eql(u8, req.path, "/index.html")) {
        return .{ .content_type = "text/html; charset=utf-8", .body = INDEX_HTML };
    }

    if (std.mem.eql(u8, req.path, "/api/new")) {
        const size: usize = @intCast(@max(6, @min(req.intParam("size", 10), 24)));
        const seed: u64 = @intCast(@max(0, req.intParam("seed", 12345)));
        const pol = req.param("policy") orelse "model";
        self.policy = if (std.mem.eql(u8, pol, "greedy"))
            .greedy
        else if (std.mem.eql(u8, pol, "random"))
            .random
        else
            .model;
        self.base = if (self.policy == .random) snake.randomChooser() else snake.greedyChooser();
        try self.newGame(size, seed);
        try self.writeState(out, true);
        return .{};
    }

    if (std.mem.eql(u8, req.path, "/api/step")) {
        if (self.game.alive) {
            var scratch: std.Io.Writer.Allocating = .init(self.gpa);
            defer scratch.deinit();
            const d = try self.chooser().moveFn(self, io, &self.game, &scratch.writer);
            self.game.step(d);
        }
        try self.writeState(out, req.intParam("prompt", 1) != 0);
        return .{};
    }

    return .{ .status = 404, .body = "{\"error\":\"no route\"}" };
}

// --------------------------------------------------------------------------- io
fn nowMs(io: std.Io) i64 {
    return @intCast(@divTrunc(std.Io.Timestamp.now(io, .awake).nanoseconds, std.time.ns_per_ms));
}

fn readFile(alloc: Allocator, io: std.Io, path: []const u8) ![]u8 {
    const f = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer f.close(io);
    const st = try f.stat(io);
    const buf = try alloc.alloc(u8, @max(@as(usize, @intCast(st.size)), 1));
    const got = try f.readPositionalAll(io, buf, 0);
    return buf[0..got];
}

const DEMO_JSON =
    \\{"state":"मुझसे इनवॉइस 4411 के लिए दो बार शुल्क लिया गया। कृपया आज ही धनवापसी करें।",
    \\ "questions":{
    \\  "department":{"type":"choice","instructions":"Which team should handle `body`?",
    \\                 "criteria":{"billing":"invoices, payments, refunds","technical":"bugs and outages","sales":"pricing"}},
    \\  "refund_requested":{"type":"noul","instructions":"Does the sender ask for money back?"}}}
;

const builtin = @import("builtin");

extern "kernel32" fn SetConsoleOutputCP(wCodePageID: u32) callconv(.winapi) i32;
extern "kernel32" fn GetStdHandle(nStdHandle: u32) callconv(.winapi) ?*anyopaque;
extern "kernel32" fn GetConsoleMode(h: ?*anyopaque, mode: *u32) callconv(.winapi) i32;
extern "kernel32" fn SetConsoleMode(h: ?*anyopaque, mode: u32) callconv(.winapi) i32;

/// Turn on ANSI escape handling so the snake board can repaint in place.
/// Returns false when the console cannot do it (redirected output, old conhost),
/// in which case frames are simply appended instead.
fn enableAnsi() bool {
    if (builtin.os.tag != .windows) return true;
    const STD_OUTPUT_HANDLE: u32 = 0xFFFFFFF5; // (DWORD)-11
    const ENABLE_VIRTUAL_TERMINAL_PROCESSING: u32 = 0x0004;
    const h = GetStdHandle(STD_OUTPUT_HANDLE) orelse return false;
    var mode: u32 = 0;
    if (GetConsoleMode(h, &mode) == 0) return false;
    return SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING) != 0;
}

fn tokCheck(gpa: Allocator, io: std.Io, tok: *const Tokenizer, path: []const u8) !void {
    const data = try readFile(gpa, io, path);
    defer gpa.free(data);
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const aa = arena.allocator();
    var pass: usize = 0;
    var fail: usize = 0;
    var lines = std.mem.splitScalar(u8, data, '\n');
    while (lines.next()) |ln| {
        if (ln.len < 3) continue;
        var p = try std.json.parseFromSlice(std.json.Value, aa, ln, .{});
        defer p.deinit();
        const text = p.value.object.get("text").?.string;
        const exp = p.value.object.get("ids").?.array.items;
        const got = try encodeAll(tok, aa, text);
        var ok = got.len == exp.len;
        if (ok) {
            for (got, 0..) |g, i| {
                if (g != @as(u32, @intCast(exp[i].integer))) ok = false;
            }
        }
        if (ok) {
            pass += 1;
        } else {
            fail += 1;
            std.debug.print("MISMATCH {s}\n  got  {any}\n  want ", .{ text, got });
            for (exp) |e| std.debug.print("{d} ", .{e.integer});
            std.debug.print("\n", .{});
        }
    }
    std.debug.print("tokcheck: {d} pass, {d} fail\n", .{ pass, fail });
}

pub fn main(init: std.process.Init) !void {
    if (builtin.os.tag == .windows) _ = SetConsoleOutputCP(65001);

    const gpa = std.heap.page_allocator;
    const io = init.io;
    var argit = try std.process.Args.Iterator.initAllocator(init.minimal.args, gpa);
    defer argit.deinit();
    var argv = AList([]const u8).init(gpa);
    defer argv.deinit();
    while (argit.next()) |a| try argv.append(a);
    const args = argv.items;

    var dir: []const u8 = ".";
    var json_path: ?[]const u8 = null;
    var tokcheck: ?[]const u8 = null;
    var play = false;
    var serve = false;
    var selftest = false;
    var port: u16 = 8080;
    var s_policy: Policy = .model;
    var s_opts = snake.Config{};
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        if (std.mem.eql(u8, a, "--snake")) {
            play = true;
            continue;
        }
        if (std.mem.eql(u8, a, "--serve")) {
            serve = true;
            continue;
        }
        if (std.mem.eql(u8, a, "--prompt")) {
            s_opts.show_prompt = true;
            continue;
        }
        if (std.mem.eql(u8, a, "--debug")) {
            g_debug = true;
            continue;
        }
        if (std.mem.eql(u8, a, "--selftest")) {
            selftest = true;
            continue;
        }
        if (std.mem.eql(u8, a, "--dumpstats")) {
            g_dump_stats = true;
            continue;
        }
        const takes_value = std.mem.eql(u8, a, "--dir") or std.mem.eql(u8, a, "--json") or
            std.mem.eql(u8, a, "--threads") or std.mem.eql(u8, a, "--tokcheck") or
            std.mem.eql(u8, a, "--size") or std.mem.eql(u8, a, "--games") or
            std.mem.eql(u8, a, "--policy") or std.mem.eql(u8, a, "--delay") or
            std.mem.eql(u8, a, "--seed") or std.mem.eql(u8, a, "--max-steps") or
            std.mem.eql(u8, a, "--port");
        if (!takes_value or i + 1 >= args.len) continue;
        const key = a;
        i += 1;
        const v = args[i];
        if (std.mem.eql(u8, key, "--dir")) dir = v;
        if (std.mem.eql(u8, key, "--json")) json_path = v;
        if (std.mem.eql(u8, key, "--tokcheck")) tokcheck = v;
        if (std.mem.eql(u8, key, "--threads")) g_threads = std.fmt.parseInt(usize, v, 10) catch 4;
        if (std.mem.eql(u8, key, "--size")) s_opts.size = @max(6, @min(std.fmt.parseInt(usize, v, 10) catch 10, 24));
        if (std.mem.eql(u8, key, "--games")) s_opts.games = @max(1, std.fmt.parseInt(usize, v, 10) catch 1);
        if (std.mem.eql(u8, key, "--delay")) s_opts.delay_ms = std.fmt.parseInt(i64, v, 10) catch 0;
        if (std.mem.eql(u8, key, "--seed")) s_opts.seed = std.fmt.parseInt(u64, v, 10) catch 12345;
        if (std.mem.eql(u8, key, "--max-steps")) s_opts.max_steps = @max(1, std.fmt.parseInt(usize, v, 10) catch 400);
        if (std.mem.eql(u8, key, "--port")) port = std.fmt.parseInt(u16, v, 10) catch 8080;
        if (std.mem.eql(u8, key, "--policy")) {
            s_policy = if (std.mem.eql(u8, v, "greedy"))
                .greedy
            else if (std.mem.eql(u8, v, "random"))
                .random
            else
                .model;
        }
    }
    g_threads = @max(1, @min(g_threads, 64));
    if (g_threads == 4) {
        // measured: past ~8 threads the matmul is memory-latency bound and gets slower
        if (std.Thread.getCpuCount()) |n| g_threads = @max(1, @min(n, 8)) else |_| {}
    }

    var out_buf: [1 << 16]u8 = undefined;
    var fw = std.Io.File.stdout().writer(io, &out_buf);
    const w = &fw.interface;

    const t0 = nowMs(io);

    if (selftest) {
        try w.print("laya selftest (SIMD kernels vs f64)\n", .{});
        try selfTest(gpa, w);
        return;
    }

    // ---- config
    var cfg = Cfg{};
    {
        const cfg_path = try std.fs.path.join(gpa, &.{ dir, "rl_agent_config.json" });
        defer gpa.free(cfg_path);
        if (readFile(gpa, io, cfg_path)) |cd| {
            defer gpa.free(cd);
            var p = try std.json.parseFromSlice(std.json.Value, gpa, cd, .{});
            defer p.deinit();
            const o = p.value.object;
            if (o.get("temperature")) |tv| {
                for (tv.array.items, 0..) |v, j| {
                    if (j < 3) cfg.temperature[j] = switch (v) {
                        .integer => |x| @floatFromInt(x),
                        .float => |x| @floatCast(x),
                        else => 1.0,
                    };
                }
            }
            if (o.get("max_len")) |v| cfg.max_len = @intCast(v.integer);
            if (o.get("head_max_len")) |v| cfg.head_max_len = @intCast(v.integer);
        } else |_| {}
    }

    // ---- tokenizer (kept alive for the whole run)
    var tok_arena = std.heap.ArenaAllocator.init(gpa);
    defer tok_arena.deinit();
    const ta = tok_arena.allocator();
    const tok_path = try std.fs.path.join(gpa, &.{ dir, "tokenizer", "tokenizer.json" });
    defer gpa.free(tok_path);
    const tok = try loadTokenizer(ta, io, tok_path);
    const t1 = nowMs(io);
    try w.print("tokenizer: {d} vocab / {d} merges / {d} added ({d} ms)\n", .{ tok.vocab.count(), tok.merges.count(), tok.added.count(), t1 - t0 });
    try w.flush();

    if (tokcheck) |tp| {
        try tokCheck(gpa, io, &tok, tp);
        return;
    }

    // ---- weights
    const mr = try loadModel(gpa, io, dir, &cfg);
    const m = mr[0];
    const t2 = nowMs(io);
    try w.print("weights: {d} MB, {d} threads ({d} ms)\n", .{ mr[1].len / (1024 * 1024), g_threads, t2 - t1 });
    try w.flush();

    if (serve) {
        const mc = try gpa.create(ModelChooser);
        mc.* = try ModelChooser.init(gpa, &m, &tok, &cfg);
        var session = WebSession{
            .gpa = gpa,
            .m = &m,
            .tok = &tok,
            .cfg = &cfg,
            .mc = mc,
            .prompt = AList(u8).init(gpa),
        };
        try session.newGame(10, 12345);
        const base = snake.greedyChooser();
        session.base = base;
        server.serve(gpa, io, .{ .port = port, .log = w }, .{ .ctx = &session, .handleFn = webHandle }) catch |e| {
            try w.print("server stopped: {s}\n", .{@errorName(e)});
            try w.flush();
        };
        return;
    }

    if (play) {
        var mc = try ModelChooser.init(gpa, &m, &tok, &cfg);
        const chooser: snake.Chooser = switch (s_policy) {
            .model => .{ .ctx = &mc, .moveFn = ModelChooser.moveFn },
            .greedy => snake.greedyChooser(),
            .random => snake.randomChooser(),
        };
        const name = switch (s_policy) {
            .model => "model",
            .greedy => "greedy (baseline)",
            .random => "random (baseline)",
        };
        try snake.run(gpa, io, w, chooser, name, s_opts, s_policy == .model and enableAnsi());
        return;
    }

    // ---- questions
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const aa = arena.allocator();
    const json_text: []const u8 = if (json_path) |p| try readFile(aa, io, p) else DEMO_JSON;
    var parsed_q = try std.json.parseFromSlice(std.json.Value, aa, json_text, .{});
    const questions = try loadQuestions(aa, parsed_q.value);
    // A non-string state is JSON-serialized, matching upstream's serialize_state().
    // It used to fall back to "" here, so the upstream quickstart -- which passes
    // {"body": "..."} -- silently handed the model an empty state.
    const state: []const u8 = if (parsed_q.value.object.get("state")) |sv|
        (if (sv == .string) sv.string else try std.json.Stringify.valueAlloc(aa, sv, .{}))
    else
        "";

    try w.print("state: {s}\n", .{state});
    try w.flush();

    // ---- sequences
    const seqs = try aa.alloc(Seq, questions.len);
    var Lmax: usize = 1;
    var Kmax: usize = 1;
    for (questions, 0..) |q, qi| {
        seqs[qi] = try buildSequence(aa, &tok, state, q.type_name, q.ins, q.opts, cfg.max_len, cfg.head_max_len);
        seqs[qi].qtype = q.qtype;
        Lmax = @max(Lmax, seqs[qi].ids.len);
        Kmax = @max(Kmax, seqs[qi].markers.len);
    }

    if (g_debug) {
        try w.print("[dbg] seq0 ids=", .{});
        for (seqs[0].ids) |t| try w.print("{d} ", .{t});
        try w.print("\n[dbg] seq0 markers=", .{});
        for (seqs[0].markers) |mm| try w.print("{d} ", .{mm});
        try w.print("\n", .{});
        try w.flush();
    }

    var s: Scratch = undefined;
    s.x = try gpa.alloc(f32, Lmax * D);
    s.xn = try gpa.alloc(f32, Lmax * D);
    s.q = try gpa.alloc(f32, Lmax * D);
    s.k = try gpa.alloc(f32, Lmax * D);
    s.v = try gpa.alloc(f32, Lmax * D);
    s.qkv = try gpa.alloc(f32, Lmax * 3 * D);
    s.ctx = try gpa.alloc(f32, Lmax * D);
    s.mlp = try gpa.alloc(f32, Lmax * 2 * INTER);
    s.mlp2 = try gpa.alloc(f32, Lmax * INTER);
    // sliding layers use rows of 2*WINDOW+1; full layers use rows of L
    s.att = try gpa.alloc(f32, Lmax * @max(Lmax, 2 * WINDOW + 1));
    s.ff = try gpa.alloc(f32, Lmax * HEAD_FF);
    s.feat = try gpa.alloc(f32, Kmax);
    s.act_in = try gpa.alloc(f32, D + 4);
    const pooled = try gpa.alloc(f32, D);
    const logits = try gpa.alloc(f32, Kmax);

    // ---- forward, one question per pass
    for (questions, 0..) |q, qi| {
        const sq = seqs[qi];
        var f = FwdCtx{
            .m = &m,
            .s = &s,
            .ids = sq.ids,
            .L = sq.ids.len,
            .qtype = sq.qtype,
            .markers = sq.markers,
            .logits = logits,
            .act = .{ 0, 0 },
            .pooled = pooled,
            .feat = s.feat,
            .act_in = s.act_in,
        };
        forward(gpa, &f);

        const k = sq.markers.len;
        const temp = @max(1e-3, m.temperature[q.qtype]);
        const p = try gpa.alloc(f32, k);
        defer gpa.free(p);
        var mx: f32 = -std.math.inf(f32);
        for (0..k) |r| {
            p[r] = f.logits[r] / temp;
            mx = @max(mx, p[r]);
        }
        var sum: f32 = 0;
        for (p) |*v| {
            v.* = @exp(v.* - mx);
            sum += v.*;
        }
        for (p) |*v| v.* /= sum;

        try w.print("\n[{s}] ({s})\n", .{ q.name, q.type_name });
        if (q.qtype == 2) {
            try w.print("  noul        {d:.4}\n", .{p[1]});
            try w.print("  confidence  {d:.4}\n", .{@max(p[1], 1.0 - p[1])});
        } else {
            var best: usize = 0;
            for (0..k) |r| {
                if (p[r] > p[best]) best = r;
            }
            if (q.qtype == 0) {
                try w.print("  choice      {s}\n", .{q.keys[best]});
            } else {
                var exp_score: f32 = 0;
                for (0..k) |r| exp_score += @as(f32, @floatFromInt(r)) * p[r];
                try w.print("  score       {d:.4}\n", .{exp_score});
            }
            for (0..k) |r| try w.print("    {s:<24} {d:.4}\n", .{ q.keys[r], p[r] });
            try w.print("  confidence  {d:.4}\n", .{confidenceFromProbs(p, k)});
        }
        try w.print("  act_prob    {d:.4}\n", .{f.act[0]});
        try w.print("  tokens      {d}\n", .{sq.ids.len});
        try w.flush();
    }

    const t3 = nowMs(io);
    try w.print("\nforward {d} ms | total {d} ms\n", .{ t3 - t2, t3 - t0 });
    try w.flush();
}
