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
//!         --profile                print where the forward pass spends its time
//!
//! snake:  --snake                let the model play Snake (one game, animated board)
//!         --snake --games 10     summary over N games, no animation
//!         --snake --policy greedy|random|tiers|cycle  baselines to compare against
//!         --snake --prompt       print the board text the model is given
//!         --snake --promptv 0..9  how the position is phrased (ladder in snake.zig)
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
var g_threads: usize = 0; // 0 = pick from the CPU count in main
var g_debug: bool = false;

/// `--dumpstats`: after every encoder layer, print a few numbers describing the
/// hidden state, so a reference implementation can be diffed against this one
/// layer by layer (see tools/refcheck.py). Only the first forward pass dumps.
var g_dump_stats: bool = false;
/// Kernel switch (see --benchgemm): 0 = plain block4, 1 = k-unrolled block4u,
/// 2 = the packed 16-column kernel. `--benchgemm` measures 2 against 1.
var g_kern_alt: u8 = 2;
var g_pretouch: bool = false;
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

/// `--profile`: per-stage timing of the forward pass. Worker threads have no
/// std.Io to read a clock with, so this goes straight to QueryPerformanceCounter.
var g_prof: bool = false;
var g_qpc_freq: i64 = 1_000_000_000;

extern "kernel32" fn QueryPerformanceCounter(lpLargeInteger: *i64) callconv(.winapi) i32;
extern "kernel32" fn QueryPerformanceFrequency(lpFrequency: *i64) callconv(.winapi) i32;

fn nowNs() i64 {
    var c: i64 = undefined;
    _ = QueryPerformanceCounter(&c);
    return @divTrunc(c * std.time.ns_per_s, g_qpc_freq);
}

const STAGE_NAMES = [_][]const u8{
    "matmul",   "attention", "norm",     "rope",   "glu+gelu",
    "residual", "embed",     "splitQkv", "serial",
};
const ST_MATMUL: u8 = 0;
const ST_ATTN: u8 = 1;
const ST_NORM: u8 = 2;
const ST_ROPE: u8 = 3;
const ST_GELU: u8 = 4;
const ST_RES: u8 = 5;
const ST_EMB: u8 = 6;
const ST_SPLIT: u8 = 7;
const ST_SERIAL: u8 = 8;

fn zeroI64s() [STAGE_NAMES.len]i64 {
    var a: [STAGE_NAMES.len]i64 = undefined;
    @memset(&a, 0);
    return a;
}

var g_stage: u8 = ST_SERIAL;
var g_stage_ns = zeroI64s();
var g_stage_calls = zeroI64s();

fn profAdd(stage: u8, ns: i64) void {
    g_stage_ns[stage] += ns;
    g_stage_calls[stage] += 1;
}

fn profReset() void {
    @memset(&g_stage_ns, 0);
    @memset(&g_stage_calls, 0);
}

fn profReport(w: *std.Io.Writer) !void {
    var tot: i64 = 0;
    for (g_stage_ns) |v| tot += v;
    if (tot == 0) return;
    try w.print("\n[profile] {d} threads, {d} ms of measured stage time\n", .{ g_threads, @as(f64, @floatFromInt(tot)) / 1e6 });
    for (STAGE_NAMES, 0..) |name, i| {
        if (g_stage_ns[i] == 0) continue;
        const ms = @as(f64, @floatFromInt(g_stage_ns[i])) / 1e6;
        const pct = @as(f64, @floatFromInt(g_stage_ns[i])) * 100.0 / @as(f64, @floatFromInt(tot));
        try w.print("  {s:<9} {d:>8.1} ms {d:>6.1}% {d:>7} calls\n", .{ name, ms, pct, g_stage_calls[i] });
    }
    try w.flush();
}

const JobFn = *const fn (ctx: *anyopaque, lo: usize, hi: usize) void;

/// Worker threads created once and parked between jobs. The previous code spawned
/// and joined `--threads` OS threads for every parallel() call, and with ~250 calls
/// per forward pass that cost more than every matmul in it (see --profile).
const Pool = struct {
    /// a job smaller than this per thread is not worth a barrier
    const min_per_thread: usize = 16;
    /// how long a worker spins before parking: longer than the gap between two
    /// back-to-back jobs, so a forward pass never pays a futex round-trip
    const spin_budget: u32 = 30_000;

    job_ctx: *anyopaque = undefined,
    job_f: JobFn = undefined,
    job_n: usize = 0,
    job_chunk: usize = 1,
    job_chunks: usize = 0,

    gen: std.atomic.Value(u64) = .init(0),
    next: std.atomic.Value(usize) = .init(0),
    live: std.atomic.Value(usize) = .init(0),
    parked: std.atomic.Value(usize) = .init(0),
    stop: std.atomic.Value(bool) = .init(false),

    threads: AList(std.Thread) = undefined,
    nworkers: usize = 0,

    /// hand out index ranges until the job is exhausted
    fn drain(self: *Pool) void {
        const f = self.job_f;
        const ctx = self.job_ctx;
        const n = self.job_n;
        const chunk = self.job_chunk;
        const chunks = self.job_chunks;
        while (true) {
            const i = self.next.fetchAdd(1, .monotonic);
            if (i >= chunks) return;
            const lo = i * chunk;
            const hi = @min(lo + chunk, n);
            f(ctx, lo, hi);
        }
    }

    fn worker(self: *Pool) void {
        var seen: u64 = 0;
        while (true) {
            var spins: u32 = 0;
            while (true) {
                const g = self.gen.load(.acquire);
                if (g != seen) {
                    seen = g;
                    break;
                }
                if (self.stop.load(.acquire)) return;
                spins += 1;
                if (spins < spin_budget) std.atomic.spinLoopHint() else self.park(seen);
            }
            if (self.stop.load(.acquire)) return;
            self.drain();
            _ = self.live.fetchSub(1, .release);
        }
    }

    fn park(self: *Pool, seen: u64) void {
        _ = self.parked.fetchAdd(1, .monotonic);
        if (self.gen.load(.acquire) == seen and !self.stop.load(.acquire)) {
            _ = std.os.windows.ntdll.RtlWaitOnAddress(&self.gen.raw, &seen, @sizeOf(u64), null);
        }
        _ = self.parked.fetchSub(1, .monotonic);
    }

    fn run(self: *Pool, n: usize, ctx: *anyopaque, f: JobFn) void {
        const nw = self.nworkers;
        if (nw == 0 or n / min_per_thread == 0) {
            f(ctx, 0, n);
            return;
        }
        const nt = @min(nw + 1, n / min_per_thread + 1);
        const chunk = @max(1, (n + nt * 4 - 1) / (nt * 4));
        self.job_ctx = ctx;
        self.job_f = f;
        self.job_n = n;
        self.job_chunk = chunk;
        self.job_chunks = (n + chunk - 1) / chunk;
        self.next.store(0, .monotonic);
        self.live.store(nw, .monotonic);
        _ = self.gen.fetchAdd(1, .release);
        if (self.parked.load(.acquire) != 0) std.os.windows.ntdll.RtlWakeAddressAll(&self.gen.raw);
        self.drain();
        var spins: u32 = 0;
        while (self.live.load(.acquire) != 0) {
            spins += 1;
            if (spins < 2000) std.atomic.spinLoopHint() else std.Thread.yield() catch {};
        }
    }

    fn start(self: *Pool, alloc: Allocator, nt: usize) void {
        self.threads = AList(std.Thread).init(alloc);
        // A worker's largest frame is the 48 KB weight tile in `mmJob16`. The 16 MB
        // default reserved ~176 MB of commit charge across 11 workers for stacks
        // that never grow past that tile.
        const opts = std.Thread.SpawnConfig{ .stack_size = 256 * 1024 };
        for (1..nt) |_| {
            const t = std.Thread.spawn(opts, worker, .{self}) catch break;
            self.threads.append(t) catch break;
            self.nworkers += 1;
        }
    }

    fn stopAll(self: *Pool) void {
        self.stop.store(true, .release);
        // A worker can pass its `stop` check and enter `park` in the window
        // between that store and the wake below. It would then wait on `gen`,
        // which no longer changes, and the join would never return -- a lost
        // wakeup that showed up as a process which finished its work (0.9 s of
        // CPU) and never exited. Bumping `gen` makes the value comparison inside
        // the wait fail, so a worker that arrives late does not sleep.
        _ = self.gen.fetchAdd(1, .release);
        std.os.windows.ntdll.RtlWakeAddressAll(&self.gen.raw);
        for (self.threads.items) |t| t.join();
        self.threads.deinit();
        self.nworkers = 0;
    }
};

var g_pool: Pool = .{};

fn parallel(alloc: Allocator, n: usize, ctx: *anyopaque, f: JobFn) void {
    _ = alloc;
    if (n == 0) return;
    if (!g_prof) {
        g_pool.run(n, ctx, f);
        return;
    }
    const t0 = nowNs();
    g_pool.run(n, ctx, f);
    profAdd(g_stage, nowNs() - t0);
}

// --------------------------------------------------------------------------- matmul
/// The parts of a matmul the kernel sees as f32. The weight slice lives in the job
/// structs below, because its element type is the only difference between them.
const MM = struct {
    out: []f32,
    x: []const f32,
    n: usize,
    k: usize,
    rows: usize,
};

const MM32 = struct { m: MM, w: []align(1) const f32 };

/// A row-major [n][k] weight matrix as it sits in the file, so the kernel converts
/// on load and no F32 copy of it is ever made.
const MM16 = struct { m: MM, w: []align(1) const f16 };

/// Widening four rows at a time needs room for the largest k a caller passes.
const tile_k_max: usize = @max(@max(D, INTER), HEAD_FF);

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
        a0 = @mulAdd(V, x0, y0, a0);
        a1 = @mulAdd(V, x1, y1, a1);
    }
    while (i + 8 <= n) : (i += 8) {
        const x0: @Vector(8, f32) = @bitCast(a[i..][0..8].*);
        const y0: @Vector(8, f32) = @bitCast(b[i..][0..8].*);
        a0 = @mulAdd(V, x0, y0, a0);
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

    // The production path is `linear16` (f16 weights, widened four rows at a time),
    // not the f32 `matmul` above: same kernel shape, different widening and storage.
    // Check the real one against a reference built from the *rounded* f16 weights,
    // so the only difference left is the summation itself. The odd row counts make
    // sure the token tail of the block kernel is exercised.
    const shapes16 = [_]Shape{
        .{ .rows = 63, .n = 768, .k = 768, .what = "linear16 63x768x768" },
        .{ .rows = 64, .n = 768, .k = 1152, .what = "linear16 64x768x1152" },
        .{ .rows = 1, .n = 768, .k = 768, .what = "linear16 1x768x768" },
    };
    var worst16: f64 = 0;
    for (shapes16) |s| {
        const x = try gpa.alloc(f32, s.rows * s.k);
        defer gpa.free(x);
        const wm = try gpa.alloc(f16, s.n * s.k);
        defer gpa.free(wm);
        const out = try gpa.alloc(f32, s.rows * s.n);
        defer gpa.free(out);
        for (x) |*v| v.* = rng.random().float(f32) * 2.0 - 1.0;
        for (wm) |*v| v.* = @floatCast(rng.random().float(f32) * 2.0 - 1.0);

        linear16(gpa, out, x, wm, null, s.rows, s.n, s.k);

        var local_worst: f64 = 0;
        for (0..s.rows) |i| {
            var j: usize = 0;
            while (j < s.n) : (j += 29) {
                var ref: f64 = 0;
                for (0..s.k) |t| ref += @as(f64, x[i * s.k + t]) * @as(f64, @as(f32, wm[j * s.k + t]));
                const rel = @abs(@as(f64, out[i * s.n + j]) - ref) / @max(1.0, @abs(ref));
                if (rel > local_worst) local_worst = rel;
            }
        }
        worst16 = @max(worst16, local_worst);
        try w.print("  {d:>3}x{d:<5} x {d:>4}x{d:<5} {s:<22} max rel err {e:.2}\n", .{ s.rows, s.k, s.k, s.n, s.what, local_worst });
    }
    try w.print("worst relative error vs f64 (f16 path): {e:.3}\n", .{worst16});
    worst = @max(worst, worst16);

    // The elementwise transcendentals run once per activation (millions of times a
    // forward). expv is checked against the f64 library exp; geluv against the
    // scalar gelu the decision head still uses -- same coefficients, different code
    // path, so a vectorisation bug shows up here. Absolute truth for gelu comes from
    // tools/refcheck.py, which evaluates math.erf in f64.
    var worst_exp: f64 = 0;
    var worst_gelu: f64 = 0;
    {
        var lane: [8]f32 = undefined;
        var t: i32 = -8000;
        while (t <= 8000) : (t += 1) {
            const base: f32 = @as(f32, @floatFromInt(t)) * 0.001;
            for (0..8) |l| lane[l] = base + @as(f32, @floatFromInt(l)) * 0.00017;
            const ea: [8]f32 = @bitCast(expv(@bitCast(lane)));
            const ga: [8]f32 = @bitCast(geluv(@bitCast(lane)));
            for (0..8) |l| {
                const x = lane[l];
                const we = @exp(@as(f64, x));
                worst_exp = @max(worst_exp, @abs(@as(f64, ea[l]) - we) / we);
                const wg: f64 = gelu(x);
                // |x|/sqrt2 > 5 drives erf to saturation, and there both this and the
                // scalar path resolve a ~1e-7 cancellation against a true value of ~0,
                // so the comparison floors at an absolute error of 5e-7.
                const rel = @abs(@as(f64, ga[l]) - wg) / @max(0.05, @abs(wg));
                worst_gelu = @max(worst_gelu, rel);
            }
        }
    }
    try w.print("expv  max rel err vs f64 exp {e:.2}\ngeluv max rel err vs scalar gelu {e:.2}\n", .{ worst_exp, worst_gelu });

    if (worst > 1e-4) {
        try w.print("FAIL: the GEMM kernel is wrong (expected < 1e-4)\n", .{});
    } else if (worst_exp > 1e-6 or worst_gelu > 1e-5) {
        try w.print("FAIL: the vector transcendentals are wrong\n", .{});
    } else {
        try w.print("PASS\n", .{});
    }
    try w.flush();
}

/// single input row against n weight rows
fn rowDot(out: []f32, x: []const f32, w: []const f32) void {
    for (0..out.len) |j| out[j] = dotv(x, w[j * x.len ..][0..x.len]);
}

const V = @Vector(8, f32);

inline fn v8(a: []align(1) const f32, o: usize) V {
    return @bitCast(a[o..][0..8].*);
}

/// The 8 lanes hold partial sums for 8 k-elements, so the block kernel reduces
/// once per output instead of once per dot product.
inline fn hsum(v: V) f32 {
    return @reduce(.Add, v);
}

inline fn splatv(comptime k: f32) V {
    return @splat(k);
}

/// Eight weight elements as f32 lanes. F16 -> F32 is a widening with nothing to
/// round, so half-precision storage changes neither the products nor the
/// summation order: inside one kernel the f32 and f16 weight types agree bit for
/// bit. The two kernels do *not* agree with each other -- the packed one sums
/// sequentially over k instead of into eight partial sums -- so `--selftest`
/// checks each against f64 and `tools/refcheck.py` checks the whole network.
///
/// The slice has to stay typed. Reading the same bytes out of a `[]const u8` with
/// a `@bitCast` load costs 8-12x here -- this form compiles to one `vcvtph2ps`
/// per eight weights.
inline fn wvec(comptime T: type, w: []align(1) const T, t: usize) V {
    const h: @Vector(8, T) = @bitCast(w[t..][0..8].*);
    return @as(V, @floatCast(h));
}

inline fn wat(comptime T: type, w: []align(1) const T, t: usize) f32 {
    return @floatCast(w[t]);
}

/// `dotv` over a weight row; same accumulation order, so f32 and f16 weights give
/// identical results.
fn dotvw(comptime T: type, a: []const f32, b: []align(1) const T) f32 {
    const n = a.len;
    var a0: V = @splat(0);
    var a1: V = @splat(0);
    var i: usize = 0;
    while (i + 16 <= n) : (i += 16) {
        a0 = @mulAdd(V, v8(a, i), wvec(T, b, i), a0);
        a1 = @mulAdd(V, v8(a, i + 8), wvec(T, b, i + 8), a1);
    }
    while (i + 8 <= n) : (i += 8) a0 = @mulAdd(V, v8(a, i), wvec(T, b, i), a0);
    var s: f32 = 0;
    while (i < n) : (i += 1) s += a[i] * wat(T, b, i);
    return s + @reduce(.Add, a0 + a1);
}

/// Elementwise exp for 8 lanes. Range reduction is on the binary exponent, so the
/// polynomial only has to cover |r| <= ln2/2 and 7th order is good to ~5e-9
/// relative. `--selftest` checks both this and `geluv` against f64.
inline fn expv(x: V) V {
    const clamped = @max(@min(x, splatv(88.0)), splatv(-87.0));
    const n = @round(clamped * splatv(1.4426950408889634));
    const ni: @Vector(8, i32) = @intFromFloat(n);
    const scale: V = @bitCast((ni + @as(@Vector(8, i32), @splat(127))) *%
        @as(@Vector(8, i32), @splat(0x800000)));
    const r = clamped - n * splatv(0.6931471805599453);
    var p: V = splatv(1.0 / 5040.0);
    p = @mulAdd(V, p, r, splatv(1.0 / 720.0));
    p = @mulAdd(V, p, r, splatv(1.0 / 120.0));
    p = @mulAdd(V, p, r, splatv(1.0 / 24.0));
    p = @mulAdd(V, p, r, splatv(1.0 / 6.0));
    p = @mulAdd(V, p, r, splatv(0.5));
    p = @mulAdd(V, p, r, splatv(1.0));
    p = @mulAdd(V, p, r, splatv(1.0));
    return p * scale;
}

/// gelu over 8 lanes. Written as 0.5*(x + |x|*erf(|x|/sqrt2)) so the sign of x never
/// has to be selected — that form is what the scalar `gelu` above evaluates to, and
/// it collapses to the same value once erf saturates at 1.
inline fn geluv(x: V) V {
    const ax = @abs(x);
    const a = ax * splatv(0.7071067811865476);
    const t: V = splatv(1.0) / (splatv(1.0) + a * splatv(0.3275911));
    var p: V = splatv(1.061405429);
    p = @mulAdd(V, p, t, splatv(-1.453152027));
    p = @mulAdd(V, p, t, splatv(1.421413741));
    p = @mulAdd(V, p, t, splatv(-0.284496736));
    p = @mulAdd(V, p, t, splatv(0.254829592));
    const e = splatv(1.0) - (p * t) * expv(splatv(0.0) - a * a);
    return splatv(0.5) * (x + ax * e);
}

/// out[*][j..j+4] against four weight rows at once, two tokens at a time.
///
/// The shape matters: with one weight row at a time (the old form, `dotv` per
/// output) there are only two accumulator chains in flight, so the kernel waits
/// on FMA latency instead of issuing. Four rows x two tokens = eight independent
/// chains, which is what it takes to keep both FMA units busy, and each row's
/// k-sweep stays a straight contiguous stream.
///
/// The rows arrive already f32: converting half precision inside this loop would
/// repeat the same conversion for every token pair, and `vcvtph2ps` issues on the
/// two ports the FMAs live on, so it costs throughput rather than being free.
fn block4(m: *const MM, j: usize, w: [4][]align(1) const f32) void {
    const k = m.k;
    const n = m.n;
    const rows = m.rows;
    const w0 = w[0];
    const w1 = w[1];
    const w2 = w[2];
    const w3 = w[3];

    var i: usize = 0;
    while (i < rows) : (i += 2) {
        const step: usize = if (i + 1 < rows) 1 else 0;
        const x0 = m.x[i * k ..];
        const x1 = m.x[(i + step) * k ..];
        const o0 = m.out[i * n ..];
        const o1 = m.out[(i + step) * n ..];
        const two = step == 1;

        var a00: V = @splat(0);
        var a01: V = @splat(0);
        var a02: V = @splat(0);
        var a03: V = @splat(0);
        var a10: V = @splat(0);
        var a11: V = @splat(0);
        var a12: V = @splat(0);
        var a13: V = @splat(0);

        var t: usize = 0;
        while (t + 8 <= k) : (t += 8) {
            const p0 = v8(x0, t);
            const p1 = v8(x1, t);
            const y0 = v8(w0, t);
            const y1 = v8(w1, t);
            const y2 = v8(w2, t);
            const y3 = v8(w3, t);
            a00 = @mulAdd(V, p0, y0, a00);
            a01 = @mulAdd(V, p1, y0, a01);
            a02 = @mulAdd(V, p0, y1, a02);
            a03 = @mulAdd(V, p1, y1, a03);
            a10 = @mulAdd(V, p0, y2, a10);
            a11 = @mulAdd(V, p1, y2, a11);
            a12 = @mulAdd(V, p0, y3, a12);
            a13 = @mulAdd(V, p1, y3, a13);
        }
        while (t < k) : (t += 1) {
            const c0 = w0[t];
            const c1 = w1[t];
            const c2 = w2[t];
            const c3 = w3[t];
            a00[0] = @mulAdd(f32, x0[t], c0, a00[0]);
            a01[0] = @mulAdd(f32, x1[t], c0, a01[0]);
            a02[0] = @mulAdd(f32, x0[t], c1, a02[0]);
            a03[0] = @mulAdd(f32, x1[t], c1, a03[0]);
            a10[0] = @mulAdd(f32, x0[t], c2, a10[0]);
            a11[0] = @mulAdd(f32, x1[t], c2, a11[0]);
            a12[0] = @mulAdd(f32, x0[t], c3, a12[0]);
            a13[0] = @mulAdd(f32, x1[t], c3, a13[0]);
        }

        o0[j + 0] = hsum(a00);
        o0[j + 1] = hsum(a02);
        o0[j + 2] = hsum(a10);
        o0[j + 3] = hsum(a12);
        if (two) {
            o1[j + 0] = hsum(a01);
            o1[j + 1] = hsum(a03);
            o1[j + 2] = hsum(a11);
            o1[j + 3] = hsum(a13);
        }
    }
}

/// The four weight rows a block works on.
fn four(w: []align(1) const f32, k: usize) [4][]align(1) const f32 {
    return .{ w[0..k], w[k..][0..k], w[2 * k ..][0..k], w[3 * k ..][0..k] };
}

/// 4 columns x 2 tokens with the k-loop unrolled twice: the same accumulators and
/// the same summation order as `block4`, but the loop overhead (two pointer
/// increments, the compare and the branch) is paid once per 16 FMAs instead of
/// once per 8, and the second half's loads can issue while the first half's FMAs
/// are still in flight.
fn block4u(m: *const MM, j: usize, w: [4][]align(1) const f32) void {
    const k = m.k;
    const n = m.n;
    const rows = m.rows;
    const w0 = w[0];
    const w1 = w[1];
    const w2 = w[2];
    const w3 = w[3];

    var i: usize = 0;
    while (i < rows) : (i += 2) {
        const step: usize = if (i + 1 < rows) 1 else 0;
        const x0 = m.x[i * k ..];
        const x1 = m.x[(i + step) * k ..];
        const o0 = m.out[i * n ..];
        const o1 = m.out[(i + step) * n ..];
        const two = step == 1;

        var a00: V = @splat(0);
        var a01: V = @splat(0);
        var a02: V = @splat(0);
        var a03: V = @splat(0);
        var a10: V = @splat(0);
        var a11: V = @splat(0);
        var a12: V = @splat(0);
        var a13: V = @splat(0);

        var t: usize = 0;
        while (t + 16 <= k) : (t += 16) {
            {
                const p0 = v8(x0, t);
                const p1 = v8(x1, t);
                const y0 = v8(w0, t);
                const y1 = v8(w1, t);
                const y2 = v8(w2, t);
                const y3 = v8(w3, t);
                a00 = @mulAdd(V, p0, y0, a00);
                a01 = @mulAdd(V, p1, y0, a01);
                a02 = @mulAdd(V, p0, y1, a02);
                a03 = @mulAdd(V, p1, y1, a03);
                a10 = @mulAdd(V, p0, y2, a10);
                a11 = @mulAdd(V, p1, y2, a11);
                a12 = @mulAdd(V, p0, y3, a12);
                a13 = @mulAdd(V, p1, y3, a13);
            }
            {
                const p0 = v8(x0, t + 8);
                const p1 = v8(x1, t + 8);
                const y0 = v8(w0, t + 8);
                const y1 = v8(w1, t + 8);
                const y2 = v8(w2, t + 8);
                const y3 = v8(w3, t + 8);
                a00 = @mulAdd(V, p0, y0, a00);
                a01 = @mulAdd(V, p1, y0, a01);
                a02 = @mulAdd(V, p0, y1, a02);
                a03 = @mulAdd(V, p1, y1, a03);
                a10 = @mulAdd(V, p0, y2, a10);
                a11 = @mulAdd(V, p1, y2, a11);
                a12 = @mulAdd(V, p0, y3, a12);
                a13 = @mulAdd(V, p1, y3, a13);
            }
        }
        while (t + 8 <= k) : (t += 8) {
            const p0 = v8(x0, t);
            const p1 = v8(x1, t);
            const y0 = v8(w0, t);
            const y1 = v8(w1, t);
            const y2 = v8(w2, t);
            const y3 = v8(w3, t);
            a00 = @mulAdd(V, p0, y0, a00);
            a01 = @mulAdd(V, p1, y0, a01);
            a02 = @mulAdd(V, p0, y1, a02);
            a03 = @mulAdd(V, p1, y1, a03);
            a10 = @mulAdd(V, p0, y2, a10);
            a11 = @mulAdd(V, p1, y2, a11);
            a12 = @mulAdd(V, p0, y3, a12);
            a13 = @mulAdd(V, p1, y3, a13);
        }
        while (t < k) : (t += 1) {
            const c0 = w0[t];
            const c1 = w1[t];
            const c2 = w2[t];
            const c3 = w3[t];
            a00[0] = @mulAdd(f32, x0[t], c0, a00[0]);
            a01[0] = @mulAdd(f32, x1[t], c0, a01[0]);
            a02[0] = @mulAdd(f32, x0[t], c1, a02[0]);
            a03[0] = @mulAdd(f32, x1[t], c1, a03[0]);
            a10[0] = @mulAdd(f32, x0[t], c2, a10[0]);
            a11[0] = @mulAdd(f32, x1[t], c2, a11[0]);
            a12[0] = @mulAdd(f32, x0[t], c3, a12[0]);
            a13[0] = @mulAdd(f32, x1[t], c3, a13[0]);
        }

        o0[j + 0] = hsum(a00);
        o0[j + 1] = hsum(a02);
        o0[j + 2] = hsum(a10);
        o0[j + 3] = hsum(a12);
        if (two) {
            o1[j + 0] = hsum(a01);
            o1[j + 1] = hsum(a03);
            o1[j + 2] = hsum(a11);
            o1[j + 3] = hsum(a13);
        }
    }
}


/// Widen four rows, in the order `block4` reads them. F16 -> F32 has nothing to
/// round, so this is the same arithmetic done in the same order.
fn widen4(src: []align(1) const f16, tile: []align(64) f32) void {
    const n = src.len;
    var t: usize = 0;
    while (t + 8 <= n) : (t += 8) {
        const h: @Vector(8, f16) = @bitCast(src[t..][0..8].*);
        tile[t..][0..8].* = @bitCast(@as(@Vector(8, f32), @floatCast(h)));
    }
    while (t < n) : (t += 1) tile[t] = @floatCast(src[t]);
}

// --------------------------------------------------------------------------- packed kernel
//
// The 4-column kernel above keeps one accumulator per (token, column) pair, so
// 16 architectural vector registers cap it at 2x4 = 8 accumulators and at four
// loads per eight FMAs. That is what it costs: ~50 % of this core's FMA peak, and
// `x` is re-streamed from L2 once per *four* columns.
//
// This kernel transposes instead. Sixteen weight rows are packed k-major into a
// KC x 16 panel, so one 8-lane load covers eight *columns* and the accumulator
// vectors span columns rather than k. Six tokens then cost twelve accumulators
// and, per k-position, two panel loads plus six broadcasts -- 8 load uops for 12
// FMA uops, i.e. FMA-bound at 6 cycles instead of load-bound. Two more things
// fall out of the wider block: `x` is swept once per 16 columns instead of once
// per 4, and a row of `out` is written as one 64-byte line instead of four
// 16-byte pieces separated by a full sweep of `x`.

/// Columns per packed block: 16 f32 is exactly one cache line.
const PN: usize = 16;
/// k-chunk. A PN x PKC f32 panel is 24 KB, which leaves a 48 KB L1 room for the
/// A side and the accumulator tile. k > PKC accumulates into `out` between
/// chunks, which is why the second chunk and later take the `accumulate` path.
const PKC: usize = 384;

inline fn unpcklo(a: V, b: V) V {
    return @shuffle(f32, a, b, [8]i32{ 0, -1, 1, -2, 2, -3, 3, -4 });
}

inline fn unpckhi(a: V, b: V) V {
    return @shuffle(f32, a, b, [8]i32{ 4, -5, 5, -6, 6, -7, 7, -8 });
}

/// Eight rows x eight k-positions in, eight columns out: element j of the result
/// is row-major column j of the input block. Packing needs the transpose because
/// the weights arrive [n][k] and the kernel wants [k][n].
inline fn transpose8(r: [8]V) [8]V {
    const t0 = unpcklo(r[0], r[1]);
    const t1 = unpckhi(r[0], r[1]);
    const t2 = unpcklo(r[2], r[3]);
    const t3 = unpckhi(r[2], r[3]);
    const t4 = unpcklo(r[4], r[5]);
    const t5 = unpckhi(r[4], r[5]);
    const t6 = unpcklo(r[6], r[7]);
    const t7 = unpckhi(r[6], r[7]);

    const q0 = @shuffle(f32, t0, t2, [8]i32{ 0, 1, -1, -2, 2, 3, -3, -4 });
    const q1 = @shuffle(f32, t0, t2, [8]i32{ 4, 5, -5, -6, 6, 7, -7, -8 });
    const q2 = @shuffle(f32, t1, t3, [8]i32{ 0, 1, -1, -2, 2, 3, -3, -4 });
    const q3 = @shuffle(f32, t1, t3, [8]i32{ 4, 5, -5, -6, 6, 7, -7, -8 });
    const p0 = @shuffle(f32, t4, t6, [8]i32{ 0, 1, -1, -2, 2, 3, -3, -4 });
    const p1 = @shuffle(f32, t4, t6, [8]i32{ 4, 5, -5, -6, 6, 7, -7, -8 });
    const p2 = @shuffle(f32, t5, t7, [8]i32{ 0, 1, -1, -2, 2, 3, -3, -4 });
    const p3 = @shuffle(f32, t5, t7, [8]i32{ 4, 5, -5, -6, 6, 7, -7, -8 });

    const lo = [8]i32{ 0, 1, 2, 3, -1, -2, -3, -4 };
    const hi = [8]i32{ 4, 5, 6, 7, -5, -6, -7, -8 };
    return .{
        @shuffle(f32, q0, p0, lo), @shuffle(f32, q0, p0, hi),
        @shuffle(f32, q1, p1, lo), @shuffle(f32, q1, p1, hi),
        @shuffle(f32, q2, p2, lo), @shuffle(f32, q2, p2, hi),
        @shuffle(f32, q3, p3, lo), @shuffle(f32, q3, p3, hi),
    };
}

/// panel[t*PN + c] = w[(jg+c)*k + kb + t] -- 16 weight rows, k-chunk [kb, kb+kc),
/// widened from f16 and transposed in 8x8 blocks. One conversion per weight
/// element, same as `widen4`, and the panel is then reused by every token group.
fn packPanel(panel: []f32, w: []align(1) const f16, k: usize, jg: usize, kb: usize, kc: usize) void {
    var t: usize = 0;
    while (t + 8 <= kc) : (t += 8) {
        var lo: [8]V = undefined;
        var hi: [8]V = undefined;
        inline for (0..8) |c| lo[c] = wvec(f16, w[(jg + c) * k + kb + t ..], 0);
        inline for (0..8) |c| hi[c] = wvec(f16, w[(jg + 8 + c) * k + kb + t ..], 0);
        const cl = transpose8(lo);
        const ch = transpose8(hi);
        inline for (0..8) |j| {
            panel[(t + j) * PN ..][0..8].* = @as([8]f32, @bitCast(cl[j]));
            panel[(t + j) * PN + 8 ..][0..8].* = @as([8]f32, @bitCast(ch[j]));
        }
    }
    while (t < kc) : (t += 1) {
        inline for (0..PN) |c| panel[t * PN + c] = @floatCast(w[(jg + c) * k + kb + t]);
    }
}

/// MRN tokens x 16 columns, accumulating k-chunk [kb, kb+kc).
///
/// `accumulate` is false for the first k-chunk and true afterwards, so a k that
/// does not fit one panel costs a read-modify-write of a 64-byte line per row
/// rather than a second buffer.
inline fn microKernel(
    comptime MRN: usize,
    panel: []const f32,
    x: []const f32,
    out: []f32,
    n: usize,
    k: usize,
    ib: usize,
    jg: usize,
    kb: usize,
    kc: usize,
    accumulate: bool,
) void {
    var acc: [MRN][2]V = undefined;
    inline for (0..MRN) |i| {
        acc[i][0] = @splat(0.0);
        acc[i][1] = @splat(0.0);
    }
    var t: usize = 0;
    while (t < kc) : (t += 1) {
        const b0: V = @bitCast(panel[t * PN ..][0..8].*);
        const b1: V = @bitCast(panel[t * PN + 8 ..][0..8].*);
        inline for (0..MRN) |i| {
            const a: V = @splat(x[(ib + i) * k + kb + t]);
            acc[i][0] = @mulAdd(V, a, b0, acc[i][0]);
            acc[i][1] = @mulAdd(V, a, b1, acc[i][1]);
        }
    }
    inline for (0..MRN) |i| {
        const o = out[(ib + i) * n + jg ..][0..PN];
        if (accumulate) {
            const p0: V = @bitCast(o[0..8].*);
            const p1: V = @bitCast(o[8..16].*);
            o[0..8].* = @as([8]f32, @bitCast(p0 + acc[i][0]));
            o[8..16].* = @as([8]f32, @bitCast(p1 + acc[i][1]));
        } else {
            o[0..8].* = @as([8]f32, @bitCast(acc[i][0]));
            o[8..16].* = @as([8]f32, @bitCast(acc[i][1]));
        }
    }
}

/// One 16-column block: pack the panel once per k-chunk, then sweep every token
/// group against it in 6/4/2/1-row pieces so no row is left to a scalar path.
fn packJob(ctx: *anyopaque, blo: usize, bhi: usize) void {
    const m: *const MM16 = @ptrCast(@alignCast(ctx));
    var buf: [PN * PKC]f32 align(64) = undefined;
    const panel: []f32 = (&buf)[0..];
    var b = blo;
    while (b < bhi) : (b += 1) {
        const jg = b * PN;
        if (jg + PN > m.m.n) break;
        var kb: usize = 0;
        while (kb < m.m.k) : (kb += PKC) {
            const kc = @min(PKC, m.m.k - kb);
            packPanel(panel, m.w, m.m.k, jg, kb, kc);
            const acc = kb != 0;
            var i: usize = 0;
            while (i + 6 <= m.m.rows) : (i += 6) microKernel(6, panel, m.m.x, m.m.out, m.m.n, m.m.k, i, jg, kb, kc, acc);
            while (i + 4 <= m.m.rows) : (i += 4) microKernel(4, panel, m.m.x, m.m.out, m.m.n, m.m.k, i, jg, kb, kc, acc);
            while (i + 2 <= m.m.rows) : (i += 2) microKernel(2, panel, m.m.x, m.m.out, m.m.n, m.m.k, i, jg, kb, kc, acc);
            while (i < m.m.rows) : (i += 1) microKernel(1, panel, m.m.x, m.m.out, m.m.n, m.m.k, i, jg, kb, kc, acc);
        }
    }
}

/// Columns the packed kernel cannot cover (n % PN != 0, which no shape in this
/// model hits) still go through the per-column path.
fn packTail(ctx: *anyopaque, lo: usize, hi: usize) void {
    const m: *const MM16 = @ptrCast(@alignCast(ctx));
    const jlo = m.m.n / PN * PN;
    for (lo..hi) |i| {
        var jj = jlo;
        while (jj < m.m.n) : (jj += 1) {
            const wrow = m.w[jj * m.m.k ..][0..m.m.k];
            m.m.out[i * m.m.n + jj] = dotvw(f16, m.m.x[i * m.m.k ..][0..m.m.k], wrow);
        }
    }
}

fn linear16Packed(alloc: Allocator, out: []f32, x: []const f32, w: []align(1) const f16, rows: usize, n: usize, k: usize) void {
    var m = MM16{ .m = .{ .out = out, .x = x, .n = n, .k = k, .rows = rows }, .w = w };
    parallel(alloc, n / PN, &m, packJob);
    if (n % PN != 0) parallel(alloc, rows, &m, packTail);
}

/// out[i][j] = dot(x[i], w[j]) -- parallelised over 4-column blocks so that each
/// weight row is streamed exactly once (this is the difference between ~2 s and
/// ~200 ms per question: the other order re-reads the whole matrix per token).
///
/// The unit of work is a block, not a column: `parallel` splits whatever range it
/// is given into chunks of an arbitrary size, and a chunk that starts or ends
/// mid-block falls back to the per-column path below, which reads each weight row
/// once per token instead of once per four columns.
fn mmJob32(ctx: *anyopaque, blo: usize, bhi: usize) void {
    const m: *const MM32 = @ptrCast(@alignCast(ctx));
    const k = m.m.k;
    var b = blo;
    while (b < bhi) : (b += 1) {
        const j = b * 4;
        if (j + 4 <= m.m.n) {
            block4(&m.m, j, four(m.w[j * k ..], k));
            continue;
        }
        var jj = j;
        while (jj < m.m.n) : (jj += 1) {
            const wrow = m.w[jj * k ..][0..k];
            for (0..m.m.rows) |i| m.m.out[i * m.m.n + jj] = dotvw(f32, m.m.x[i * k ..][0..k], wrow);
        }
    }
}

/// The same sweep over half-precision weights. Each block is widened once into a
/// tile on this thread's stack and then reused by every token pair, which is what
/// keeps the conversions at one per weight element instead of one per pair.
fn mmJob16(ctx: *anyopaque, blo: usize, bhi: usize) void {
    const m: *const MM16 = @ptrCast(@alignCast(ctx));
    const k = m.m.k;
    var buf: [4 * tile_k_max]f32 align(64) = undefined;
    const tile: []align(64) f32 = (&buf)[0..];
    var b = blo;
    while (b < bhi) : (b += 1) {
        const j = b * 4;
        if (j + 4 <= m.m.n) {
            widen4(m.w[j * k ..][0 .. 4 * k], tile);
            const rows = four(tile[0 .. 4 * k], k);
            if (g_kern_alt == 1) block4u(&m.m, j, rows) else block4(&m.m, j, rows);
            continue;
        }
        var jj = j;
        while (jj < m.m.n) : (jj += 1) {
            const wrow = m.w[jj * k ..][0..k];
            for (0..m.m.rows) |i| m.m.out[i * m.m.n + jj] = dotvw(f16, m.m.x[i * k ..][0..k], wrow);
        }
    }
}

const Bias = struct { out: []f32, b: []const f32, n: usize };

fn biasJob(ctx: *anyopaque, lo: usize, hi: usize) void {
    const m: *Bias = @ptrCast(@alignCast(ctx));
    for (lo..hi) |i| {
        const row = m.out[i * m.n ..][0..m.n];
        for (row, 0..) |*v, j| v.* += m.b[j];
    }
}

fn matmul(alloc: Allocator, out: []f32, x: []const f32, w: []const f32, rows: usize, n: usize, k: usize) void {
    var m = MM32{ .m = .{ .out = out, .x = x, .n = n, .k = k, .rows = rows }, .w = w };
    parallel(alloc, (n + 3) / 4, &m, mmJob32);
}

/// F16 weights straight out of the file, converted inside the kernel: no 478 MB
/// F32 copy, and half the bytes streamed per forward.
fn linear16(alloc: Allocator, out: []f32, x: []const f32, w: []align(1) const f16, bias: ?[]const f32, rows: usize, n: usize, k: usize) void {
    if (g_kern_alt == 2) {
        linear16Packed(alloc, out, x, w, rows, n, k);
    } else {
        var m = MM16{ .m = .{ .out = out, .x = x, .n = n, .k = k, .rows = rows }, .w = w };
        parallel(alloc, (n + 3) / 4, &m, mmJob16);
    }
    if (bias) |b| {
        var bias_job = Bias{ .out = out, .b = b, .n = n };
        parallel(alloc, rows, &bias_job, biasJob);
    }
}

fn rowDot1(a: []const f32, b: []const f32) f32 {
    var o: [1]f32 = undefined;
    rowDot(&o, a, b);
    return o[0];
}

/// Peak memory of this process, so a run reports its own cost instead of
/// depending on an external sampler that may miss a 200 ms process entirely.
const MemCounters = extern struct {
    cb: u32 = @sizeOf(MemCounters),
    PageFaultCount: u32 = 0,
    PeakWorkingSetSize: usize = 0,
    WorkingSetSize: usize = 0,
    QuotaPeakPagedPoolUsage: usize = 0,
    QuotaPagedPoolUsage: usize = 0,
    QuotaPeakNonPagedPoolUsage: usize = 0,
    QuotaNonPagedPoolUsage: usize = 0,
    PagefileUsage: usize = 0,
    PeakPagefileUsage: usize = 0,
};

extern "kernel32" fn GetCurrentProcess() callconv(.winapi) ?*anyopaque;
extern "kernel32" fn K32GetProcessMemoryInfo(h: ?*anyopaque, pmc: *MemCounters, cb: u32) callconv(.winapi) i32;

fn peakMem() ?MemCounters {
    var pmc = MemCounters{};
    if (K32GetProcessMemoryInfo(GetCurrentProcess(), &pmc, @sizeOf(MemCounters)) == 0) return null;
    return pmc;
}

const Filetime = extern struct { low: u32, high: u32 };
extern "kernel32" fn GetProcessTimes(h: ?*anyopaque, creation: *Filetime, exit: *Filetime, kernel: *Filetime, user: *Filetime) callconv(.winapi) i32;

/// CPU seconds this process has burned so far (kernel + user), so a run can be
/// compared on cost as well as on wall time -- the pool's workers spin at every
/// barrier, and that shows up here and nowhere else.
fn cpuSeconds() ?f64 {
    var c: Filetime = undefined;
    var e: Filetime = undefined;
    var k: Filetime = undefined;
    var u: Filetime = undefined;
    if (GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u) == 0) return null;
    const ticks: u64 = (@as(u64, k.high) << 32 | k.low) + (@as(u64, u.high) << 32 | u.low);
    return @as(f64, @floatFromInt(ticks)) / 1e7;
}

/// Signed percentage, since this Zig's formatter has no `{d:+}` spec.
fn reportDelta(w: *std.Io.Writer, frac: f64) !void {
    if (frac >= 0) try w.print("(+{d:.1}%)\n", .{frac * 100.0}) else try w.print("({d:.1}%)\n", .{frac * 100.0});
}

/// `--benchgemm`: the linear16 kernel on the model's real shapes, without the
/// model. Reports GFLOPS per shape so a kernel change can be judged in seconds
/// instead of by a noisy whole-forward A/B.
fn benchGemm(gpa: Allocator, w: *std.Io.Writer) !void {
    const Shape = struct { rows: usize, n: usize, k: usize, name: []const u8 };
    const shapes = [_]Shape{
        .{ .rows = 63, .n = 3 * D, .k = D, .name = "qkv 63" },
        .{ .rows = 63, .n = D, .k = D, .name = "attn-out 63" },
        .{ .rows = 63, .n = 2 * INTER, .k = D, .name = "mlp-in 63" },
        .{ .rows = 63, .n = D, .k = INTER, .name = "mlp-out 63" },
        .{ .rows = 63, .n = HEAD_FF, .k = D, .name = "head-ff1 63" },
        .{ .rows = 175, .n = 3 * D, .k = D, .name = "qkv 175" },
        .{ .rows = 175, .n = 2 * INTER, .k = D, .name = "mlp-in 175" },
    };
    // This box drifts 2-3x between "cool" and "sustained" within tens of seconds,
    // so a batch of one kernel followed by a batch of the other measures the
    // thermometer. Alternate the two on every single call and keep each one's best.
    const reps = 8;
    const saved_kern = g_kern_alt;
    defer g_kern_alt = saved_kern;
    try w.print("gemm benchmark, {d} threads, rotated A/B, best of {d} calls each\n", .{ g_threads, reps });
    var sum3: i64 = 0;
    var sum4: i64 = 0;
    var flops_all: f64 = 0;
    for (shapes) |s| {
        const x = try gpa.alloc(f32, s.rows * s.k);
        defer gpa.free(x);
        const wm = try gpa.alloc(f16, s.n * s.k);
        defer gpa.free(wm);
        const out = try gpa.alloc(f32, s.rows * s.n);
        defer gpa.free(out);
        for (x, 0..) |*v, idx| v.* = @as(f32, @floatFromInt(idx % 17)) * 0.01 - 0.08;
        for (wm, 0..) |*v, idx| v.* = @floatFromInt(@as(i32, @intCast(idx % 13)) - 6);
        const one_flops: f64 = 2.0 * @as(f64, @floatFromInt(s.rows * s.n * s.k));

        g_kern_alt = 2;
        linear16(gpa, out, x, wm, null, s.rows, s.n, s.k);
        g_kern_alt = 1;
        linear16(gpa, out, x, wm, null, s.rows, s.n, s.k);

        var best3: i64 = std.math.maxInt(i64);
        var best4: i64 = std.math.maxInt(i64);
        for (0..reps) |_| {
            g_kern_alt = 2;
            var t0 = nowNs();
            linear16(gpa, out, x, wm, null, s.rows, s.n, s.k);
            best3 = @min(best3, nowNs() - t0);
            g_kern_alt = 1;
            t0 = nowNs();
            linear16(gpa, out, x, wm, null, s.rows, s.n, s.k);
            best4 = @min(best4, nowNs() - t0);
        }
        const gf3 = one_flops / @as(f64, @floatFromInt(best3));
        const gf4 = one_flops / @as(f64, @floatFromInt(best4));
        try w.print("  {s:<14} {d:>4}x{d:<5} x {d:>4}x{d:<5}  packed {d:>6.1}  block4u {d:>6.1} GFLOPS  ", .{
            s.name, s.rows, s.k, s.k, s.n, gf3, gf4,
        });
        try reportDelta(w, gf3 / gf4 - 1.0);
        sum3 += best3;
        sum4 += best4;
        flops_all += one_flops;
    }
    const agg3 = flops_all / @as(f64, @floatFromInt(sum3));
    const agg4 = flops_all / @as(f64, @floatFromInt(sum4));
    try w.print("  aggregate      packed {d:>6.1}  block4u {d:>6.1} GFLOPS  ", .{ agg3, agg4 });
    try reportDelta(w, agg3 / agg4 - 1.0);
    try w.flush();
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

const ConvJob = struct {
    raw: []const u8,
    out: []f32,
};

/// One read per 4 KB page of the mapped weights, spread over the workers. The
/// pages are cold in this process until something touches them, and the first
/// forward would otherwise take that cost -- single-threaded, inside the GEMM
/// inner loop. Doing it up front on all cores turns ~60k serial faults into a
/// short parallel pass.
const TouchJob = struct { mem: []const u8, pages: usize };

fn touchJob(ctx: *anyopaque, lo: usize, hi: usize) void {
    const j: *const TouchJob = @ptrCast(@alignCast(ctx));
    const base = @intFromPtr(j.mem.ptr);
    var p = lo;
    while (p < hi) : (p += 1) {
        const off = p * 4096;
        if (off >= j.mem.len) return;
        _ = @as(*const volatile u8, @ptrFromInt(base + off)).*;
    }
}

fn pretouch(alloc: Allocator, mem: []const u8) void {
    var job = TouchJob{ .mem = mem, .pages = (mem.len + 4095) / 4096 };
    parallel(alloc, job.pages, &job, touchJob);
}

/// lo..hi are counted in groups of 8 elements. f16 -> f32 is exact, so this is a
/// pure bandwidth operation; doing it a value at a time made the weight load
/// (668 ms of the ~950 ms startup) cost more than a whole forward pass.
fn convF16Job(ctx: *anyopaque, lo: usize, hi: usize) void {
    const j: *ConvJob = @ptrCast(@alignCast(ctx));
    var u = lo;
    while (u < hi) : (u += 1) {
        const o = u * 8;
        const h: @Vector(8, f16) = @bitCast(j.raw[o * 2 ..][0..16].*);
        j.out[o..][0..8].* = @as([8]f32, @bitCast(@as(V, @floatCast(h))));
    }
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
    var job = ConvJob{ .raw = raw, .out = out };
    parallel(alloc, n / 8, &job, convF16Job);
    var i = (n / 8) * 8;
    while (i < n) : (i += 1) out[i] = f16at(raw, i);
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

/// A matrix the kernels consume as half precision: a view of the file image, so
/// it costs no memory beyond the mapping itself.
fn tensorF16(infos: *const std.StringHashMap(TensorInfo), data: []const u8, name: []const u8) ![]align(1) const f16 {
    const t = infos.get(name) orelse {
        std.debug.print("missing tensor: {s}\n", .{name});
        return error.MissingTensor;
    };
    if (!std.mem.eql(u8, t.dtype, "F16")) {
        std.debug.print("tensor {s} is {s}, expected F16\n", .{ name, t.dtype });
        return error.UnsupportedDtype;
    }
    const raw = data[t.offset .. t.offset + t.len];
    if (raw.len % 2 != 0) {
        std.debug.print("tensor {s} has an odd byte length\n", .{name});
        return error.BadSafetensors;
    }
    return @ptrCast(raw);
}

// --------------------------------------------------------------------------- tokenizer
const Tokenizer = struct {
    vocab: TokMap,
    merges: TokMap,
    added: TokMap,
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
    /// The string at the cursor, without copying or decoding it when it holds no
    /// escape sequences -- which is true of all but a few hundred of the 256k
    /// vocab keys, and of every merge half. The slice then points into the source
    /// buffer, so `loadTokenizer` has to keep the file mapping alive; escaped keys
    /// take the slow path and land in `buf` as before.
    fn stringLinked(self: *JScan, buf: []u8) ![]const u8 {
        self.ws();
        const start = self.i;
        try self.expect('"');
        const rest = self.s[self.i..];
        const q = std.mem.indexOfScalar(u8, rest, '"') orelse return error.TruncatedJson;
        if (std.mem.indexOfScalar(u8, rest[0..q], '\\') == null) {
            self.i += q + 1;
            return rest[0..q];
        }
        self.i = start;
        return try self.stringInto(buf);
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
    pool: BytePool = undefined,
    t_vocab: i64 = 0,
    t_merges: i64 = 0,
    t_added: i64 = 0,
};

/// Append-only byte pool for the tokenizer's keys, carved from 1 MB blocks that
/// are never freed (the tokenizer arena owns them). Offsets are stable across
/// growth, which is what lets `TokMap` -- and the on-disk cache -- refer to a key
/// by offset instead of by pointer.
const BytePool = struct {
    alloc: Allocator,
    blocks: AList([]u8) = undefined,
    lens: AList(u32) = undefined,
    cur: []u8 = &.{},
    base: u32 = 0,
    used: u32 = 0,

    const Entry = struct { off: u32, key: []const u8 };

    fn init(alloc: Allocator) BytePool {
        return .{ .alloc = alloc, .blocks = AList([]u8).init(alloc), .lens = AList(u32).init(alloc) };
    }

    fn add(self: *BytePool, bytes: []const u8) !Entry {
        const at = try self.room(bytes.len);
        const dst = self.cur[self.used..][0..bytes.len];
        @memcpy(dst, bytes);
        self.used += @intCast(bytes.len);
        return .{ .off = at, .key = dst };
    }

    /// `a` NUL `b` -- the key a BPE merge is stored under, never allocated
    /// separately the way the old `allocPrint` path did.
    fn addPair(self: *BytePool, a: []const u8, b: []const u8) !Entry {
        const n = a.len + 1 + b.len;
        const at = try self.room(n);
        const dst = self.cur[self.used..][0..n];
        @memcpy(dst[0..a.len], a);
        dst[a.len] = 0;
        @memcpy(dst[a.len + 1 ..], b);
        self.used += @intCast(n);
        return .{ .off = at, .key = dst };
    }

    fn room(self: *BytePool, n: usize) !u32 {
        if (self.used + n > self.cur.len) {
            if (self.cur.len != 0) {
                self.base += self.used;
                try self.blocks.append(self.cur);
                try self.lens.append(self.used);
            }
            self.cur = try self.alloc.alloc(u8, @max(n, 1 << 20));
            self.used = 0;
        }
        return self.base + self.used;
    }

    /// Everything added so far, contiguous. Only valid on the build path, where
    /// the blocks are written out in their used order (that is the layout the
    /// cache stores and the offsets already refer to).
    fn serializeInto(self: *const BytePool, out: []u8) void {
        var at: usize = 0;
        for (self.blocks.items, self.lens.items) |blk, n| {
            @memcpy(out[at..][0..n], blk[0..n]);
            at += n;
        }
        @memcpy(out[at..][0..self.used], self.cur[0..self.used]);
    }

    fn total(self: *const BytePool) usize {
        return self.base + self.used;
    }
};

/// A string -> u32 map as four parallel arrays over one shared byte pool, open
/// addressed with linear probing. Deliberately a fixed, self-describing layout:
/// the whole table can be written to disk once and mapped back on the next run,
/// so the 840k keys are never hashed or inserted again.
const TokMap = struct {
    hash: []const u64 = &.{}, // 0 = empty, otherwise the key's hash with bit 0 set
    off: []const u32 = &.{},
    len: []const u32 = &.{},
    val: []const u32 = &.{},
    pool: []const u8 = &.{},
    live: u32 = 0,

    fn count(self: *const TokMap) usize {
        return self.live;
    }

    fn get(self: *const TokMap, key: []const u8) ?u32 {
        const n = self.hash.len;
        if (n == 0) return null;
        const h = std.hash.Wyhash.hash(0, key) | 1;
        var i: usize = h & (n - 1);
        while (self.hash[i] != 0) {
            if (self.hash[i] == h and self.len[i] == key.len and
                std.mem.eql(u8, self.pool[self.off[i]..][0..self.len[i]], key))
            {
                return self.val[i];
            }
            i = (i + 1) & (n - 1);
        }
        return null;
    }

    fn put(self: *TokMap, alloc: Allocator, key: []const u8, off: u32, value: u32) !void {
        if (self.hash.len == 0 or @as(u64, self.live + 1) * 10 >= @as(u64, self.hash.len) * 7) {
            try self.grow(alloc);
        }
        // Only reached while building from JSON, when the arrays are ours; a map
        // loaded from the cache is read-only and never written again.
        const hash = @constCast(self.hash);
        const offs = @constCast(self.off);
        const lens = @constCast(self.len);
        const vals = @constCast(self.val);
        const h = std.hash.Wyhash.hash(0, key) | 1;
        var i: usize = h & (self.hash.len - 1);
        while (hash[i] != 0) : (i = (i + 1) & (self.hash.len - 1)) {}
        hash[i] = h;
        offs[i] = off;
        lens[i] = @intCast(key.len);
        vals[i] = value;
        self.live += 1;
    }

    fn grow(self: *TokMap, alloc: Allocator) !void {
        const n = if (self.hash.len == 0) 1 << 12 else self.hash.len * 2;
        const h = try alloc.alloc(u64, n);
        @memset(h, 0);
        const o = try alloc.alloc(u32, n);
        const l = try alloc.alloc(u32, n);
        const v = try alloc.alloc(u32, n);
        for (self.hash, 0..) |old_h, i| {
            if (old_h == 0) continue;
            var j: usize = old_h & (n - 1);
            while (h[j] != 0) : (j = (j + 1) & (n - 1)) {}
            h[j] = old_h;
            o[j] = self.off[i];
            l[j] = self.len[i];
            v[j] = self.val[i];
        }
        self.hash = h;
        self.off = o;
        self.len = l;
        self.val = v;
    }
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
    const t0 = nowNs();
    defer sink.t_vocab += nowNs() - t0;
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
        const s = try sc.stringLinked(buf);
        try sc.expect(':');
        const id = try sc.uint();
        const e = try sink.pool.add(s);
        try sink.tok.vocab.put(sink.alloc, e.key, e.off, id);
    }
}

fn parseMerges(sc: *JScan, sink: *TokSink) !void {
    const t0 = nowNs();
    defer sink.t_merges += nowNs() - t0;
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
        const a = try sc.stringLinked(buf);
        sc.ws();
        _ = try sc.byte(); // ','
        const b = try sc.stringLinked(buf[a.len..]);
        try sc.expect(']');
        const e = try sink.pool.addPair(a, b);
        try sink.tok.merges.put(sink.alloc, e.key, e.off, rank);
        rank += 1;
    }
}

fn parseAdded(sc: *JScan, sink: *TokSink) !void {
    const t0 = nowNs();
    defer sink.t_added += nowNs() - t0;
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
        if (content) |c| {
            const e = try sink.pool.add(c);
            try sink.tok.added.put(sink.alloc, e.key, e.off, id);
        }
    }
}

/// The tokenizer.json mapping behind the tokenizer's string keys. Kept for the
/// life of the process, like `g_weights`.
var g_tok_file: ?std.Io.File.MemoryMap = null;
/// Same, for the tables loaded from `tokenizer.bin` instead of built from JSON.
var g_tok_cache: ?TokCache = null;

// ---- tokenizer cache: the tables, written once and mapped later -------------
// Layout, all little-endian, arrays in slot order:
//   0  u32 magic         4  u32 version
//   8  u64 src_size     16  u64 src_mtime      (of tokenizer.json: the identity
//   24 u32 vocab_slots  28  u32 merges_slots    check that invalidates the file)
//   32 u32 added_slots  36  u32 vocab_live
//   40 u32 merges_live  44  u32 added_live
//   48 u64 pool_len     56  u64 arr_bytes
//   [vocab: hash u64[] off u32[] len u32[] val u32[]]
//   [merges: ...] [added: ...] [pool bytes]
// Every slot count is a power of two >= 4096, so each table's block
// (slots*8 + 3*slots*4 bytes) stays 8-byte aligned and the arrays can be sliced
// straight out of the mapping.
const TC_MAGIC: u32 = 0x4B5A544C; // "LTZK"
const TC_VERSION: u32 = 1;
const tc_header: usize = 64;

const TokCache = struct {
    map: std.Io.File.MemoryMap,
    vocab: TokMap,
    merges: TokMap,
    added: TokMap,
};

/// Bytes one table's block occupies in the cache: slots * (8 for the hash +
/// 3 * 4 for off/len/val).
fn tcArrayBytes(v: usize, m: usize, a: usize) usize {
    return (v + m + a) * 20;
}

/// Best effort: on a read-only model directory the write simply fails and the
/// next run pays the JSON parse again.
fn tokCacheSave(alloc: Allocator, io: std.Io, cache_path: []const u8, src_size: u64, src_mtime: u64, pool: []const u8, vocab: TokMap, merges: TokMap, added: TokMap) void {
    saveImpl(alloc, io, cache_path, src_size, src_mtime, pool, vocab, merges, added) catch {};
}

fn saveImpl(alloc: Allocator, io: std.Io, cache_path: []const u8, src_size: u64, src_mtime: u64, pool: []const u8, vocab: TokMap, merges: TokMap, added: TokMap) !void {
    const arr = tcArrayBytes(vocab.hash.len, merges.hash.len, added.hash.len);
    const buf = try alloc.alloc(u8, tc_header + arr + pool.len);
    defer alloc.free(buf);
    @memset(buf[0..tc_header], 0);
    std.mem.writeInt(u32, buf[0..4], TC_MAGIC, .little);
    std.mem.writeInt(u32, buf[4..8], TC_VERSION, .little);
    std.mem.writeInt(u64, buf[8..16], src_size, .little);
    std.mem.writeInt(u64, buf[16..24], src_mtime, .little);
    std.mem.writeInt(u32, buf[24..28], @intCast(vocab.hash.len), .little);
    std.mem.writeInt(u32, buf[28..32], @intCast(merges.hash.len), .little);
    std.mem.writeInt(u32, buf[32..36], @intCast(added.hash.len), .little);
    std.mem.writeInt(u32, buf[36..40], vocab.live, .little);
    std.mem.writeInt(u32, buf[40..44], merges.live, .little);
    std.mem.writeInt(u32, buf[44..48], added.live, .little);
    std.mem.writeInt(u64, buf[48..56], pool.len, .little);
    std.mem.writeInt(u64, buf[56..64], arr, .little);
    var at: usize = tc_header;
    for ([3]TokMap{ vocab, merges, added }) |m| {
        for (m.hash) |v| {
            std.mem.writeInt(u64, buf[at..][0..8], v, .little);
            at += 8;
        }
        for ([3][]const u32{ m.off, m.len, m.val }) |col| {
            for (col) |v| {
                std.mem.writeInt(u32, buf[at..][0..4], v, .little);
                at += 4;
            }
        }
    }
    @memcpy(buf[at..][0..pool.len], pool);
    const f = try std.Io.Dir.cwd().createFile(io, cache_path, .{ .truncate = true });
    defer f.close(io);
    try f.writePositionalAll(io, buf, 0);
}

fn isPow2Slots(n: usize) bool {
    return n >= 4096 and n <= (1 << 26) and (n & (n - 1)) == 0;
}

fn tokCacheLoad(io: std.Io, cache_path: []const u8, src_size: u64, src_mtime: u64) ?TokCache {
    const mm = mapFile(io, cache_path) catch return null;
    const m = mm.memory;
    if (m.len < tc_header) return null;
    if (std.mem.readInt(u32, m[0..4], .little) != TC_MAGIC) return null;
    if (std.mem.readInt(u32, m[4..8], .little) != TC_VERSION) return null;
    if (std.mem.readInt(u64, m[8..16], .little) != src_size) return null;
    if (std.mem.readInt(u64, m[16..24], .little) != src_mtime) return null;
    const vs: usize = std.mem.readInt(u32, m[24..28], .little);
    const ms: usize = std.mem.readInt(u32, m[28..32], .little);
    const as: usize = std.mem.readInt(u32, m[32..36], .little);
    const live = [3]u32{
        std.mem.readInt(u32, m[36..40], .little),
        std.mem.readInt(u32, m[40..44], .little),
        std.mem.readInt(u32, m[44..48], .little),
    };
    const pool_len: usize = @intCast(std.mem.readInt(u64, m[48..56], .little));
    const arr: usize = @intCast(std.mem.readInt(u64, m[56..64], .little));
    if (!isPow2Slots(vs) or !isPow2Slots(ms) or !isPow2Slots(as)) return null;
    if (arr != tcArrayBytes(vs, ms, as)) return null;
    if (tc_header + arr + pool_len != m.len) return null;
    const pool = m[tc_header + arr ..][0..pool_len];
    const base = @intFromPtr(m.ptr);
    var at: usize = tc_header;
    var maps: [3]TokMap = undefined;
    for (&maps, [3]usize{ vs, ms, as }, live) |*t, slots, nlive| {
        t.* = .{ .pool = pool, .live = nlive };
        const hp: [*]const u64 = @ptrCast(@alignCast(m.ptr + at));
        t.hash = hp[0..slots];
        at += slots * 8;
        inline for ([3]*[]const u32{ &t.off, &t.len, &t.val }) |col| {
            const p: [*]const u32 = @ptrCast(@alignCast(m.ptr + at));
            col.* = p[0..slots];
            at += slots * 4;
        }
        if ((base + at) % 8 != 0) at += 8 - ((base + at) % 8);
    }
    return .{ .map = mm, .vocab = maps[0], .merges = maps[1], .added = maps[2] };
}

/// Everything that is the same whether the tables came from JSON or from the
/// cache: the added-token length index, the first-byte set and the byte fallback
/// ids.
fn finishTokenizer(alloc: Allocator, vocab: TokMap, merges: TokMap, added: TokMap) !Tokenizer {
    var tok = Tokenizer{
        .vocab = vocab,
        .merges = merges,
        .added = added,
        .added_lens = &.{},
        .start_byte = undefined,
        .byte_ids = undefined,
    };
    @memset(tok.start_byte[0..], false);
    @memset(tok.byte_ids[0..], 0);

    var lens = AList(usize).init(alloc);
    for (added.hash, 0..) |h, i| {
        if (h == 0) continue;
        const k = added.pool[added.off[i]..][0..added.len[i]];
        if (k.len == 0) continue;
        var seen = false;
        for (lens.items) |x| {
            if (x == k.len) seen = true;
        }
        if (!seen) try lens.append(k.len);
        tok.start_byte[k[0]] = true;
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

fn loadTokenizer(alloc: Allocator, io: std.Io, path: []const u8) !Tokenizer {
    // tokenizer.json's size and mtime are the cache's identity check.
    var src_size: u64 = 0;
    var src_mtime: u64 = 0;
    var have_src = false;
    if (std.Io.Dir.cwd().openFile(io, path, .{})) |f| {
        defer f.close(io);
        if (f.stat(io)) |st| {
            src_size = st.size;
            src_mtime = @intCast(st.mtime.nanoseconds);
            have_src = true;
        } else |_| {}
    } else |_| {}

    const cache_path = try std.fs.path.join(alloc, &.{ std.fs.path.dirname(path) orelse ".", "tokenizer.bin" });
    defer alloc.free(cache_path);

    if (have_src) {
        if (tokCacheLoad(io, cache_path, src_size, src_mtime)) |c| {
            g_tok_cache = c;
            return finishTokenizer(alloc, c.vocab, c.merges, c.added);
        }
    }

    // ---- no usable cache: parse tokenizer.json. The keys point into this buffer
    // rather than into copies, so it has to live as long as the tokenizer: a
    // mapping of the file when one can be made, a plain read otherwise.
    var data: []const u8 = undefined;
    if (mapFile(io, path)) |mm| {
        g_tok_file = mm;
        data = mm.memory;
    } else |_| data = try readFile(std.heap.page_allocator, io, path);

    var tok = Tokenizer{
        .vocab = .{},
        .merges = .{},
        .added = .{},
        .added_lens = &.{},
        .start_byte = undefined,
        .byte_ids = undefined,
    };

    var sc = JScan{ .s = data, .i = 0 };
    var sink = TokSink{ .tok = &tok, .alloc = alloc, .pool = BytePool.init(alloc) };
    walkValue(&sc, &sink) catch |e| {
        const lo = if (sc.i > 120) sc.i - 120 else 0;
        const hi = @min(data.len, sc.i + 120);
        std.debug.print("tokenizer.json parse error {} at offset {d}\n...{s}...\n", .{ e, sc.i, data[lo..hi] });
        return e;
    };

    if (g_prof) std.debug.print("[toksplit] vocab {d} ms, merges {d} ms, added {d} ms\n", .{ @divTrunc(sink.t_vocab, 1_000_000), @divTrunc(sink.t_merges, 1_000_000), @divTrunc(sink.t_added, 1_000_000) });

    // One contiguous pool for all three tables. The offsets were handed out
    // against this exact layout, so they still mean the same thing, and the maps
    // end up with the single slice `get` needs.
    const pool_bytes = try alloc.alloc(u8, sink.pool.total());
    sink.pool.serializeInto(pool_bytes);
    tok.vocab.pool = pool_bytes;
    tok.merges.pool = pool_bytes;
    tok.added.pool = pool_bytes;

    if (have_src) tokCacheSave(alloc, io, cache_path, src_size, src_mtime, pool_bytes, tok.vocab, tok.merges, tok.added);
    return finishTokenizer(alloc, tok.vocab, tok.merges, tok.added);
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
    wqkv: []align(1) const f16,
    wo: []align(1) const f16,
    wi: []align(1) const f16,
    wo2: []align(1) const f16,
    attn_norm: ?[]const f32,
    mlp_norm: []const f32,
    sliding: bool,
};

const HeadLayer = struct {
    norm1_w: []const f32,
    norm1_b: []const f32,
    norm2_w: []const f32,
    norm2_b: []const f32,
    in_proj_w: []align(1) const f16,
    in_proj_b: []const f32,
    out_proj_w: []align(1) const f16,
    out_proj_b: []const f32,
    lin1_w: []align(1) const f16,
    lin1_b: []const f32,
    lin2_w: []align(1) const f16,
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

/// Buffers for `n` tokens (plus the two fixed-size ones). `att` rows are as wide
/// as a sliding window needs, or `n` when that is wider.
fn allocScratch(gpa: Allocator, n: usize, k: usize) !Scratch {
    const lens = [13]usize{
        n * D,     n * D,   n * D, n * D, n * D, n * 3 * D, n * D,
        n * 2 * INTER, n * INTER, n * @max(n, 2 * WINDOW + 1), n * HEAD_FF, k, D + 4,
    };
    var s: Scratch = undefined;
    inline for (@typeInfo(Scratch).@"struct".field_names, 0..) |name, i| {
        @field(s, name) = try gpa.alloc(f32, lens[i]);
    }
    return s;
}

fn freeScratch(gpa: Allocator, s: Scratch) void {
    inline for (@typeInfo(Scratch).@"struct".field_names) |name| gpa.free(@field(s, name));
}

const Cfg = struct {
    temperature: [3]f32 = .{ 1.0, 1.0, 1.0 },
    max_len: usize = MAX_LEN,
    head_max_len: usize = HEAD_MAX_LEN,
};

/// The weights are a view of `model.safetensors`, not a copy: 614 MB of private
/// commit becomes demand-paged file memory the OS can drop and re-read at will.
/// `m.emb_raw` points into it, so it lives as long as the process.
var g_weights: ?std.Io.File.MemoryMap = null;

fn loadModel(alloc: Allocator, io: std.Io, dir: []const u8, cfg: *const Cfg) !struct { Model, []u8 } {
    const path = try std.fs.path.join(alloc, &.{ dir, "model.safetensors" });
    defer alloc.free(path);
    var data: []u8 = undefined;
    if (mapFile(io, path)) |mm| {
        g_weights = mm;
        data = mm.memory;
    } else |_| data = try readFile(alloc, io, path);
    // Only the fallback copy is ours to free; unmapping a view with
    // alloc.free would hand the allocator a pointer it never handed out.
    errdefer if (g_weights == null) alloc.free(data);

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

    // Two buffers, not one: `pre` must not point into the buffer being written, or
    // bufPrint copies a slice onto itself -- harmless under ReleaseFast, but @memcpy
    // asserts on overlap in Debug/Safe and panics at startup.
    var pre: [24]u8 = undefined;
    var nm: [160]u8 = undefined;
    for (0..NL) |l| {
        const pfx = try std.fmt.bufPrint(&pre, "encoder.layers.{d}.", .{l});
        var an: ?[]const f32 = null;
        if (l > 0) an = try tensorF32(alloc, &infos, data, try std.fmt.bufPrint(&nm, "{s}attn_norm.weight", .{pfx}));
        m.layers[l] = .{
            .wqkv = try tensorF16(&infos, data, try std.fmt.bufPrint(&nm, "{s}attn.Wqkv.weight", .{pfx})),
            .wo = try tensorF16(&infos, data, try std.fmt.bufPrint(&nm, "{s}attn.Wo.weight", .{pfx})),
            .wi = try tensorF16(&infos, data, try std.fmt.bufPrint(&nm, "{s}mlp.Wi.weight", .{pfx})),
            .wo2 = try tensorF16(&infos, data, try std.fmt.bufPrint(&nm, "{s}mlp.Wo.weight", .{pfx})),
            .attn_norm = an,
            .mlp_norm = try tensorF32(alloc, &infos, data, try std.fmt.bufPrint(&nm, "{s}mlp_norm.weight", .{pfx})),
            // config: layer_types[i] == "full_attention" iff i % global_attn_every_n_layers == 0 (3)
            .sliding = (l % 3) != 0,
        };
    }

    for (0..HEAD_LAYERS) |l| {
        const pfx = try std.fmt.bufPrint(&pre, "head.layers.{d}.", .{l});
        m.head[l] = .{
            .norm1_w = try tensorF32(alloc, &infos, data, try std.fmt.bufPrint(&nm, "{s}norm1.weight", .{pfx})),
            .norm1_b = try tensorF32(alloc, &infos, data, try std.fmt.bufPrint(&nm, "{s}norm1.bias", .{pfx})),
            .norm2_w = try tensorF32(alloc, &infos, data, try std.fmt.bufPrint(&nm, "{s}norm2.weight", .{pfx})),
            .norm2_b = try tensorF32(alloc, &infos, data, try std.fmt.bufPrint(&nm, "{s}norm2.bias", .{pfx})),
            .in_proj_w = try tensorF16(&infos, data, try std.fmt.bufPrint(&nm, "{s}self_attn.in_proj_weight", .{pfx})),
            .in_proj_b = try tensorF32(alloc, &infos, data, try std.fmt.bufPrint(&nm, "{s}self_attn.in_proj_bias", .{pfx})),
            .out_proj_w = try tensorF16(&infos, data, try std.fmt.bufPrint(&nm, "{s}self_attn.out_proj.weight", .{pfx})),
            .out_proj_b = try tensorF32(alloc, &infos, data, try std.fmt.bufPrint(&nm, "{s}self_attn.out_proj.bias", .{pfx})),
            .lin1_w = try tensorF16(&infos, data, try std.fmt.bufPrint(&nm, "{s}linear1.weight", .{pfx})),
            .lin1_b = try tensorF32(alloc, &infos, data, try std.fmt.bufPrint(&nm, "{s}linear1.bias", .{pfx})),
            .lin2_w = try tensorF16(&infos, data, try std.fmt.bufPrint(&nm, "{s}linear2.weight", .{pfx})),
            .lin2_b = try tensorF32(alloc, &infos, data, try std.fmt.bufPrint(&nm, "{s}linear2.bias", .{pfx})),
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

/// one head's score against one key: 64 MACs in 8 vector FMAs
inline fn dot64(a: []const f32, ao: usize, b: []const f32, bo: usize) f32 {
    var a0 = v8(a, ao) * v8(b, bo);
    var a1 = v8(a, ao + 8) * v8(b, bo + 8);
    var a2 = v8(a, ao + 16) * v8(b, bo + 16);
    var a3 = v8(a, ao + 24) * v8(b, bo + 24);
    a0 = @mulAdd(V, v8(a, ao + 32), v8(b, bo + 32), a0);
    a1 = @mulAdd(V, v8(a, ao + 40), v8(b, bo + 40), a1);
    a2 = @mulAdd(V, v8(a, ao + 48), v8(b, bo + 48), a2);
    a3 = @mulAdd(V, v8(a, ao + 56), v8(b, bo + 56), a3);
    return hsum((a0 + a1) + (a2 + a3));
}

fn attWorker(ctx: *anyopaque, lo: usize, hi: usize) void {
    const a: *AttJob = @ptrCast(@alignCast(ctx));
    const L = a.L;
    const scale: f32 = 1.0 / @sqrt(@as(f32, HD));
    for (lo..hi) |i| {
        const row = a.out[i * D ..][0..D];
        var klo: usize = 0;
        var khi: usize = L;
        var off: usize = 0;
        if (a.win) |w| {
            klo = if (i >= w) i - w else 0;
            khi = @min(L, i + w + 1);
            off = if (i >= w) 0 else w - i;
        }
        const cnt = khi - klo;
        const rowp = a.att[i * a.stride + off ..];
        for (0..H) |h| {
            const qo = i * D + h * HD;
            var mx: f32 = -std.math.inf(f32);
            for (klo..khi) |j| {
                const s = dot64(a.q, qo, a.k, j * D + h * HD) * scale;
                rowp[j - klo] = s;
                mx = @max(mx, s);
            }
            var sv_acc: V = splatv(0);
            const mxv = @as(V, @splat(mx));
            var tv: usize = 0;
            while (tv + 8 <= cnt) : (tv += 8) {
                const e = expv(v8(rowp, tv) - mxv);
                rowp[tv..][0..8].* = @as([8]f32, @bitCast(e));
                sv_acc += e;
            }
            var sum = hsum(sv_acc);
            while (tv < cnt) : (tv += 1) {
                const e = @exp(rowp[tv] - mx);
                rowp[tv] = e;
                sum += e;
            }
            const inv = 1.0 / sum;
            // the head's 64 outputs stay in registers across the whole key range;
            // accumulating through `row` instead turned every key into 8 loads
            // and 8 stores. Normalising afterwards rather than per key takes the
            // multiply out of the inner loop, and `sum >= 1` so dropping a weight
            // below 1e-20 cannot change a single output bit.
            var r: [HD / 8]V = undefined;
            inline for (0..HD / 8) |c| r[c] = splatv(0);
            for (0..cnt) |t| {
                const p = rowp[t];
                if (p < 1e-20) continue;
                const vo = (klo + t) * D + h * HD;
                const pv: V = @splat(p);
                inline for (0..HD / 8) |c| r[c] = @mulAdd(V, pv, v8(a.v, vo + c * 8), r[c]);
            }
            const iv: V = @splat(inv);
            const base = h * HD;
            inline for (0..HD / 8) |c| row[base + c * 8 ..][0..8].* = @as([8]f32, @bitCast(r[c] * iv));
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
        const m = f.s.mlp[i * 2 * INTER ..];
        const o = f.s.mlp2[i * INTER ..];
        var j: usize = 0;
        while (j + 8 <= INTER) : (j += 8) {
            const g = geluv(v8(m, j)) * v8(m, j + INTER);
            o[j..][0..8].* = @as([8]f32, @bitCast(g));
        }
        while (j < INTER) : (j += 1) o[j] = gelu(m[j]) * m[j + INTER];
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

const SplitJob = struct {
    q: []f32,
    k: []f32,
    v: []f32,
    qkv: []const f32,
};

fn splitJob(ctx: *anyopaque, lo: usize, hi: usize) void {
    const s: *SplitJob = @ptrCast(@alignCast(ctx));
    for (lo..hi) |i| {
        const src = s.qkv[i * 3 * D ..];
        @memcpy(s.q[i * D ..][0..D], src[0..D]);
        @memcpy(s.k[i * D ..][0..D], src[D .. 2 * D]);
        @memcpy(s.v[i * D ..][0..D], src[2 * D .. 3 * D]);
    }
}

fn splitQkv(alloc: Allocator, q: []f32, k: []f32, v: []f32, qkv: []const f32, L: usize) void {
    var job = SplitJob{ .q = q, .k = k, .v = v, .qkv = qkv };
    parallel(alloc, L, &job, splitJob);
}

fn forward(alloc: Allocator, f: *FwdCtx) void {
    const m = f.m;
    const s = f.s;
    const L = f.L;

    // ---- embeddings + norm (layer 0 has no attn_norm; embeddings.norm is folded in)
    g_stage = ST_EMB;
    parallel(alloc, L, f, embedJob);
    g_stage = ST_NORM;
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
        g_stage = ST_NORM;
        if (lay.attn_norm) |an| {
            normInto(alloc, s.x, s.xn, L, an, null);
            attn_in = s.xn;
        }
        g_stage = ST_MATMUL;
        linear16(alloc, s.qkv, attn_in, lay.wqkv, null, L, 3 * D, D);
        g_stage = ST_SPLIT;
        splitQkv(alloc, s.q, s.k, s.v, s.qkv, L);
        g_stage = ST_ROPE;
        ropeApply(alloc, s.q, L, &m.inv_freq);
        ropeApply(alloc, s.k, L, &m.inv_freq);

        g_stage = ST_ATTN;
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
        g_stage = ST_MATMUL;
        linear16(alloc, s.xn, s.ctx, lay.wo, null, L, D, D);
        g_stage = ST_RES;
        parallel(alloc, L, f, addResJob);

        g_stage = ST_NORM;
        normInto(alloc, s.x, s.xn, L, lay.mlp_norm, null);
        g_stage = ST_MATMUL;
        linear16(alloc, s.mlp, s.xn, lay.wi, null, L, 2 * INTER, D);
        g_stage = ST_GELU;
        parallel(alloc, L, f, mlpJob);
        g_stage = ST_MATMUL;
        linear16(alloc, s.xn, s.mlp2, lay.wo2, null, L, D, INTER);
        g_stage = ST_RES;
        parallel(alloc, L, f, addResJob);
        if (dumping) {
            var tag: [8]u8 = undefined;
            const t = std.fmt.bufPrint(&tag, "l{d}", .{l}) catch "l?";
            dumpStats(t, s.x, L);
        }
    }

    g_stage = ST_NORM;
    for (0..L) |i| layerNorm(s.x[i * D ..][0..D], m.final_norm, null);
    if (dumping) dumpStats("final", s.x, L);
    parallel(alloc, L, f, typeEmbJob);
    if (dumping) dumpStats("temb", s.x, L);

    // ---- decision head: 2 pre-norm transformer layers, bidirectional
    for (0..HEAD_LAYERS) |l| {
        const hl = &m.head[l];
        g_stage = ST_NORM;
        normInto(alloc, s.x, s.xn, L, hl.norm1_w, hl.norm1_b);
        g_stage = ST_MATMUL;
        linear16(alloc, s.qkv, s.xn, hl.in_proj_w, hl.in_proj_b, L, 3 * D, D);
        g_stage = ST_SPLIT;
        splitQkv(alloc, s.q, s.k, s.v, s.qkv, L);
        g_stage = ST_ATTN;
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
        g_stage = ST_MATMUL;
        linear16(alloc, s.xn, s.ctx, hl.out_proj_w, hl.out_proj_b, L, D, D);
        g_stage = ST_RES;
        parallel(alloc, L, f, addResJob);

        g_stage = ST_NORM;
        normInto(alloc, s.x, s.xn, L, hl.norm2_w, hl.norm2_b);
        g_stage = ST_MATMUL;
        linear16(alloc, s.ff, s.xn, hl.lin1_w, hl.lin1_b, L, HEAD_FF, D);
        g_stage = ST_GELU;
        parallel(alloc, L, f, reluJob);
        g_stage = ST_MATMUL;
        linear16(alloc, s.xn, s.ff, hl.lin2_w, hl.lin2_b, L, D, HEAD_FF);
        g_stage = ST_RES;
        parallel(alloc, L, f, addResJob);
        if (dumping) {
            var tag: [8]u8 = undefined;
            const t = std.fmt.bufPrint(&tag, "h{d}", .{l}) catch "h?";
            dumpStats(t, s.x, L);
        }
    }

    @memcpy(f.pooled, s.x[0..D]);

    // ---- option markers -> scorer
    const t_ser = if (g_prof) nowNs() else 0;
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
    if (g_prof) profAdd(ST_SERIAL, nowNs() - t_ser);
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

const Policy = enum { model, greedy, random, tiers, cycle };

const SNAKE_INS = "Which move keeps the snake alive and reaches the food?";

/// Model plus scratch buffers, kept alive across moves so a game never reloads 614 MB.
const initial_cap: usize = 256;

const ModelChooser = struct {
    gpa: Allocator,
    m: *const Model,
    tok: *const Tokenizer,
    cfg: *const Cfg,
    s: Scratch,
    /// tokens the buffers in `s` can hold (never shrinks)
    cap: usize,
    logits: []f32,
    pooled: []f32,
    ms: i64 = 0,
    last: WebSession.LastMove = .{ .dir = .up, .probs = .{ 0.25, 0.25, 0.25, 0.25 }, .conf = 0, .act = 0, .tokens = 0, .ms = 0 },

    fn init(gpa: Allocator, m: *const Model, tok: *const Tokenizer, cfg: *const Cfg) !ModelChooser {
        const mc = ModelChooser{
            .gpa = gpa,
            .m = m,
            .tok = tok,
            .cfg = cfg,
            .s = try allocScratch(gpa, initial_cap, 8),
            .cap = initial_cap,
            .logits = try gpa.alloc(f32, 8),
            .pooled = try gpa.alloc(f32, D),
        };
        return mc;
    }

    /// Grow the buffers to hold `L` tokens. The chooser outlives the question and
    /// `max_len` is 1024, but a Snake or web prompt is 50-200 tokens: allocating
    /// for the cap rather than for the prompt cost ~60 MB of commit the OS could
    /// never take back. Buffers only grow, so the usual cost is zero allocations
    /// per move.
    fn ensure(self: *ModelChooser, L: usize) !void {
        if (L <= self.cap) return;
        const n = @max(L, initial_cap);
        const old = self.s;
        self.s = try allocScratch(self.gpa, n, 8);
        self.cap = n;
        freeScratch(self.gpa, old);
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
        try self.ensure(sq.ids.len);
        if (g_prof) profReset();
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
        if (g_prof) try profReport(status);

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

        // `--promptv 6` treats death as a constraint of the environment rather
        // than a preference to be learned: the model then ranks only the moves
        // that are still legal. `--promptv 7` also drops the moves that step
        // away from the food, leaving the model to break the tie between the
        // survivors that close the distance. `--promptv 8` replaces that single
        // distance rule with the tiered one in snake.zig, where a move only
        // counts if the snake can still reach the food afterwards. Without a
        // mask the reported probabilities are the model's own, fatal ones
        // included.
        if (snake.prompt_variant >= 6) {
            var allowed: [4]bool = .{ true, true, true, true };
            if (snake.prompt_variant == 8) allowed = g.rankedMoves();
            if (snake.prompt_variant == 9) allowed = g.cycleMoves();
            var left: f32 = 0;
            for (0..k) |r| {
                const keep = switch (snake.prompt_variant) {
                    8, 9 => allowed[r],
                    7 => g.moveIsBest(snake.DIRS[r]),
                    else => !g.moveIsFatal(snake.DIRS[r]),
                };
                if (keep) {
                    left += p[r];
                } else {
                    p[r] = 0;
                }
            }
            if (left > 0) {
                for (0..k) |r| p[r] /= left;
            }
        }

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
    /// which rung of the prompt ladder the page is playing (`?v=` on /api/new)
    variant: u8 = 0,

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
        const tail = g.tailCell();
        for (snake.DIRS, 0..) |d, i| {
            self.fatal[i] = if (g.neighbour(h, d)) |nb|
                // The tail vacates on this very move, so stepping onto it is how
                // a snake that fills the board finishes; only leaving the board
                // or entering the rest of the body is fatal.
                g.occupied[nb[0] * g.size + nb[1]] and !(nb[0] == tail[0] and nb[1] == tail[1])
            else
                true;
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
            .greedy, .random, .tiers, .cycle => {
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
            if (g.alive) "" else if (g.won) "won" else if (g.hit_wall) "wall" else "self",
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
        // The prompt ladder is a global the chooser reads, so it belongs to the
        // session: the page asks for one rung per game and every later request
        // puts it back, whatever the last request left behind.
        self.variant = @intCast(@max(0, @min(req.intParam("v", @as(i64, @intCast(snake.prompt_variant))), 9)));
        snake.prompt_variant = self.variant;
        self.policy = if (std.mem.eql(u8, pol, "greedy"))
            .greedy
        else if (std.mem.eql(u8, pol, "random"))
            .random
        else if (std.mem.eql(u8, pol, "tiers"))
            .tiers
        else if (std.mem.eql(u8, pol, "cycle"))
            .cycle
        else
            .model;
        self.base = switch (self.policy) {
            .random => snake.randomChooser(),
            .tiers => snake.tierChooser(),
            .cycle => snake.cycleChooser(),
            else => snake.greedyChooser(),
        };
        try self.newGame(size, seed);
        try self.writeState(out, true);
        return .{};
    }

    if (std.mem.eql(u8, req.path, "/api/step")) {
        snake.prompt_variant = self.variant;
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

/// The weight file as a read-only view instead of a copy. On Windows this is a
/// section plus a mapping of it, so the handle has to stay open for as long as
/// the view is used -- `MemoryMap` keeps it for exactly that reason.
fn mapFile(io: std.Io, path: []const u8) !std.Io.File.MemoryMap {
    const f = try std.Io.Dir.cwd().openFile(io, path, .{});
    const st = try f.stat(io);
    return f.createMemoryMap(io, .{
        .len = @intCast(st.size),
        .protection = .{ .read = true },
    });
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

/// `--tokdump FILE`: one JSON line per input line, `{"text":..,"ids":[..]}`.
/// Together with `--tokcheck FILE` this is the tokenizer's regression test: the
/// corpus is dumped once from a build that has been verified against the model,
/// and every later change to the tokenizer has to reproduce it byte for byte.
fn tokDump(gpa: Allocator, io: std.Io, tok: *const Tokenizer, path: []const u8, w: *std.Io.Writer) !void {
    const data = try readFile(gpa, io, path);
    defer gpa.free(data);
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const aa = arena.allocator();
    var lines = std.mem.splitScalar(u8, data, '\n');
    while (lines.next()) |ln| {
        if (ln.len == 0) continue;
        const ids = try encodeAll(tok, aa, ln);
        const esc = try std.json.Stringify.valueAlloc(aa, @as(std.json.Value, .{ .string = ln }), .{});
        try w.print("{{\"text\":{s},\"ids\":[", .{esc});
        for (ids, 0..) |id, i| {
            if (i > 0) try w.writeByte(',');
            try w.print("{d}", .{id});
        }
        try w.print("]}}\n", .{});
        _ = arena.reset(.retain_capacity);
    }
    try w.flush();
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
    if (builtin.os.tag == .windows) {
        var fr: i64 = 0;
        if (QueryPerformanceFrequency(&fr) != 0 and fr > 0) g_qpc_freq = fr;
    }

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
    var tokdump: ?[]const u8 = null;
    var play = false;
    var serve = false;
    var selftest = false;
    var bench = false;
    var port: u16 = 8080;
    var s_policy: Policy = .model;
    var s_opts = snake.Config{};
    var snake_prompt_v: u8 = 0;
    var s_steps_set = false;
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
        if (std.mem.eql(u8, a, "--promptv") and i + 1 < args.len) {
            snake_prompt_v = std.fmt.parseInt(u8, args[i + 1], 10) catch 0;
            i += 1;
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
        if (std.mem.eql(u8, a, "--profile")) {
            g_prof = true;
            continue;
        }
        if (std.mem.eql(u8, a, "--benchgemm")) {
            bench = true;
            continue;
        }
        if (std.mem.eql(u8, a, "--pretouch")) {
            g_pretouch = true;
            continue;
        }
        const takes_value = std.mem.eql(u8, a, "--dir") or std.mem.eql(u8, a, "--json") or
            std.mem.eql(u8, a, "--threads") or std.mem.eql(u8, a, "--kernel") or
            std.mem.eql(u8, a, "--tokcheck") or
            std.mem.eql(u8, a, "--tokdump") or
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
        if (std.mem.eql(u8, key, "--tokdump")) tokdump = v;
        if (std.mem.eql(u8, key, "--threads")) g_threads = std.fmt.parseInt(usize, v, 10) catch 0;
        if (std.mem.eql(u8, key, "--kernel")) g_kern_alt = @min(std.fmt.parseInt(u8, v, 10) catch 1, 2);
        if (std.mem.eql(u8, key, "--size")) s_opts.size = @max(6, @min(std.fmt.parseInt(usize, v, 10) catch 10, 24));
        if (std.mem.eql(u8, key, "--games")) s_opts.games = @max(1, std.fmt.parseInt(usize, v, 10) catch 1);
        if (std.mem.eql(u8, key, "--delay")) s_opts.delay_ms = std.fmt.parseInt(i64, v, 10) catch 0;
        if (std.mem.eql(u8, key, "--seed")) s_opts.seed = std.fmt.parseInt(u64, v, 10) catch 12345;
        if (std.mem.eql(u8, key, "--max-steps")) {
            s_opts.max_steps = std.fmt.parseInt(usize, v, 10) catch 0;
            s_steps_set = s_opts.max_steps > 0;
        }
        if (std.mem.eql(u8, key, "--port")) port = std.fmt.parseInt(u16, v, 10) catch 8080;
        if (std.mem.eql(u8, key, "--policy")) {
            s_policy = if (std.mem.eql(u8, v, "greedy"))
                .greedy
            else if (std.mem.eql(u8, v, "random"))
                .random
            else if (std.mem.eql(u8, v, "tiers"))
                .tiers
            else if (std.mem.eql(u8, v, "cycle"))
                .cycle
            else
                .model;
        }
    }
    if (g_threads == 0) {
        // Measured here (4 P-cores + 8 E-cores, 16 logical): on the 60-180 token
        // prompts this model actually sees, 8 threads is as quick as 12 and costs
        // a third of the CPU-seconds -- the last four only add a straggler for the
        // barrier to wait on. Past ~300 tokens 12 wins again (~15% on a 634).
        if (std.Thread.getCpuCount()) |n| g_threads = @max(1, @min(n, 8)) else |_| {
            g_threads = 4;
        }
    }
    g_threads = @max(1, @min(g_threads, 64));
    snake.prompt_variant = @min(snake_prompt_v, 9);
    if (!s_steps_set) s_opts.max_steps = snake.defaultMaxSteps(s_opts.size);

    var out_buf: [1 << 16]u8 = undefined;
    var fw = std.Io.File.stdout().writer(io, &out_buf);
    const w = &fw.interface;

    const t0 = nowMs(io);

    if (selftest) {
        try w.print("laya selftest (SIMD kernels vs f64)\n", .{});
        try selfTest(gpa, w);
        return;
    }

    g_pool.start(gpa, g_threads);
    defer g_pool.stopAll();

    if (bench) {
        try benchGemm(gpa, w);
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
    // `--tokdump` writes the golden file to stdout, so the banner has to stay out
    // of it; the counts it carries are checked by `--tokcheck` anyway.
    if (tokdump == null) {
        try w.print("tokenizer: {d} vocab / {d} merges / {d} added ({d} ms)\n", .{ tok.vocab.count(), tok.merges.count(), tok.added.count(), t1 - t0 });
        try w.flush();
    }

    if (tokcheck) |tp| {
        try tokCheck(gpa, io, &tok, tp);
        return;
    }
    if (tokdump) |tp| {
        try tokDump(gpa, io, &tok, tp, w);
        return;
    }

    // ---- weights
    const mr = try loadModel(gpa, io, dir, &cfg);
    const m = mr[0];
    const t2 = nowMs(io);
    try w.print("weights: {d} MB, {d} threads ({d} ms)\n", .{ mr[1].len / (1024 * 1024), g_threads, t2 - t1 });
    try w.flush();

    if (g_pretouch) {
        const t3 = nowMs(io);
        pretouch(gpa, mr[1]);
        try w.print("pretouch {d} MB ({d} ms)\n", .{ mr[1].len / (1024 * 1024), nowMs(io) - t3 });
        try w.flush();
    }

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
            .tiers => snake.tierChooser(),
            .cycle => snake.cycleChooser(),
        };
        const name = switch (s_policy) {
            .model => "model",
            .greedy => "greedy (baseline)",
            .random => "random (baseline)",
            .tiers => "tiered rules, no model (baseline)",
            .cycle => "cycle + shortcuts, no model (baseline)",
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
    profReset();
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
    try profReport(w);
    try w.print("\nforward {d} ms | total {d} ms", .{ t3 - t2, t3 - t0 });
    if (peakMem()) |pmc| {
        try w.print(" | peak {d} MB working set, {d} MB commit", .{ pmc.PeakWorkingSetSize / (1024 * 1024), pmc.PeakPagefileUsage / (1024 * 1024) });
    }
    if (cpuSeconds()) |cs| {
        try w.print(" | cpu {d:.2} s", .{cs});
    }
    try w.print("\n", .{});
    try w.flush();
}
