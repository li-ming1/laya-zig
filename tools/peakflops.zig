//! Peak AVX2 f32 FMA throughput probe: what this chip can do with nothing but
//! registers, so a GEMM number has a denominator.
//!
//! build: zig build-exe -OReleaseFast -mcpu native tools/peakflops.zig
//! run:   peakflops.exe [threads]
//!
//! Each thread runs NACC independent 8-wide FMA chains from registers. NACC=12 is
//! more than the FMA latency/throughput product on Golden Cove (4 cyc x 2/cyc = 8)
//! so the loop is port-limited, not dependency-limited, which is what "peak" means.

const std = @import("std");

const V = @Vector(8, f32);
const NACC = 12;
var g_iters: usize = 4_000_000;

var g_best: std.atomic.Value(i64) = .init(std.math.maxInt(i64));
var g_qpc_freq: i64 = 1_000_000_000;

extern "kernel32" fn QueryPerformanceCounter(lpLargeInteger: *i64) callconv(.winapi) i32;
extern "kernel32" fn QueryPerformanceFrequency(lpFrequency: *i64) callconv(.winapi) i32;

fn nowNs() i64 {
    var c: i64 = undefined;
    _ = QueryPerformanceCounter(&c);
    return @divTrunc(c * std.time.ns_per_s, g_qpc_freq);
}

fn work() void {
    var acc: [NACC]V = undefined;
    inline for (0..NACC) |k| acc[k] = @splat(@as(f32, @floatFromInt(k)) * 0.5 + 1.0);
    const a: V = @splat(1.000001);
    const b: V = @splat(0.0000001);
    var i: usize = 0;
    while (i < g_iters) : (i += 1) {
        inline for (0..NACC) |k| acc[k] = @mulAdd(V, acc[k], a, b);
    }
    var s: f32 = 0;
    inline for (0..NACC) |k| s += @reduce(.Add, acc[k]);
    if (s == 12345.678) std.debug.print("{d}\n", .{s}); // keep the loop alive
}

fn threadMain(ns: *std.atomic.Value(i64)) void {
    const t0 = nowNs();
    work();
    const dt: i64 = nowNs() - t0;
    var cur = ns.load(.monotonic);
    while (dt < cur) {
        cur = ns.cmpxchgWeak(cur, dt, .monotonic, .monotonic) orelse break;
    }
}

pub fn main(init: std.process.Init) !void {
    const gpa = std.heap.page_allocator;
    var argit = try std.process.Args.Iterator.initAllocator(init.minimal.args, gpa);
    defer argit.deinit();
    var nt: usize = 8;
    _ = argit.next(); // exe name
    if (argit.next()) |s| nt = std.fmt.parseInt(usize, s, 10) catch 8;
    if (argit.next()) |s| g_iters = std.fmt.parseInt(usize, s, 10) catch 4_000_000;
    if (nt == 0) nt = 1;

    const alloc = std.heap.page_allocator;
    const threads = try alloc.alloc(std.Thread, nt);
    var fr: i64 = 0;
    if (QueryPerformanceFrequency(&fr) != 0 and fr > 0) g_qpc_freq = fr;
    var ns = std.atomic.Value(i64).init(std.math.maxInt(i64));
    const wall0 = nowNs();
    for (threads) |*t| t.* = try std.Thread.spawn(.{ .stack_size = 128 * 1024 }, threadMain, .{&ns});
    for (threads) |t| t.join();
    const wall: f64 = @floatFromInt(nowNs() - wall0);
    _ = &g_best;

    const per_thread = 2.0 * 8.0 * @as(f64, NACC) * @as(f64, @floatFromInt(g_iters)); // flops
    const slowest: f64 = @floatFromInt(ns.load(.monotonic));
    const total = per_thread * @as(f64, @floatFromInt(nt));
    std.debug.print("{d} threads: {d:.1} GFLOPS aggregate ({d:.1} per thread, slowest thread)\n", .{
        nt, total / (slowest / 1e9) / 1e9, per_thread / (slowest / 1e9) / 1e9,
    });
    std.debug.print("wall {d:.1} ms\n", .{wall / 1e6});
}
