const std = @import("std");

pub fn build(b: *std.Build) void {
    // The matmul/attention kernels are built around 8-wide SIMD, so default to the
    // host CPU model instead of the conservative baseline. `-Dtarget=` still works
    // for cross builds, where the model then falls back to baseline features.
    const target = b.resolveTargetQuery(.{ .cpu_model = .native });

    // This is a numeric workload: optimized by default.
    const optimize = optimizeMode(b);

    const exe = b.addExecutable(.{
        .name = "laya",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    const run_step = b.step("run", "Build and run the laya decision model (no CLI args)");
    run_step.dependOn(&run_cmd.step);

    const exe_check = b.addExecutable(.{
        .name = "laya-check",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = b.graph.host,
            .optimize = .debug,
        }),
    });
    const check_step = b.step("check", "Type-check src/main.zig without installing");
    check_step.dependOn(&exe_check.step);
}

/// `-Doptimize=` accepts both the current names (debug/safe/fast/small) and the
/// spellings this repo has historically used (Debug/ReleaseFast/...).
fn optimizeMode(b: *std.Build) std.builtin.Optimize {
    const name = b.option(
        []const u8,
        "optimize",
        "debug | safe | fast | small (Debug / ReleaseFast / ReleaseSafe / ReleaseSmall also accepted)",
    ) orelse return .fast;

    const table = [_]struct { name: []const u8, mode: std.builtin.Optimize }{
        .{ .name = "debug", .mode = .debug },
        .{ .name = "Debug", .mode = .debug },
        .{ .name = "safe", .mode = .safe },
        .{ .name = "ReleaseSafe", .mode = .safe },
        .{ .name = "fast", .mode = .fast },
        .{ .name = "ReleaseFast", .mode = .fast },
        .{ .name = "small", .mode = .small },
        .{ .name = "ReleaseSmall", .mode = .small },
    };
    for (table) |e| {
        if (std.mem.eql(u8, name, e.name)) return e.mode;
    }
    std.debug.print("unknown -Doptimize={s}; using fast\n", .{name});
    return .fast;
}
