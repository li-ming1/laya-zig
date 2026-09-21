//! A small Snake game, used as a decision task for the Laya model.
//!
//! The game owns the rules, the rendering and the baseline policies. Whatever is
//! going to choose the moves is supplied as a `Chooser`, so this file has no
//! dependency on the model runtime: `main.zig` plugs the model in there.
//!
//! The board is rendered as text and handed to the chooser as `Snake.stateText`;
//! that text is exactly what the model sees as its `state` argument.

const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Dir = enum(u8) {
    up,
    down,
    left,
    right,

    pub fn delta(self: Dir) [2]isize {
        return switch (self) {
            .up => .{ -1, 0 },
            .down => .{ 1, 0 },
            .left => .{ 0, -1 },
            .right => .{ 0, 1 },
        };
    }
    pub fn opposite(self: Dir) Dir {
        return switch (self) {
            .up => .down,
            .down => .up,
            .left => .right,
            .right => .left,
        };
    }
    pub fn label(self: Dir) []const u8 {
        return switch (self) {
            .up => "up",
            .down => "down",
            .left => "left",
            .right => "right",
        };
    }
};

pub const DIRS = [_]Dir{ .up, .down, .left, .right };

pub const Snake = struct {
    size: usize,
    body: std.array_list.Managed([2]usize), // [0] is the head
    occupied: []bool,
    dir: Dir,
    food: [2]usize,
    rng: std.Random.DefaultPrng,
    score: usize,
    steps: usize,
    alive: bool,
    hit_wall: bool,
    illegal: usize,

    pub fn init(alloc: Allocator, size: usize, seed: u64) !Snake {
        var g = Snake{
            .size = size,
            .body = std.array_list.Managed([2]usize).init(alloc),
            .occupied = try alloc.alloc(bool, size * size),
            .dir = .right,
            .food = .{ 0, 0 },
            .rng = std.Random.DefaultPrng.init(seed),
            .score = 0,
            .steps = 0,
            .alive = true,
            .hit_wall = false,
            .illegal = 0,
        };
        try g.reset();
        return g;
    }

    pub fn deinit(self: *Snake) void {
        self.body.deinit();
    }

    fn idx(self: *const Snake, r: usize, c: usize) usize {
        return r * self.size + c;
    }

    pub fn reset(self: *Snake) !void {
        @memset(self.occupied, false);
        self.body.clearRetainingCapacity();
        const mid = self.size / 2;
        for (0..3) |k| {
            const c = mid - k;
            try self.body.append(.{ mid, c });
            self.occupied[self.idx(mid, c)] = true;
        }
        self.dir = .right;
        self.score = 0;
        self.steps = 0;
        self.alive = true;
        self.hit_wall = false;
        self.illegal = 0;
        self.spawnFood();
    }

    fn spawnFood(self: *Snake) void {
        var free: usize = 0;
        for (0..self.size * self.size) |i| {
            if (!self.occupied[i]) free += 1;
        }
        if (free == 0) { // board full: the snake won
            self.alive = false;
            return;
        }
        var pick = self.rng.random().uintLessThan(usize, free);
        for (0..self.size) |r| {
            for (0..self.size) |c| {
                if (self.occupied[self.idx(r, c)]) continue;
                if (pick == 0) {
                    self.food = .{ r, c };
                    return;
                }
                pick -= 1;
            }
        }
    }

    pub fn head(self: *const Snake) [2]usize {
        return self.body.items[0];
    }

    /// the cell a move in `d` would enter, or null when it leaves the board
    pub fn neighbour(self: *const Snake, p: [2]usize, d: Dir) ?[2]usize {
        const dl = d.delta();
        const r = @as(isize, @intCast(p[0])) + dl[0];
        const c = @as(isize, @intCast(p[1])) + dl[1];
        if (r < 0 or c < 0 or r >= self.size or c >= self.size) return null;
        return .{ @intCast(r), @intCast(c) };
    }

    pub fn step(self: *Snake, want: Dir) void {
        if (!self.alive) return;
        self.steps += 1;
        if (want == self.dir.opposite()) {
            // a 180 degree turn is not a legal snake move; the snake keeps going
            self.illegal += 1;
        } else {
            self.dir = want;
        }
        const h = self.head();
        const n = self.neighbour(h, self.dir) orelse {
            self.alive = false;
            self.hit_wall = true;
            return;
        };
        const eating = (n[0] == self.food[0] and n[1] == self.food[1]);
        const tail = self.body.items[self.body.items.len - 1];
        const into_tail = (n[0] == tail[0] and n[1] == tail[1]);
        if (self.occupied[self.idx(n[0], n[1])] and !(into_tail and !eating)) {
            self.alive = false; // ran into its own body
            return;
        }
        if (eating) {
            self.score += 1;
        } else {
            const t = self.body.pop().?;
            self.occupied[self.idx(t[0], t[1])] = false;
        }
        self.body.insert(0, n) catch unreachable;
        self.occupied[self.idx(n[0], n[1])] = true;
        if (eating) self.spawnFood();
    }

    pub fn dist(a: [2]usize, b: [2]usize) usize {
        const dr = if (a[0] > b[0]) a[0] - b[0] else b[0] - a[0];
        const dc = if (a[1] > b[1]) a[1] - b[1] else b[1] - a[1];
        return dr + dc;
    }

    pub fn cellDesc(self: *const Snake, r: usize, c: usize) []const u8 {
        if (r == self.food[0] and c == self.food[1]) return "the food";
        if (r == self.head()[0] and c == self.head()[1]) return "your head";
        if (self.occupied[self.idx(r, c)]) return "your body";
        return "empty";
    }

    /// The text handed to the chooser (and so to the model) for this position.
    /// Kept deliberately terse: the model is quadratic-ish in prompt length, and
    /// ~115 tokens instead of ~295 makes a move roughly 2x faster.
    pub fn stateText(self: *const Snake, alloc: Allocator) ![]u8 {
        var aw: std.Io.Writer.Allocating = .init(alloc);
        defer aw.deinit();
        const w = &aw.writer;
        const h = self.head();
        try w.print("{d}x{d} snake board. H=head o=body *=food .=empty. Board edge and body kill.\n", .{ self.size, self.size });
        for (0..self.size) |r| {
            try w.print("{d} ", .{r});
            for (0..self.size) |c| {
                try w.writeByte(if (r == h[0] and c == h[1]) 'H' else if (r == self.food[0] and c == self.food[1]) '*' else if (self.occupied[self.idx(r, c)]) 'o' else '.');
            }
            try w.writeByte('\n');
        }
        try w.print("head {d},{d}  food {d},{d}  length {d}\n", .{ h[0], h[1], self.food[0], self.food[1], self.body.items.len });
        for (DIRS) |d| {
            if (self.neighbour(h, d)) |nb| {
                try w.print("{s} {d},{d} {s}\n", .{ d.label(), nb[0], nb[1], self.cellDesc(nb[0], nb[1]) });
            } else {
                try w.print("{s} edge\n", .{d.label()});
            }
        }
        return try alloc.dupe(u8, w.buffered());
    }

    /// One description per direction, in DIRS order.
    pub fn optionTexts(self: *const Snake, alloc: Allocator, opts: *[4][]const u8) !void {
        const h = self.head();
        for (DIRS, 0..) |d, i| {
            if (self.neighbour(h, d)) |nb| {
                opts[i] = try std.fmt.allocPrint(alloc, "{s} to {d},{d} ({s})", .{ d.label(), nb[0], nb[1], self.cellDesc(nb[0], nb[1]) });
            } else {
                opts[i] = try std.fmt.allocPrint(alloc, "{s} off the board (dies)", .{d.label()});
            }
        }
    }
};

/// Anything that can pick a move. `status` gets a one-line explanation of the
/// choice, which the game prints under the board.
pub const Chooser = struct {
    ctx: *anyopaque,
    moveFn: *const fn (ctx: *anyopaque, io: std.Io, g: *Snake, status: *std.Io.Writer) anyerror!Dir,
};

pub const Config = struct {
    size: usize = 10,
    games: usize = 1,
    delay_ms: i64 = 0,
    seed: u64 = 12345,
    max_steps: usize = 400,
    show_prompt: bool = false,
};

pub const Stats = struct {
    games: usize = 0,
    score: usize = 0,
    best: usize = 0,
    wall: usize = 0,
    suicide: usize = 0,
    illegal: usize = 0,
    moves: usize = 0,

    fn note(self: *Stats, g: *const Snake) void {
        self.games += 1;
        self.score += g.score;
        self.best = @max(self.best, g.score);
        if (!g.alive) {
            if (g.hit_wall) self.wall += 1 else self.suicide += 1;
        }
        self.illegal += g.illegal;
        self.moves += g.steps;
    }
};

pub fn sleepMs(io: std.Io, ms: i64) void {
    if (ms <= 0) return;
    const d = std.Io.Clock.Duration{ .raw = std.Io.Duration.fromMilliseconds(ms), .clock = .awake };
    d.sleep(io) catch {};
}

fn render(w: *std.Io.Writer, g: *const Snake, status: []const u8, ansi: bool) !void {
    if (ansi) try w.writeAll("\x1b[H\x1b[2J");
    const h = g.head();
    try w.print("score {d}   step {d}   length {d}   heading {s}\n", .{ g.score, g.steps, g.body.items.len, g.dir.label() });
    try w.writeAll("    +");
    for (0..g.size) |_| try w.writeAll("--");
    try w.writeAll("+\n");
    for (0..g.size) |r| {
        try w.print("{d:>2}  |", .{r});
        for (0..g.size) |c| {
            try w.writeAll(if (r == h[0] and c == h[1]) "H " else if (r == g.food[0] and c == g.food[1]) "* " else if (g.occupied[g.idx(r, c)]) "o " else ". ");
        }
        try w.writeAll("|\n");
    }
    try w.writeAll("    +");
    for (0..g.size) |_| try w.writeAll("--");
    try w.writeAll("+\n");
    if (status.len > 0) try w.writeAll(status);
}

/// Play `cfg.games` games with `chooser` and print the board (single game) or a
/// per-game line plus a summary.
pub fn run(gpa: Allocator, io: std.Io, w: *std.Io.Writer, chooser: Chooser, name: []const u8, cfg: Config, ansi: bool) !void {
    var stats = Stats{};
    var g = try Snake.init(gpa, cfg.size, cfg.seed);
    defer g.deinit();

    try w.print("snake {d}x{d}  chooser {s}  seed {d}  max_steps {d}\n", .{ cfg.size, cfg.size, name, cfg.seed, cfg.max_steps });
    if (cfg.games > 1) try w.print("playing {d} games, summary only\n", .{cfg.games});
    try w.flush();

    if (cfg.show_prompt) {
        const txt = try g.stateText(gpa);
        defer gpa.free(txt);
        try w.writeAll("\n--- the state text handed to the chooser ---\n");
        try w.writeAll(txt);
        try w.writeAll("--- end ---\n\n");
        try w.flush();
    }

    for (0..cfg.games) |gi| {
        try g.reset();
        while (g.alive and g.steps < cfg.max_steps) {
            var status: std.Io.Writer.Allocating = .init(gpa);
            defer status.deinit();
            const d = try chooser.moveFn(chooser.ctx, io, &g, &status.writer);
            g.step(d);
            if (cfg.games == 1) {
                try render(w, &g, status.writer.buffered(), ansi);
                try w.flush();
                sleepMs(io, cfg.delay_ms);
            }
        }
        stats.note(&g);
        if (cfg.games > 1) {
            try w.print("  game {d:>3}: score {d:>3}  steps {d:>3}  {s}  illegal {d}\n", .{
                gi + 1, g.score, g.steps, if (g.alive) "board full" else if (g.hit_wall) "hit wall" else "hit itself", g.illegal,
            });
            try w.flush();
        }
    }

    if (cfg.games == 1) try render(w, &g, "", ansi);

    const ngames: f64 = @floatFromInt(stats.games);
    try w.print("\n--- {s} ---\n", .{name});
    try w.print("score: total {d}  best {d}  mean {d:.2}\n", .{ stats.score, stats.best, @as(f64, @floatFromInt(stats.score)) / ngames });
    try w.print("deaths: wall {d}  self {d}   illegal 180-degree moves: {d}\n", .{ stats.wall, stats.suicide, stats.illegal });
    try w.flush();
}

// ------------------------------------------------------------------ baseline choosers

const GreedyCtx = struct {
    /// how many plies of lookahead; 1 = pure "step toward the food"
    depth: usize = 1,
};

fn greedyFn(ctx: *anyopaque, io: std.Io, g: *Snake, status: *std.Io.Writer) anyerror!Dir {
    _ = ctx;
    _ = io;
    const d = greedyMove(g);
    try status.print("greedy: chose {s}\n", .{d.label()});
    return d;
}

fn greedyMove(g: *Snake) Dir {
    const h = g.head();
    var best: Dir = g.dir;
    var best_d: usize = std.math.maxInt(usize);
    for (DIRS) |d| {
        if (d == g.dir.opposite()) continue;
        const n = g.neighbour(h, d) orelse continue;
        if (g.occupied[n[0] * g.size + n[1]]) continue;
        const dd = Snake.dist(n, g.food);
        if (dd < best_d) {
            best_d = dd;
            best = d;
        }
    }
    return best;
}

var greedy_ctx = GreedyCtx{};

pub fn greedyChooser() Chooser {
    return .{ .ctx = &greedy_ctx, .moveFn = greedyFn };
}

fn randomFn(ctx: *anyopaque, io: std.Io, g: *Snake, status: *std.Io.Writer) anyerror!Dir {
    _ = ctx;
    _ = io;
    const d: Dir = DIRS[g.rng.random().uintLessThan(usize, 4)];
    try status.print("random: chose {s}\n", .{d.label()});
    return d;
}

pub fn randomChooser() Chooser {
    return .{ .ctx = &greedy_ctx, .moveFn = randomFn };
}
