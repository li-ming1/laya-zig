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

/// Which way the position is described to the chooser. Only ever changes the
/// text -- the environment, the seeds and the legality checks are untouched, so
/// scores across variants are comparable.
///
/// var  state   option wording
/// 0    grid    terse, with cell coordinates            (what shipped first)
/// 1    grid    one sentence naming what the move does
/// 2    grid    verdict first: "fatal" / "safe, N steps from the food"
/// 3    prose   verdict first
/// 4    prose   one sentence naming what the move does
/// 5    prose   bare geometry in the state; the option carries only a word --
///              "would kill the snake" / "eats the food" / "one step closer"
/// 6    as 5, but directions that die are masked out before the argmax, so the
///      model only ever ranks the moves the environment still allows
/// 7    as 5, but the argmax is restricted to the moves that both survive and
///      close the distance as much as any surviving move does. What is left for
///      the model is the tie between them -- the question it is not asked to
///      answer from geometry it cannot read
pub var prompt_variant: u8 = 0;

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

    /// What one move would actually do, computed from the board rather than
    /// phrased for the model. Every prompt variant below is a different way of
    /// saying these same facts.
    const Move = struct {
        cell: ?[2]usize,
        what: []const u8,
        fatal: bool,
        eats: bool,
        food_dist: usize,
    };

    fn moveOf(self: *const Snake, d: Dir) Move {
        const h = self.head();
        const nb = self.neighbour(h, d) orelse
            return .{ .cell = null, .what = "off the board", .fatal = true, .eats = false, .food_dist = 0 };
        const fd = dist(nb, self.food);
        if (nb[0] == self.food[0] and nb[1] == self.food[1])
            return .{ .cell = nb, .what = "the food", .fatal = false, .eats = true, .food_dist = 0 };
        if (self.occupied[self.idx(nb[0], nb[1])])
            return .{ .cell = nb, .what = "its own body", .fatal = true, .eats = false, .food_dist = fd };
        return .{ .cell = nb, .what = "open ground", .fatal = false, .eats = false, .food_dist = fd };
    }

    fn proseState() bool {
        return prompt_variant >= 3;
    }

    /// `--promptv 6` asks the model to rank only the moves the board still
    /// allows, so the chooser needs the same fact the prompt states.
    pub fn moveIsFatal(self: *const Snake, d: Dir) bool {
        return self.moveOf(d).fatal;
    }

    /// `--promptv 7` narrows the choice one step further: a move counts only if
    /// it survives *and* ends no farther from the food than any surviving move
    /// does. The board then supplies the objective and the model is left with the
    /// tie between equally good survivors.
    pub fn moveIsBest(self: *const Snake, d: Dir) bool {
        const m = self.moveOf(d);
        if (m.fatal) return false;
        var best = m.food_dist;
        for (DIRS) |o| {
            const om = self.moveOf(o);
            if (!om.fatal and om.food_dist < best) best = om.food_dist;
        }
        return m.food_dist == best;
    }

    /// The text handed to the chooser (and so to the model) for this position.
    /// Kept deliberately terse for variant 0: the model is quadratic-ish in prompt
    /// length, and ~115 tokens instead of ~295 makes a move roughly 2x faster.
    pub fn stateText(self: *const Snake, alloc: Allocator) ![]u8 {
        var aw: std.Io.Writer.Allocating = .init(alloc);
        defer aw.deinit();
        const w = &aw.writer;
        const h = self.head();
        if (proseState()) {
            try w.print("Snake on a {d}x{d} board. Reaching the edge of the board or touching its own body kills it, and the food is what the snake wants.\n", .{ self.size, self.size });
            try w.print("The head is at row {d}, column {d}, moving {s}; the body is {d} long.\n", .{ h[0], h[1], self.dir.label(), self.body.items.len });
            try w.print("The food is at row {d}, column {d}, {d} steps away.\n", .{ self.food[0], self.food[1], dist(h, self.food) });
            for (DIRS) |d| {
                const m = self.moveOf(d);
                if (prompt_variant >= 5) {
                    // Geometry only; the verdict lives in the option text, which is
                    // the whole point of this variant.
                    try w.print("Moving {s}: {s}.\n", .{ d.label(), m.what });
                    continue;
                }
                const where = if (m.cell) |c|
                    try std.fmt.allocPrint(alloc, "row {d}, column {d}", .{ c[0], c[1] })
                else
                    "off the board";
                const outcome = if (m.fatal)
                    "fatal"
                else if (m.eats)
                    "it eats the food"
                else
                    try std.fmt.allocPrint(alloc, "safe, {d} steps from the food", .{ m.food_dist });
                try w.print("Moving {s} enters {s}: {s}\n", .{ d.label(), where, outcome });
            }
            return try alloc.dupe(u8, w.buffered());
        }
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
        const hd = dist(self.head(), self.food);
        for (DIRS, 0..) |d, i| {
            const m = self.moveOf(d);
            opts[i] = switch (prompt_variant) {
                5, 6, 7 => if (m.fatal)
                    try std.fmt.allocPrint(alloc, "{s} would kill the snake", .{d.label()})
                else if (m.eats)
                    try std.fmt.allocPrint(alloc, "{s} eats the food", .{d.label()})
                else if (m.food_dist < hd)
                    try std.fmt.allocPrint(alloc, "{s} moves one step closer to the food", .{d.label()})
                else
                    try std.fmt.allocPrint(alloc, "{s} moves one step away from the food", .{d.label()}),
                1, 4 => if (m.fatal)
                    try std.fmt.allocPrint(alloc, "{s} enters {s}: the snake dies", .{ d.label(), m.what })
                else if (m.eats)
                    try std.fmt.allocPrint(alloc, "{s} enters the food: the snake eats and lives", .{d.label()})
                else
                    try std.fmt.allocPrint(alloc, "{s} enters {s}, {d} steps from the food: the snake lives", .{ d.label(), m.what, m.food_dist }),
                2, 3 => if (m.fatal)
                    try std.fmt.allocPrint(alloc, "{s}: fatal, {s}", .{ d.label(), m.what })
                else if (m.eats)
                    try std.fmt.allocPrint(alloc, "{s}: eats the food", .{d.label()})
                else
                    try std.fmt.allocPrint(alloc, "{s}: safe, {d} steps from the food", .{ d.label(), m.food_dist }),
                else => if (m.cell) |c|
                    try std.fmt.allocPrint(alloc, "{s} to {d},{d} ({s})", .{ d.label(), c[0], c[1], self.cellDesc(c[0], c[1]) })
                else
                    try std.fmt.allocPrint(alloc, "{s} off the board (dies)", .{d.label()}),
            };
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
        // Init already reset the board. Resetting again would draw a second food
        // from the RNG, so `--seed N` here and `?seed=N` on the web would start
        // from different positions.
        if (gi > 0) try g.reset();
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
