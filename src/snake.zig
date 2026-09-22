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

/// The flood fill runs on the stack, so it covers boards up to this edge;
/// above it `--promptv 8` degrades to `--promptv 7` instead of allocating.
pub const MAX_BOARD: usize = 32;

fn anyOf(p: [4]bool) bool {
    for (p) |v| {
        if (v) return true;
    }
    return false;
}

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
/// 8    as 5, but the argmax is restricted by a flood fill instead of a distance
///      rule: a move counts only if the snake can still reach the food after it,
///      and when nothing leaves the food reachable the roomiest survivors are
///      offered instead. This is the variant that can grow past a few cells
/// 9    as 5, but only the moves that keep a Hamiltonian cycle over the board
///      intact are offered, plus the shortcuts that cannot break it. The snake
///      then fills the board whatever the model answers, and the model decides
///      which way round the cycle to go. Needs an even number of rows: an odd
///      board has no such cycle and rung 9 degrades to rung 8 there
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
    /// the food could not be placed because every cell is body: the snake won
    won: bool,
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
            .won = false,
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
        self.won = false;
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
            self.won = true;
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

    /// The body cell that will still be occupied longest, i.e. the one the
    /// head wants to be able to step onto: the tail, or the segment before it
    /// when the tail is about to vacate.
    pub fn tailCell(self: *const Snake) [2]usize {
        return self.body.items[self.body.items.len - 1];
    }

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

    /// How far the head can still spread one move later, and whether the food is
    /// in that region. A snake that seals off a corner dies in it later, however
    /// close the food looked. The tail cell counts as free because this very move
    /// vacates it, which keeps the test from firing on a snake that is merely
    /// long instead of one that is trapped.
    const Reach = struct {
        area: usize,
        food_dist: usize, // steps to the food over free cells, `none` if walled off
        tail_dist: usize, // steps to the tail, the cell that is guaranteed to move

        const none = std.math.maxInt(usize);
    };

    fn reachable(self: *const Snake, from: [2]usize, blocked: *const [MAX_BOARD * MAX_BOARD]bool) Reach {
        const cells = MAX_BOARD * MAX_BOARD;
        var seen = std.mem.zeroes([cells]bool);
        var queue = std.mem.zeroes([cells]u16);
        var depth = std.mem.zeroes([cells]u16);
        var out = Reach{ .area = 0, .food_dist = Reach.none, .tail_dist = Reach.none };
        const w: u16 = @intCast(self.size);
        var qh: usize = 0;
        var qt: usize = 0;
        const first = from[0] * self.size + from[1];
        const food_i = self.food[0] * self.size + self.food[1];
        const tail_i = self.tailCell()[0] * self.size + self.tailCell()[1];
        // The head's own cell is marked occupied by the caller and is still the
        // start of the fill: without that an eating move would measure zero room
        // and be rejected by the very tier that is supposed to prefer it.
        seen[first] = true;
        queue[qt] = @intCast(first);
        qt += 1;
        while (qh < qt) : (qh += 1) {
            const at = queue[qh];
            const dep = depth[at];
            const r: usize = at / w;
            const c: usize = at % w;
            out.area += 1;
            if (at == food_i and out.food_dist == Reach.none) out.food_dist = dep;
            if (at == tail_i and out.tail_dist == Reach.none) out.tail_dist = dep;
            for (DIRS) |d| {
                const dl = d.delta();
                const nr: isize = @as(isize, @intCast(r)) + dl[0];
                const nc: isize = @as(isize, @intCast(c)) + dl[1];
                if (nr < 0 or nc < 0 or nr >= self.size or nc >= self.size) continue;
                const i = @as(usize, @intCast(nr)) * self.size + @as(usize, @intCast(nc));
                if (seen[i] or blocked[i]) continue;
                seen[i] = true;
                depth[i] = dep + 1;
                queue[qt] = @intCast(i);
                qt += 1;
            }
        }
        return out;
    }

    /// The first steps of a shortest walk to the food that takes the body apart
    /// as it goes. A body cell is not a wall forever: the segment at index `j`
    /// vacates `len - j` steps later, so a path may cross it once it is gone.
    /// Spatial flood fill cannot see that, and a snake that cannot see it stops
    /// eating the moment its own body rings the food -- which is where the
    /// purely spatial tiers plateau at twenty-odd.
    fn unwindChase(self: *const Snake, pick: *[4]bool) bool {
        const cells = MAX_BOARD * MAX_BOARD;
        var free_at = std.mem.zeroes([cells]u16);
        var depth = std.mem.zeroes([cells]u16);
        var parent = std.mem.zeroes([cells]u16);
        var queue = std.mem.zeroes([cells]u16);
        const n = self.size * self.size;
        const w: u16 = @intCast(self.size);
        const len: u16 = @intCast(self.body.items.len);

        for (0..n) |i| free_at[i] = 0xFFFF; // nothing free until the body is placed
        for (0..n) |i| {
            if (!self.occupied[i]) free_at[i] = 0;
        }
        for (self.body.items, 0..) |c, j| {
            const at = @as(u16, @intCast(c[0])) * w + @as(u16, @intCast(c[1]));
            const vacates: u16 = len - @as(u16, @intCast(j));
            if (vacates < free_at[at]) free_at[at] = vacates;
        }

        const h = self.head();
        const start = h[0] * self.size + h[1];
        const goal = self.food[0] * self.size + self.food[1];
        free_at[start] = 0;
        var seen = std.mem.zeroes([cells]bool);
        seen[start] = true;
        var qh: usize = 0;
        var qt: usize = 0;
        queue[qt] = @intCast(start);
        qt += 1;
        var found: usize = Reach.none;
        while (qh < qt) : (qh += 1) {
            const at = queue[qh];
            const t = depth[at];
            if (at == goal) {
                found = t;
                break;
            }
            const r: usize = at / w;
            const c: usize = at % w;
            for (DIRS) |d| {
                const dl = d.delta();
                const nr: isize = @as(isize, @intCast(r)) + dl[0];
                const nc: isize = @as(isize, @intCast(c)) + dl[1];
                if (nr < 0 or nc < 0 or nr >= self.size or nc >= self.size) continue;
                const i = @as(usize, @intCast(nr)) * self.size + @as(usize, @intCast(nc));
                if (seen[i] or t + 1 < free_at[i]) continue;
                seen[i] = true;
                depth[i] = t + 1;
                parent[i] = at;
                queue[qt] = @intCast(i);
                qt += 1;
            }
        }
        if (found == Reach.none or found == 0) return false;

        // Walk the path back to the cell whose parent is the head: that is the
        // first step, and the tier reports every direction that starts one.
        var first: u16 = @intCast(goal);
        while (parent[first] != start and first != start) {
            first = parent[first];
        }
        if (parent[first] != start) return false;
        const fr = first / w;
        const fc = first % w;
        var any = false;
        for (DIRS, 0..) |d, i| {
            const dl = d.delta();
            const nr: usize = @intCast(@as(isize, @intCast(h[0])) + dl[0]);
            const nc: usize = @intCast(@as(isize, @intCast(h[1])) + dl[1]);
            pick[i] = nr == fr and nc == fc;
            if (pick[i]) any = true;
        }
        return any;
    }

    /// Which of the four directions the chooser is still asked about. The tiers
    /// are tried in order and the first non-empty one wins; everything inside
    /// that tier stays rankable, so the model breaks real ties rather than
    /// grading moves the board already ruled out. The objective the model does
    /// not hold -- stay alive long enough to matter -- is the board's job.
    ///
    /// 1 chase:   on a shortest path to the food that dodges the body, and it
    ///            still leaves the snake room as big as itself
    /// 2 coil:    the food is walled off; close in on the tail, the one cell of
    ///            the body that is guaranteed to move out of the way
    /// 3 open:    keep the largest region the snake can still see
    /// 4 survive: the same, without the reachability requirement
    /// 5 nothing is left but a legal question, so every direction except the
    ///   reverse stays offered
    pub fn rankedMoves(self: *const Snake) [4]bool {
        var tail_ok = std.mem.zeroes([4]bool);
        var both_ok = std.mem.zeroes([4]bool);
        var alive = std.mem.zeroes([4]bool);
        var roomy = std.mem.zeroes([4]bool);
        var area = std.mem.zeroes([4]usize);
        var fdist = std.mem.zeroes([4]usize);
        var tdist = std.mem.zeroes([4]usize);
        if (self.size > MAX_BOARD) { // too big for the stack buffers: variant 7
            var fall: [4]bool = undefined;
            for (DIRS, 0..) |d, i| fall[i] = self.moveIsBest(d);
            return fall;
        }

        var blocked = std.mem.zeroes([MAX_BOARD * MAX_BOARD]bool);
        for (0..self.size * self.size) |i| blocked[i] = self.occupied[i];
        const len = self.body.items.len;
        const tail = self.tailCell();
        const ti = tail[0] * self.size + tail[1];

        for (DIRS, 0..) |d, i| {
            const m = self.moveOf(d);
            // A 180-degree turn is not a move the snake can make, so it is not
            // offered either -- variant 8 cannot add to the illegal count.
            if (m.fatal or d == self.dir.opposite()) continue;
            const nb = m.cell orelse continue;
            const ni = nb[0] * self.size + nb[1];
            // The tail is free for the fill even when this move eats: eating
            // grows the snake, but the tail still vacates on the next step, and a
            // test that reads "cannot reach my own tail" for the move that
            // reaches the food would reject every meal.
            blocked[ti] = false;
            blocked[ni] = true;
            const r = self.reachable(nb, &blocked);
            blocked[ni] = false;
            blocked[ti] = true;
            tail_ok[i] = r.tail_dist != Reach.none;
            both_ok[i] = tail_ok[i] and r.food_dist != Reach.none;
            alive[i] = true;
            area[i] = r.area;
            fdist[i] = r.food_dist;
            tdist[i] = r.tail_dist;
            roomy[i] = r.area >= len;
        }

        var pick = std.mem.zeroes([4]bool);
        var best: usize = std.math.maxInt(usize);
        var wide: usize = 0;

        for (0..4) |i| {
            if (both_ok[i] and roomy[i] and fdist[i] < best) best = fdist[i];
        }
        for (0..4) |i| {
            if (both_ok[i] and roomy[i] and fdist[i] == best) pick[i] = true;
        }
        if (anyOf(pick)) return pick; // tier 1: chase

        // The unwind path is a route, not a safety argument: it says the food
        // can be reached once the body unties, not that the first step leaves the
        // snake room. So it only counts where the spatial tiers already vouch
        // for the move, and the model still ranks whatever survives that AND.
        pick = std.mem.zeroes([4]bool);
        var route = std.mem.zeroes([4]bool);
        if (self.unwindChase(&route)) {
            var any = false;
            for (0..4) |i| {
                pick[i] = route[i] and tail_ok[i] and roomy[i];
                if (pick[i]) any = true;
            }
            if (any) return pick; // tier 2: walk through the untangling body
        }

        pick = std.mem.zeroes([4]bool);
        best = std.math.maxInt(usize);
        for (0..4) |i| {
            if (tail_ok[i] and roomy[i] and tdist[i] < best) best = tdist[i];
        }
        for (0..4) |i| {
            if (tail_ok[i] and roomy[i] and tdist[i] == best) pick[i] = true;
        }
        if (anyOf(pick)) return pick; // tier 2: coil toward the tail


        pick = std.mem.zeroes([4]bool);
        wide = 0;
        for (0..4) |i| {
            if (tail_ok[i] and area[i] > wide) wide = area[i];
        }
        for (0..4) |i| {
            if (tail_ok[i] and area[i] == wide) pick[i] = true;
        }
        if (anyOf(pick)) return pick; // tier 3: keep the region open

        pick = std.mem.zeroes([4]bool);
        wide = 0;
        for (0..4) |i| {
            if (alive[i] and area[i] > wide) wide = area[i];
        }
        for (0..4) |i| {
            if (alive[i] and area[i] == wide) pick[i] = true;
        }
        if (anyOf(pick)) return pick; // tier 4: survive one more step

        for (DIRS, 0..) |d, i| {
            pick[i] = d != self.dir.opposite();
        }
        return pick; // tier 5: every move loses; keep the question legal
    }

    /// A Hamiltonian cycle over the whole board: column 0 is the way back and
    /// everything to its right serpents. It exists whenever the board has an even
    /// number of rows, which covers every size the CLI and the UI offer except
    /// the odd ones, where variant 9 degrades to variant 8.
    fn cycleIndex(size: usize, r: usize, c: usize) usize {
        if (c == 0) return if (r == 0) 0 else size * (size - 1) + (size - r);
        const row_start = r * (size - 1) + 1;
        return if (r % 2 == 0) row_start + (c - 1) else row_start + (size - 2) - (c - 1);
    }

    /// Which way round the cycle the body runs: `forward` when each segment sits
    /// one step *ahead* of the one behind it in cycle order, `back` when it sits
    /// one step behind.
    ///
    /// This has to be read off the body and cannot be assumed. The snake starts
    /// as a three-cell horizontal line, and whether that line points with the
    /// cycle or against it is decided by the parity of the middle row -- on a
    /// 10x10 board it points against it, on 8x8 with it. Guessing `forward`
    /// turns every legal move into an apparent overtaking manoeuvre, which is
    /// enough to drop variant 9 onto its fallbacks for whole games.
    const CycleSense = enum { forward, back };

    fn cycleSense(self: *const Snake) CycleSense {
        const items = self.body.items;
        if (items.len < 2) return .forward;
        const n = self.size * self.size;
        var vote: i32 = 0;
        for (items[0 .. items.len - 1], items[1..]) |a, b| {
            const ia = cycleIndex(self.size, a[0], a[1]);
            const ib = cycleIndex(self.size, b[0], b[1]);
            // In forward motion the segment ahead was reached from the one
            // behind it, so it sits a short distance further along the cycle;
            // downstream reads it as "how far to travel to get from a to b".
            const reach = (ia + n - ib) % n;
            if (reach == 0) continue;
            vote += if (reach * 2 < n) 1 else -1;
        }
        return if (vote < 0) .back else .forward;
    }

    /// How many steps it takes to get from cycle position `from` to `to` going
    /// the way the snake is travelling. Everything this tier promises is
    /// measured this way round: "the head may not pass the tail" only means
    /// something relative to the direction the visits actually advance in.
    fn cycleAhead(self: *const Snake, sense: CycleSense, from: usize, to: usize) usize {
        const n = self.size * self.size;
        return switch (sense) {
            .forward => (to + n - from) % n,
            .back => (from + n - to) % n,
        };
    }

    /// Variant 9: the board's order is the cycle, and the snake may leave it only
    /// where leaving cannot break the promise the cycle makes. The step that
    /// follows the cycle is always offered, so the set is never empty and the
    /// snake cannot trap itself -- that is what fills a board. A shortcut counts
    /// only if it stays behind the tail in cycle order, jumps over nothing but
    /// free cells, and shortens the trip to the food. Everything the snake may
    /// legally do next is offered to the chooser at once, so the model decides
    /// which way round the cycle to go rather than whether to survive.
    pub fn cycleMoves(self: *const Snake) [4]bool {
        var pick = std.mem.zeroes([4]bool);
        if (self.size % 2 != 0 or self.size > MAX_BOARD) return self.rankedMoves();
        const size = self.size;
        const n = size * size;
        const h = self.head();
        const t = self.tailCell();
        const sense = self.cycleSense();
        const hi = cycleIndex(size, h[0], h[1]);
        const fi = cycleIndex(size, self.food[0], self.food[1]);
        const to_tail = self.cycleAhead(sense, hi, cycleIndex(size, t[0], t[1]));
        const to_food = self.cycleAhead(sense, hi, fi);
        const free = n - self.body.items.len;

        for (DIRS, 0..) |d, i| {
            if (d == self.dir.opposite()) continue;
            const nb = self.neighbour(h, d) orelse continue;
            // The tail is the one body cell this very move vacates, so it is
            // enterable, and counting it as fatal would drop the last cycle step
            // before the tail out of the set exactly when the board is filling
            // up -- the one moment the set must not be empty.
            const into_tail = (nb[0] == t[0] and nb[1] == t[1]);
            if (self.occupied[self.idx(nb[0], nb[1])] and !into_tail) continue;
            const advance = self.cycleAhead(sense, hi, cycleIndex(size, nb[0], nb[1]));
            if (advance == 0 or advance > to_tail) continue; // would pass the tail
            if (advance == 1) {
                pick[i] = true; // one step along the cycle: always offered
                continue;
            }
            if (advance <= free + 1 and self.cycleAhead(sense, cycleIndex(size, nb[0], nb[1]), fi) < to_food)
                pick[i] = true; // a shortcut that gets the food sooner
        }
        // While the visits march one way round the cycle the single step is
        // always there, so this is only reached when the body has lost that
        // order -- after a fallback once took it off the cycle. rankedMoves
        // keeps it alive and gets it back within reach of one.
        if (anyOf(pick)) return pick;
        return self.rankedMoves();
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
                const verdict = if (m.fatal)
                    "fatal"
                else if (m.eats)
                    "it eats the food"
                else
                    try std.fmt.allocPrint(alloc, "safe, {d} steps from the food", .{ m.food_dist });
                try w.print("Moving {s} enters {s}: {s}\n", .{ d.label(), where, verdict });
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
                5, 6, 7, 8, 9 => if (m.fatal)
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

/// The step cap `--max-steps` falls back to when nothing was asked for.
///
/// Filling the board takes thousands of moves, not hundreds: walking the cycle
/// round costs up to one lap (`size^2`) per meal and there are nearly that many
/// meals, so the bound for a chooser that never takes a shortcut is `size^4/2`
/// moves. The flat 400 this used to be ended every tiered variant with the board
/// still almost empty, which read as "it cannot finish" rather than "it was not
/// given the moves". Measured worst cases for the cycle chooser, which takes its
/// shortcuts and so needs roughly `4 * size^3`: 6 -> 292, 8 -> 1046, 10 -> 2629,
/// 12 -> 4953, 16 -> 16056, 24 -> 88765 steps.
pub fn defaultMaxSteps(size: usize) usize {
    return size * size * size * size / 2;
}

pub const Stats = struct {
    games: usize = 0,
    score: usize = 0,
    best: usize = 0,
    wall: usize = 0,
    suicide: usize = 0,
    won: usize = 0,
    cap: usize = 0,
    illegal: usize = 0,
    moves: usize = 0,

    fn note(self: *Stats, g: *const Snake) void {
        self.games += 1;
        self.score += g.score;
        self.best = @max(self.best, g.score);
        if (g.alive) self.cap += 1 else if (g.won) self.won += 1 else if (g.hit_wall) self.wall += 1 else self.suicide += 1;
        self.illegal += g.illegal;
        self.moves += g.steps;
    }
};

pub fn sleepMs(io: std.Io, ms: i64) void {
    if (ms <= 0) return;
    const d = std.Io.Clock.Duration{ .raw = std.Io.Duration.fromMilliseconds(ms), .clock = .awake };
    d.sleep(io) catch {};
}

fn outcome(g: *const Snake) []const u8 {
    return if (g.won) "board full, won" else if (g.alive) "alive at cap" else if (g.hit_wall) "hit wall" else "hit itself";
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
                gi + 1, g.score, g.steps, outcome(&g), g.illegal,
            });
            try w.flush();
        }
    }

    if (cfg.games == 1) {
        try render(w, &g, "", ansi);
        // The last frame shows where the snake ended, not how long it took to
        // get there, and "finished in 61 moves" vs "finished in 1400" is most of
        // what distinguishes one chooser from another once both win.
        try w.print("\n{s} after {d} steps\n", .{ outcome(&g), g.steps });
    }

    const ngames: f64 = @floatFromInt(stats.games);
    try w.print("\n--- {s} ---\n", .{name});
    try w.print("score: total {d}  best {d}  mean {d:.2}\n", .{ stats.score, stats.best, @as(f64, @floatFromInt(stats.score)) / ngames });
    try w.print("outcome: won {d}  wall {d}  self {d}  capped {d}   illegal 180-degree moves: {d}\n", .{ stats.won, stats.wall, stats.suicide, stats.cap, stats.illegal });
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

/// The same candidate tiers with no model in the loop: first move the board
/// still offers, in `up, down, left, right` order. Runs in microseconds, so it
/// is what the tiers are tuned against, and it is the honest denominator for
/// whatever the model adds on top.
fn tierFn(ctx: *anyopaque, io: std.Io, g: *Snake, status: *std.Io.Writer) anyerror!Dir {
    _ = ctx;
    _ = io;
    const allowed = g.rankedMoves();
    for (DIRS, 0..) |d, i| {
        if (allowed[i]) {
            try status.print("tiers: chose {s}\n", .{d.label()});
            return d;
        }
    }
    return g.dir;
}

var tier_ctx = GreedyCtx{};

fn cycleFn(ctx: *anyopaque, io: std.Io, g: *Snake, status: *std.Io.Writer) anyerror!Dir {
    _ = ctx;
    _ = io;
    const allowed = g.cycleMoves();
    for (DIRS, 0..) |d, i| {
        if (allowed[i]) {
            try status.print("cycle: chose {s}\n", .{d.label()});
            return d;
        }
    }
    return g.dir;
}

var cycle_ctx = GreedyCtx{};

pub fn cycleChooser() Chooser {
    return .{ .ctx = &cycle_ctx, .moveFn = cycleFn };
}

pub fn tierChooser() Chooser {
    return .{ .ctx = &tier_ctx, .moveFn = tierFn };
}


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
