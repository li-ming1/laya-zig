//! A tiny blocking HTTP/1.1 server, just enough to drive the browser UI.
//!
//! One connection at a time and `Connection: close`, which is fine for a local
//! single-user tool: the page issues one request per move and waits for it.
//! The model call blocks for ~1 s inside the handler, so nothing else may run
//! concurrently anyway.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;

pub const Request = struct {
    method: []const u8,
    path: []const u8,
    query: []const u8,

    /// first value of `key` in the query string, if present
    pub fn param(self: Request, key: []const u8) ?[]const u8 {
        var it = std.mem.splitScalar(u8, self.query, '&');
        while (it.next()) |kv| {
            const eq = std.mem.indexOfScalar(u8, kv, '=') orelse continue;
            if (std.mem.eql(u8, kv[0..eq], key)) return kv[eq + 1 ..];
        }
        return null;
    }

    pub fn intParam(self: Request, key: []const u8, default: i64) i64 {
        const v = self.param(key) orelse return default;
        return std.fmt.parseInt(i64, v, 10) catch default;
    }
};

pub const Response = struct {
    status: u16 = 200,
    content_type: []const u8 = "application/json; charset=utf-8",
    /// empty means "use whatever the handler wrote into `out`"
    body: []const u8 = "",
};

pub const Handler = struct {
    ctx: *anyopaque,
    /// `out` is an allocating writer the handler may build its body in; the
    /// returned slice must point into it (or be static).
    handleFn: *const fn (ctx: *anyopaque, io: Io, req: Request, out: *Io.Writer) anyerror!Response,
};

pub const Options = struct {
    host: []const u8 = "127.0.0.1",
    port: u16 = 8080,
    log: ?*Io.Writer = null,
};

pub fn serve(gpa: Allocator, io: Io, opts: Options, handler: Handler) !void {
    const addr = try Io.net.IpAddress.parse(opts.host, opts.port);
    var server = try addr.listen(io, .{ .reuse_address = true });
    defer server.deinit(io);

    if (opts.log) |l| {
        try l.print("listening on http://{s}:{d}/  (Ctrl+C to stop)\n", .{ opts.host, opts.port });
        try l.flush();
    }

    while (true) {
        const stream = server.accept(io) catch |e| switch (e) {
            error.Canceled => return,
            else => continue,
        };
        defer stream.close(io);
        handleConn(gpa, io, stream, handler) catch {};
    }
}

fn handleConn(gpa: Allocator, io: Io, stream: Io.net.Stream, handler: Handler) !void {
    var rbuf: [4096]u8 = undefined;
    var r = stream.reader(io, &rbuf);

    // readVec does a single read, so we stop as soon as the headers arrive instead
    // of blocking until the buffer is full (readSliceShort would wait for the peer).
    var head: [8192]u8 = undefined;
    var n: usize = 0;
    var data: [1][]u8 = undefined;
    while (n < head.len) {
        data[0] = head[n..];
        const got = r.interface.readVec(&data) catch break;
        if (got == 0) break;
        n += got;
        if (std.mem.indexOf(u8, head[0..n], "\r\n\r\n") != null) break;
    }
    if (n == 0) return;

    var lines = std.mem.splitSequence(u8, head[0..n], "\r\n");
    const first = lines.next() orelse return;
    var parts = std.mem.tokenizeScalar(u8, first, ' ');
    const method = parts.next() orelse return;
    const target = parts.next() orelse return;

    var path = target;
    var query: []const u8 = "";
    if (std.mem.indexOfScalar(u8, target, '?')) |q| {
        path = target[0..q];
        query = target[q + 1 ..];
    }
    const req = Request{ .method = method, .path = path, .query = query };

    var out: Io.Writer.Allocating = .init(gpa);
    defer out.deinit();

    var res = handler.handleFn(handler.ctx, io, req, &out.writer) catch Response{
        .status = 500,
        .body = "{\"error\":\"handler failed\"}",
    };
    if (res.body.len == 0 and out.writer.buffered().len > 0) res.body = out.writer.buffered();

    var wbuf: [64 * 1024]u8 = undefined;
    var w = stream.writer(io, &wbuf);
    const wi = &w.interface;
    try wi.print("HTTP/1.1 {d} {s}\r\nContent-Type: {s}\r\nContent-Length: {d}\r\nConnection: close\r\nCache-Control: no-store\r\n\r\n", .{
        res.status,
        if (res.status == 200) "OK" else if (res.status == 404) "Not Found" else "Error",
        res.content_type,
        res.body.len,
    });
    try wi.writeAll(res.body);
    try wi.flush();
}

/// JSON string escaper (the prompt text is full of newlines and quotes)
pub fn jsonString(w: *Io.Writer, s: []const u8) !void {
    try w.writeByte('"');
    for (s) |c| {
        switch (c) {
            '"' => try w.writeAll("\\\""),
            '\\' => try w.writeAll("\\\\"),
            '\n' => try w.writeAll("\\n"),
            '\r' => try w.writeAll("\\r"),
            '\t' => try w.writeAll("\\t"),
            0...8, 11, 12, 14...31 => try w.print("\\u{x:0>4}", .{c}),
            else => try w.writeByte(c),
        }
    }
    try w.writeByte('"');
}
