import halide as hl


def test_reindex_wavefront():
    n = 16
    x, y, u, v = hl.Var("x"), hl.Var("y"), hl.Var("u"), hl.Var("v")

    def wavefront(name):
        f = hl.Func(name)
        f[x, y] = x + y
        r = hl.RDom([(1, n - 1), (1, n - 1)])
        f[r.x, r.y] = (f[r.x - 1, r.y] + f[r.x, r.y - 1]) % 1009
        return f, r

    expected, _ = wavefront("f_serial")
    expected = expected.realize([n, n])

    f, r = wavefront("f_skewed")
    # Skew into wavefronts: v numbers the wavefront, u the position on it.
    res = f.update().reindex([u, v], [r.x, r.x + r.y], [u, v - u])
    assert list(res.parallel_safe) == [True, False]
    f.update().parallel(res.vars[0])
    out = f.realize([n, n])

    for iy in range(n):
        for ix in range(n):
            assert out[ix, iy] == expected[ix, iy]


def test_reindex_storage():
    n = 8
    x, u = hl.Var("x"), hl.Var("u")
    f = hl.Func("f")
    f[x] = x * 3
    g = hl.Func("g")
    g[x] = f[x] + 1
    # Store f reflected.
    stored = f.reindex_storage([u], [n - 1 - x], [n - 1 - u])
    stored.compute_root()
    out = g.realize([n])
    for i in range(n):
        assert out[i] == i * 3 + 1


if __name__ == "__main__":
    test_reindex_wavefront()
    test_reindex_storage()
