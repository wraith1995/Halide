"""
Prove the diamond-tiled wavefront schedule is data-race-free, so that
allow_race_conditions() is a TRUTHFUL assertion (not masking a real race).

Schedule (tile size T): iteration coord (bv, nyo, nyi, nxi).
  block indices:  rxo = bv - nyo,  ryo = nyo
  data written:   X = T*rxo + nxi + 1,  Y = T*ryo + nyi + 1
Loop nest:  for bv (SERIAL) :  parallel nyo :  for nyi :  for nxi
Body (recurrence):  f(X,Y) = f(X-1,Y) + f(X,Y-1) + in(X,Y)

Race-free  <=>  no dependence edge connects two iterations with the SAME bv but
DIFFERENT nyo (then, at a fixed bv, the parallel nyo tasks are independent), AND
the write map is injective (no two iterations write the same cell).

The writer of data cell (X,Y) has:
  rxo = (X-1)//T,  ryo = (Y-1)//T  =>  bv = rxo+ryo,  nyo = ryo
"""

import sympy as sp

T = 4
a, c = sp.symbols('a c', integer=True, nonnegative=True)  # block indices of the consumer

def check_edge(name, dbv, dnyo):
    # dbv = producer.bv - consumer.bv ; dnyo = producer.nyo - consumer.nyo  (constants)
    dbv, dnyo = sp.simplify(dbv), sp.simplify(dnyo)
    # Lemma: producer.bv == consumer.bv  =>  producer.nyo == consumer.nyo
    # i.e. never (dbv == 0 and dnyo != 0).
    bad = (dbv == 0) and (dnyo != 0)
    print(f"    {name}: Δbv={dbv}, Δnyo={dnyo}  ->  same-bv&different-nyo? {'YES (RACE!)' if bad else 'no'}")
    return not bad

print("=== SYMBOLIC PROOF (block indices a,c symbolic; within-block residues i,k enumerated) ===")
ok = True
for i in range(T):        # consumer within-block x residue: X-1 = T*a + i
    for k in range(T):    # consumer within-block y residue: Y-1 = T*c + k
        # consumer C writes (X,Y): block_x=a, block_y=c
        C_bv, C_nyo = a + c, c
        # x-neighbour read is column X-1, 0-based (X-1)-1 = T*a + (i-1):
        #   i>=1 -> block a ;  i==0 -> block a-1
        P1_blockx = a if i >= 1 else a - 1
        P1_bv, P1_nyo = P1_blockx + c, c
        # y-neighbour read is row Y-1, 0-based (Y-1)-1 = T*c + (k-1):
        #   k>=1 -> block c ;  k==0 -> block c-1
        P2_blocky = c if k >= 1 else c - 1
        P2_bv, P2_nyo = a + P2_blocky, P2_blocky
        ok &= check_edge(f"i={i},k={k} x-neighbour", P1_bv - C_bv, P1_nyo - C_nyo)
        ok &= check_edge(f"i={i},k={k} y-neighbour", P2_bv - C_bv, P2_nyo - C_nyo)
print(f"Symbolic lemma holds for all residues: {ok}\n")

# ---------------------------------------------------------------------------
print("=== EXHAUSTIVE CROSS-CHECK (concrete grids) ===")
def exhaustive(W, H, T):
    import math
    nbx = math.ceil((W - 1) / T)
    nby = math.ceil((H - 1) / T)
    writer = {}          # data cell (X,Y) -> iteration (bv,nyo,nyi,nxi)
    pts = []
    for nyo in range(nby):
        for bv in range(nyo, nyo + nbx):          # rxo = bv-nyo in [0, nbx-1]
            rxo = bv - nyo
            for nyi in range(T):
                for nxi in range(T):
                    X = T * rxo + nxi + 1
                    Y = T * nyo + nyi + 1
                    if 1 <= X <= W - 1 and 1 <= Y <= H - 1:
                        it = (bv, nyo, nyi, nxi)
                        assert (X, Y) not in writer, "WAW: two iterations write same cell!"
                        writer[(X, Y)] = it
                        pts.append((it, X, Y))
    # dependence check: for each iteration, its two reads; if read cell has a
    # producer, it must not be same-bv/different-nyo.
    races = 0
    for (bv, nyo, nyi, nxi), X, Y in pts:
        for (rx, ry) in [(X - 1, Y), (X, Y - 1)]:
            p = writer.get((rx, ry))
            if p is not None and p[0] == bv and p[1] != nyo:
                races += 1
    inj = (len(writer) == len(pts))
    print(f"  W={W} H={H} T={T}: valid pts={len(pts)}, write-map injective={inj}, "
          f"same-bv/diff-nyo dependences={races}  -> {'RACE-FREE' if races==0 and inj else 'RACE!'}")
    return races == 0 and inj

allok = ok
for (W, H, T) in [(9, 9, 2), (16, 12, 4), (17, 13, 4), (33, 17, 8), (64, 48, 8)]:
    allok &= exhaustive(W, H, T)

print()
print("VERDICT:", "RACE-FREE  (allow_race_conditions is truthful)" if allok else "RACE FOUND")
