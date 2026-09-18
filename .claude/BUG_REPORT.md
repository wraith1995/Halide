# `solve_for_outer_interval` mishandles `min`/`max` of booleans → `trim_no_ops` over-trims a serial loop → miscompile

## Summary

`solve_for_outer_interval` (src/Solve.cpp, `SolveForInterval`) does not treat
`min`/`max` of boolean operands as logical `And`/`Or`. But `and_condition_over_domain`
and the simplifier represent boolean AND/OR of comparisons as `min`/`max`. As a
result, `solve_for_outer_interval` returns a wrong (too-small) region for a
condition like `!min(a, b)`. `trim_no_ops` relies on this to decide how far to
trim a loop, so it can trim a serial loop down to a single iteration and drop
real work — a silent miscompile.

## Minimal root-cause reproducer (no schedules, no parallelism)

```cpp
#include "Halide.h"
#include "Solve.h"   // Internal
using namespace Halide; using namespace Halide::Internal;
int main() {
    Expr bv = Variable::make(Int(32), "bv");
    Expr m  = min(bv < 0, bv < 2);                 // == (bv < 0)
    Interval i = solve_for_outer_interval(!m, "bv");
    // EXPECT [0, +inf)  (since !m == (bv >= 0));  ACTUAL [2, +inf)
    std::cout << i.min << " .. " << i.max << "\n";
}
```
`solve_for_outer_interval(!(bv<0), "bv")` and `solve_for_outer_interval(bv>=0, "bv")`
both correctly give `[0, +inf)`; only the `min`-encoded form is wrong. Also
`simplify(min(bv<0, bv<2))` does not reduce to `bv<0`.

Cause: `SolveForInterval` (src/Solve.cpp:867) implements `visit(And)`, `visit(Or)`,
`visit(Not)` but not `visit(Min)`/`visit(Max)`, so `!min(P,Q)` is not turned into
`max(!P,!Q)` (De Morgan); it behaves like `!P && !Q` (=> `bv>=0 && bv>=2` = `bv>=2`)
instead of `!P || !Q` (= `bv>=0`).

## End-to-end manifestation (silent wrong result)

A diamond/wavefront-tiled 2D recurrence, expressed with public API only. After
loop partitioning (which is correct) the serial block-anti-diagonal loop `bv` has
a steady part `for (bv, 0, 2)` guarded by the parallelogram `bv-3<=nyo && nyo<=bv`
over an inner (parallel) `nyo`. `trim_no_ops` relaxes the no-op condition over
`nyo` to `min(bv<0, bv<2)`, then `solve_for_outer_interval(!min(...), bv)` returns
`[2, +inf)`, so `for (bv,0,2)` is trimmed to `bv==2`; `bv in {0,1}` are never
computed.

```cpp
#include "Halide.h"
#include <cstdio>
using namespace Halide;
static Buffer<int> run(const Buffer<int>&in,int W,int H,int T,bool use_likely){
    Func f("f"); Var x("x"),y("y"); f(x,y)=in(x,y);
    int nbx=(W-1+T-1)/T, nby=(H-1+T-1)/T;
    RDom d({{0,T},{0,T},{0,nby},{0,nbx+nby-1}},"d");
    RVar nxi=d[0],nyi=d[1],nyo=d[2],bv=d[3]; Expr rxo=bv-nyo;
    Expr tx=T*rxo+nxi, ty=T*nyo+nyi;
    if(use_likely){tx=likely(tx);ty=likely(ty);}
    d.where(0<=rxo && rxo<=nbx-1 && tx<=W-2 && ty<=H-2);
    Expr X=unsafe_promise_clamped(T*(bv-nyo)+nxi+1,1,W-1);
    Expr Y=unsafe_promise_clamped(T*nyo+nyi+1,1,H-1);
    f(X,Y)=f(X-1,Y)+f(X,Y-1)+in(X,Y);
    f.update().allow_race_conditions().parallel(nyo);   // race-free (proof below)
    return f.realize({W,H});
}
int main(){
    const int W=16,H=12,T=4; Buffer<int> in(W,H);
    for(int y=0;y<H;y++)for(int x=0;x<W;x++)in(x,y)=(x*3+y*5+1)%11;
    Buffer<int> ref(W,H);
    for(int y=0;y<H;y++)for(int x=0;x<W;x++)
        ref(x,y)=(x==0||y==0)?in(x,y):ref(x-1,y)+ref(x,y-1)+in(x,y);
    auto c=[&](Buffer<int>o,const char*l){int m=0;for(int y=0;y<H;y++)for(int x=0;x<W;x++)m+=o(x,y)!=ref(x,y);printf("  %-16s: %s (%d)\n",l,m?"WRONG":"correct",m);};
    c(run(in,W,H,T,false),"plain");     // correct
    c(run(in,W,H,T,true), "likely");    // WRONG (165) -- the likely tail guard is
                                        // what routes trim_no_ops into the buggy path
}
```
Expected: both `correct (0)`. Actual: `likely` → `WRONG (165)`.
`likely()` is documented as a value-identity (`remove_likelies`), so the two must agree.

## Not a race

The parallelized `nyo` (block position on a block-anti-diagonal, `bv` serial) is
data-race-free: for fixed `bv`, distinct `nyo` write disjoint cells and every
recurrence dependence that crosses `nyo` also crosses `bv` (carried by the serial
loop); the write map is a bijection. Verified symbolically (all within-block
residues) and exhaustively (e.g. W=16,H=12,T=4: 165 valid points, 0 same-`bv`/
different-`nyo` dependences — matching the 165 wrong cells). So
`allow_race_conditions()` is truthful and the wrong output is not a race.

## Where it goes wrong (pass by pass)
- `partition_loops`: correct (serial `bv` split into `[0,2]` + `[3,5]`, guards retained).
- `trim_no_ops` (Lower.cpp:408): `IsNoOp::visit(For)` (TrimNoOps.cpp:120-134) relaxes
  the body's no-op condition over the inner loop's full range via
  `and_condition_over_domain`, yielding `min(bv<0, bv<2)`; then
  `solve_for_outer_interval(!is_no_op, bv)` (TrimNoOps.cpp:386) returns `[2,+inf)`
  due to the `Solve.cpp` bug above, trimming `for (bv,0,2)` to `bv==2`.

## Fix candidates
- In `SolveForInterval` (src/Solve.cpp), handle `Min`/`Max` of boolean operands as
  `And`/`Or` (so `Not` distributes correctly), OR
- have the simplifier canonicalize `min`/`max` of `bool` to `&&`/`||` (then
  `min(bv<0,bv<2)` folds to `bv<0`), OR
- have `and_condition_over_domain` emit `&&`/`||` rather than `min`/`max` for bools.

## Environment
- Halide upstream `main` @ `d33fd254f`; reproducers use only public API (the local
  tree adds unrelated code that does not touch Solve.cpp/TrimNoOps.cpp/PartitionLoops.cpp).
- LLVM 21.1.8; macOS arm64, Apple clang 21; host target `arm-64-osx`.

## Workaround (in user code)
Avoid feeding `trim_no_ops` a no-op condition that becomes `min`/`max` of bools:
e.g. don't use a `likely` tail guard on a predicate that couples a serial loop
with a parallel inner loop (use a plain guard, or pad to tile-multiple extents so
no tail guard is generated).
