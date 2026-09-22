# ===================== RESUME / CURRENT STATE (2026-09-22) =====================
# Two parallel tracks. Read this block first after a compaction.
#
# TRACK A — Upstream Halide bug fix (DONE except user-side filing):
#   Bug: SolveForInterval (src/Solve.cpp) lacked visit(Min)/visit(Max); boolean
#   min/max (which and_condition_over_domain + simplifier emit as &&/||) weren't
#   De-Morgan'd, so solve_for_outer_interval(!min(x<0,x<2)) gave [2,+inf) not
#   [0,+inf). Surfaced as a silent miscompile via trim_no_ops (trims a serial loop
#   to 1 iteration). Fix = add visit(Min)->And, visit(Max)->Or (mirror visit(Select)).
#   - Tested on PRISTINE main worktree /Users/teoc/projects/halides-mainfix
#     (branch fix/solve-min-max-bool, commit 43ff34a69, build build/fix):
#     bug present on stock main -> fixed; ctest correctness 471/471, error+warning
#     168/168, fuzz_solve clean, clang-format 0-diff.
#   - Commit 43ff34a69 has Co-authored-by (Claude Opus 4.8); Claude-Session trailer
#     STRIPPED for upstream.
#   - PUSHED to fork wraith1995/Halide (isFork of halide/Halide), branch
#     fix/solve-min-max-bool. Nothing posted to halide/Halide yet.
#   - USER FILES: issue -> https://github.com/halide/Halide/issues/new?template=bug-report.yml
#     PR   -> https://github.com/halide/Halide/compare/main...wraith1995:Halide:fix/solve-min-max-bool?expand=1
#     (base MUST be halide/Halide:main; fork default is master.) Put issue # in pr "Fixes #".
#   - Texts (durable): .claude/halide-solve-bug/{issue.md, pr.md, BUG_REPORT.md is .claude/BUG_REPORT.md},
#     repros solve_probe.cpp (minimal), bug_repro.cpp (e2e), race_free_proof.py,
#     artifact halide_bug.html (published: https://claude.ai/artifact/KZawHihuoxbJCthyNjNudT).
#
# TRACK B — reindex/rfactor experiment (branch experiment/rfactor-two-phase, this
#   repo; origin=wraith1995/halide-experiments PR #1; upstream=halide/Halide):
#   12 commits: two-phase rfactor materialization -> Phase A default + dead-code del
#   -> isolate helpers (partition_reduction_splits, select_subdomain, build_reduced_dims,
#   set_loop_structure/make_sub_schedule, promote_preserved_dims) -> pure core
#   reindex_reduction_domain -> Stage::reindex (affine iteration-space bijection) with
#   ReindexResult{vars,parallel_safe} + dependence check + loop_order -> wavefront_skew
#   + diamond_tile tests -> the SAME Solve.cpp fix (commit 55969e95f) which made the
#   diamond parallel work for arbitrary (non-tile-multiple) sizes.
#   NOTE: the Track A fix makes the earlier "strip likely / tile-multiple restriction"
#   reindex workaround MOOT — diamond parallel is correct for any size now.
#
# HOW TO CONTINUE (options, user picks):
#   1. (after user files) respond to Halide maintainer review on the PR.
#   2. reindex redesign: "valid Halide by construction" -> rfactor-shaped parallel
#      axis (pure Var via factoring, not race-asserted rvar) and/or exact-partition
#      geometry; the parallel/serial framing was wrong, aim = emit normally-composable
#      Halide. (See design discussion in this file / history.)
#   3. storage bijection (deferred) = layout analogue of reindex; the real tool for
#      Toeplitz reuse (reduction-domain reindex can't express it: pure-var-dependent
#      rvar bounds don't lower).
#   4. build/test the reindex branch broadly; consider upstreaming reindex separately.
# Build cmd (either tree): ninja -C build/<exp|fix> -j6 <target>; LLVM @ /opt/homebrew/opt/llvm@21; WASM off.
# ==============================================================================

# PLAN — Two-phase rfactor experiment (branch: experiment/rfactor-two-phase)

## Repo / PR
- Private repo: github.com/wraith1995/halide-experiments (origin). upstream=halide/Halide.
- Internal PR #1: experiment/rfactor-two-phase -> main (isCrossRepository=false).
- Commit 1 (9a8ac02f4): two-phase materialization, env-gated.

## DONE (commit 2): Phase A default + dead code deleted + first isolation
- Phase A now unconditional (env gate removed); materialize_reduction_domain(def, name()).
- rfactor_validate_args() is validation-only (returns void); dead split-partition removed.
  rfactor/hoist read var_splits = schedule().splits() (Phase A leaves only pure splits).
- Extracted shared reduction-schedule primitives (anon namespace in Func.cpp):
    * partition_reduction_splits(sched, stage_name) -> {var,rvar}; owns the
      "Var fused into RVar" user error (canonical msg preserved; verified exit 1).
    * select_subdomain(dims, rdom) = project_rdom(dims, rdom, {}) -- the split-free
      filter+rename used by Phase B; project_rdom keeps full split handling for Phase A.
    * materialize_reduction_domain shared by rfactor() + hoist_invariants().
- Func.h: rfactor_validate_args declaration updated to void.
- Tests: correctness_rfactor 0, correctness_hoist_invariants 0, error test exit 1
  ("...fused a Var into an RVar: y, r$z").

## DONE (commit 3): shared build_reduced_dims
- build_reduced_dims(sched, update_args, dim_vars, keep_rvars): keep pure dims +
  rvar dims in keep_rvars, drop the rest, re-insert missing pure Vars (histogram).
  Used by rfactor reducing stage (keep=preserved) and hoist write-back (keep={}).
- Tests still green (rfactor 0, hoist 0, error 1).

## Reusable reduction-schedule primitives now isolated (anon ns, Func.cpp)
  partition_reduction_splits | select_subdomain | build_reduced_dims |
  materialize_reduction_domain | (existing: project_rdom, RFactorProjection, add_let)

## DONE (commit 4): sub-schedule builder
- set_loop_structure(sched, dims, rvars, splits): stamp the loop structure onto a
  schedule in place, leaving parallel/vectorize/fusion/storage intact. Used by
  rfactor reducing stage and hoist write-back stage.
- make_sub_schedule(src, dims, rvars, splits) = src.get_copy() + set_loop_structure:
  build a derived stage's schedule inheriting the parent's loop-level scheduling.
  Used for the rfactor intermediate. (hoist intermediate stays a plain get_copy.)
- Tests still green (rfactor 0, hoist 0, error 1).

## DONE (commit 5): promote_preserved_dims
- promote_preserved_dims(source_dims, preserved_rvars, preserved_vars, pure_dims):
  the RVar->Var promotion for the rfactor intermediate -- replace each preserved
  rvar Dim with its pure Var Dim (from the intermediate's pure dims), then insert
  any preserved Var missing, just before outermost. Replaces the inline intm_dims
  loops. Tests still green (rfactor 0, hoist 0, error 1).

## Reusable reduction-schedule primitives now isolated (anon ns, Func.cpp)
  partition_reduction_splits | select_subdomain | build_reduced_dims |
  set_loop_structure | make_sub_schedule | promote_preserved_dims |
  materialize_reduction_domain
  (existing: project_rdom, RFactorProjection, add_let)

## DONE (commit 6): pure reduction-rewrite API
- ReductionStage {args,values,predicate,dims,rvars,splits} + read/write_reduction_stage.
- reindex_reduction_domain(in, inverse, new_rdom, new_dims, new_splits) -> ReductionStage:
  PURE core. inverse: old rvar name -> Expr(new rvars) (= phi^{-1}); wraps old rvars
  in bounds promises, projects args/values/predicate, strips domain back-pointers.
- helpers: reduction_bounds_promises, drop_rvar_domains.
- materialize_reduction_domain is now a thin client: project_rdom gives (flat_rdom,
  inverse) + dim rename, then reindex_reduction_domain does the rewrite.
- Tests green.

## DONE (commit 7): index-bijection / skew directive (Stage::reindex)
- Stage::reindex(new_vars(Var), forward, inverse): affine iteration-space bijection
  on a reduction domain, built on reindex_reduction_domain.
    * materializes reduction-domain splits first (composes with Phase A);
    * image bounding box via bounds_of_expr_in_scope over the old box;
    * predicate carves the parallelogram (original box constraints in new coords)
      so it's a bijection onto exactly the original points;
    * inverse rewrites accesses (old = phi^{-1}(new)); dims renamed positionally.
- new_vars are Var (not RVar) so forward/inverse Exprs don't need a bound domain
  (RVar->Expr requires defined bounds; caught that during bring-up).
- test/correctness/reindex_skew.cpp (registered): skew, reverse-shear, identity;
  all match the plain sum. Verified loop nest: v in [0,W-1+H-1], u in [0,W-1].
- SECOND client validates the pure API generalizes beyond split-materialization.

## DONE (commit 8): wavefront parallelism via skew
- test/correctness/wavefront_skew.cpp: 2D neighbour recurrence
  f(x,y)=f(x-1,y)+f(x,y-1)+in; reindex u=r.x, v=r.x+r.y makes anti-diagonals
  independent. Loop nest: `for v in [2,12]: parallel u in [1,7]:`. All 3 modes
  (unskewed / skewed serial / skewed parallel(u)) MATCH the serial reference.
- Confirms reindex works on a SELF-REFERENTIAL (recurrence) update, not just a
  commutative sum, and that the exposed dim is genuinely parallelizable
  (allow_race_conditions().parallel(RVar("u"))).
- API guidance discovered: reindex takes Var new_vars but produces RVar dims, so
  later parallel/reorder must reference them as RVar (name clash otherwise).
  Candidate refinement: have reindex return the new RVars, or accept RVar.

## DONE (commit 9): reindex returns RVars + dependence check
- Stage::reindex now returns ReindexResult {vector<RVar> vars; vector<bool> parallel_safe}.
- Dependence check (reindex_parallel_safe, anon ns): extracts uniform integer
  dependence vectors from the update's self-refs (LHS = rvar_i+const, self-ref =
  rvar_i+const; else conservative all-false), transforms by the forward Jacobian,
  and per new dim decides parallel-safety = carries no loop-carried dependence in
  the produced (outer->inner) loop order. Warns if the order is dep-illegal.
- helpers: affine_int_coeffs (integer-affine coeff extraction), collect_self_ref_args.
- wavefront_skew.cpp now parallelizes exactly the dims the check proves safe (u),
  not v; still MATCHes. Confirms the check is discriminating (parallelizing v
  would race). as_const_int returns std::optional in this tree (fixed).

## SPEEDUP FINDING (answer to "is there a speedup for parallelism in skewing?")
Benchmark 2048x2048 recurrence, 14 cores:
  unskewed serial   3.54 ms
  skewed serial    16.09 ms   (~4.5x slower: parallelogram predicate + bigger
                                bounding box + worse locality than row-major)
  skewed parallel  374.29 ms  (~100x SLOWER: a parallel region spawned per
                                anti-diagonal, ~4096 of them, tiny/varying width
                                -> spawn overhead dominates the 2-add payload)
=> Naive point-wise wavefront parallelism does NOT speed this up. The parallelism
   is real/correct but too fine-grained + skew hurts locality. A real win needs
   coarsening (wavefront/diamond TILING: skew then split/tile so each parallel
   task is a block) and a heavier per-point body. That composition (reindex +
   tile) is the natural follow-up.

## TOEPLITZ FINDING (attempted; a boundary of reduction-domain reindex)
Tried: out(x) += wt(r)*in(x+r); reindex r -> k = x+r (inverse r = k-x) so in(x+r)
= in(k) depends only on k (sliding-window/Toeplitz reuse).
- reindex BUILDS it fine; loop nest = `for x: for k:` with in(...) indexed by k.
- But it FAILS at codegen: "Symbol not found: x". The forward k=x+r couples the
  pure var x with rvar r, so the new reduction var k has bounds [x, x+R-1] --
  i.e. the reduction domain depends on a pure loop var, which Halide computes
  loop-invariantly and cannot lower.
=> Efficient Toeplitz reuse is fundamentally a STORAGE/producer-scheduling
   transform (line-buffer `in` via store_at/compute_at/fold), NOT a reduction-
   domain reindex. Guidance for the API: reduction-domain reindex is sound only
   when the new domain bounds are independent of pure vars (rvar-only bijections
   like the 2D skew). A pure/reduction-coupled map would need the pure vars'
   ranges supplied to build a constant predicated box -- but that yields the
   explicit O(N^2) Toeplitz matrix (no reuse win). So: defer to the storage-side
   bijection (which the user deferred) for real Toeplitz.

## DONE (commit 11): diamond (wavefront) tiling -> REAL SPEEDUP
- Added reindex loop_order param: sets the new rvar loop order directly, bypassing
  reorder()'s assoc/comm check on RVars (which refuses recurrence reorders). This
  lets reindex produce dependence-legal orders reorder() can't (e.g. diamond).
- Diamond tiling = split(r.x)/split(r.y) then reindex a 4D bijection skewing the
  BLOCK indices: bv = rxo+ryo (block anti-diagonal, serial), nyo = ryo (block
  position, PARALLEL), nxi/nyi within-block; loop_order {nxi,nyi,nyo,bv}.
  Refer to split leaves by name-matched Var placeholders (leaves are schedule-only).
- test/correctness/diamond_tile.cpp (registered): divisible interior extents, with
  parallel, all correct.
- SPEEDUP (compute-heavy recurrence, 2049x2049, 14 cores):
    serial 35.5 ms; diamond T=32 12.4 ms (2.86x); T=64 11.2 ms (3.18x); bit-exact.
  (Cheap 2-add recurrence: diamond fixes the 100x point-wise disaster -> ~par with
   serial, since it's memory-bandwidth-bound. Speedup needs a compute-bound body.)
- LIMITATION: non-divisible extents emit a `likely` split tail guard on the now-
  parallel block index; allow_race_conditions()+predicated-parallel-RVar lowering
  miscompiles it (serial reindexed schedule is still correct -> Halide lowering
  interaction, not a reindex bug). Diamond tiling assumes/pads tile-multiple sizes.

## DONE (commit 12): fix the Halide solve bug + enable non-divisible diamond
- Root cause (found by digging): NOT the partitioner. partition_loops is correct.
  trim_no_ops relaxes a loop's no-op condition over an inner loop -> min(bv<0,bv<2)
  (AND-of-bools as min); then solve_for_outer_interval(!min(...),bv) returned
  [2,+inf) instead of [0,+inf). SolveForInterval (src/Solve.cpp) had visit(And/Or/
  Not) but NO visit(Min)/visit(Max), so !min-of-bools wasn't De-Morgan'd.
- FIX (src/Solve.cpp): add visit(Min)=And, visit(Max)=Or for bool operands
  (mirrors the existing visit(Select)). ~14 lines.
- Verified: solve_probe now [0,+inf); bug_repro both plain+likely correct;
  diamond_tile extended with NON-divisible cases (16x12,17x13,23x19) -> Success.
  Speedup intact (~2.2x T=64, maxerr 0).
- No regressions: trim_no_ops, boundary_conditions, bad_likely, bounds,
  bounds_inference(+complex), bounds_of_split, align_bounds, reduction_schedule,
  argmax, histogram, vectorize_guard_with_if, multi_splits_with_diff_tail_strategies,
  + rfactor/hoist/reindex/wavefront/diamond (all pass); fuzz_solve 8s clean.
- Artifacts: .claude/BUG_REPORT.md (file-ready, min repro = solve_probe),
  .claude/halide-solve-bug/{solve_probe.cpp,bug_repro.cpp,race_free_proof.py}.

## Branch state for review: 12 commits. Halide solve bug FIXED; diamond parallel
## works for arbitrary sizes; all reindex/rfactor/hoist tests + regressions pass.

## UPSTREAM PREP (in progress): test on pristine main + issue + PR
- Worktree: /Users/teoc/projects/halides-mainfix, branch fix/solve-min-max-bool off
  upstream/main (d33fd254f). build dir build/fix (Debug, LLVM21, WASM off).
- CONFIRMED bug on STOCK main: solve_outer(!min(bv<0,bv<2)) = [2,+inf).
- Applied fix (src/Solve.cpp visit(Min)/visit(Max) bool->And/Or) + regression checks
  in test/correctness/solve.cpp. After fix: probe [0,+inf); correctness_solve Success.
- Full build (runjob hmain-all) -> then ctest correctness (+error+warning).
- Drafts: scratchpad/issue.md (minimal, solve repro), scratchpad/pr.md.
- Fix branch will be commit for the PR (fix + solve.cpp test only; NO reindex).
- Outward actions (push fix branch to fork / open issue+PR on halide/Halide) held
  for explicit user go-ahead.
Goal (user): "index bijection command" = given a stage + forward map (new = phi(old))
and its inverse (old = phi^{-1}(new)) on the iteration space, rewrite the loop nest to
iterate the IMAGE of phi and access via the inverse. Prototype affine SKEW on a
reduction domain via reindex_reduction_domain; test numerically (commutative sum over a
box, skewed -> same result). Then: skew for parallelism (wavefront), Toeplitz codegen,
and a storage analogue. Design decisions to settle: directive surface (Func-based
forward/inverse vs explicit affine exprs), image bounding-box + predicate computation,
pure-Func vs reduction iteration spaces.

## (superseded) original NEXT plan
Step A [make default + delete dead code]:
  - Remove env gate: maybe_materialize_reduction_domain always materializes.
  - Move the "Var fused into RVar" user_error into Phase A (needs stage-name ctx);
    Phase A currently BAILS expecting rfactor_validate_args to catch it.
  - Delete the split-partition loop from rfactor_validate_args (now validation-only;
    rvar_splits always empty after Phase A). rfactor/hoist read var_splits =
    schedule().splits() directly (Phase A leaves only pure splits there).
  - Phase B project_rdom calls now always pass empty splits (degenerate filter+rename).
    NOTE: project_rdom's split-replay is NOT dead -- Phase A uses it with real splits.
Step B [isolate reusable schedule ops]:
  - partition_reduction_splits(StageSchedule&) -> {var,rvar} (used by Phase A).
  - select_subdomain(dims, rdom) = project_rdom(dims, rdom, {}) for Phase B.
  - build_reduction_dims / writeback-dim builder shared by rfactor + hoist
    (drop rvar dims, keep preserved, histogram pure-dim insertion).
  - Verify correctness_rfactor + correctness_hoist_invariants + error test after each.


## Goal
Reimplement Stage::rfactor as two decoupled phases:
- Phase A `materialize_reduction_domain(Definition&)`: apply ALL rvar-affecting
  splits/fuses/renames to the update definition — expand the RDom to the flat
  product domain, substitute args/values/predicate onto the new leaf RVars
  (r.x -> 8*rxo+rxi), fold split-tail strategies into the predicate. Reuses
  project_rdom / apply_split. Pure-var splits are left in schedule().splits().
  Postcondition: rvar dims 1:1 with rvars(), NO rvar splits remain.
- Phase B (rfactor core): operate on the materialized (split-free) definition.
  Partition flat rvars into preserved vs intermediate by set membership only
  (no project_rdom / no split math). Promote preserved rvars->pure Vars, build
  intermediate Func, rewrite caller. Maps become pure renames.

## Decisions (from user)
- Phase A materializes the REDUCTION domain only; pure-var splits & reorder stay as var_splits.
- Validate by building Halide + running test/correctness/rfactor.cpp for parity.
- Reuse apply_split / compute_loop_bounds_after_split (don't reimplement split math).

## Dispatch / A-B
- Gate new path behind env var HL_RFACTOR_TWO_PHASE (Stage::rfactor dispatches).
  Run correctness_rfactor with and without it -> both must pass (parity).

## Build
- build dir: build/exp ; Debug ; LLVM 21 (/opt/homebrew/opt/llvm@21) ; WASM backend OFF
- configure: DONE. Targets: `Halide` (libHalide), `correctness_rfactor`,
  error tests e.g. `error_rfactor_after_var_and_rvar_fusion`.
- Baseline build: runjob name=halide-baseline (ninja -C build/exp -j6 correctness_rfactor).
  Incremental rebuilds after editing Func.cpp only recompile Func.cpp + relink libHalide.
- Run test: build/exp/test/correctness/correctness_rfactor

## STATUS: DONE — two-phase rfactor at parity with baseline.

## Final design (simpler than planned)
Only ONE new function needed. Phase B is the EXISTING rfactor body unchanged:
once Phase A materializes the rvar splits, the existing project_rdom() calls see
empty rvar_splits and degrade to a pure filter+rename. So:

  Stage::rfactor(): if HL_RFACTOR_TWO_PHASE set -> materialize_reduction_domain(definition);
                    then run the existing (unchanged) factoring body.

materialize_reduction_domain(Definition&) [Phase A], src/Func.cpp ~line 772:
 - partition splits into var/rvar (mirrors rfactor_validate_args);
   illegal Var-fused-into-RVar -> RETURN early (no mutation) so the existing
   rfactor_validate_args emits the canonical user_error (NOT internal_assert).
 - if no rvar splits: return (already flat).
 - project_rdom(all rvar leaf dims, rdom, rvar_splits) -> flat_rdom + projection.
 - RFactorProjection{rdom_promises(ORIGINAL tight bounds), projection}:
     def.args()/values() = project(...)   (promise_clamped keeps original bound)
     def.predicate() = strip_reduction_domain(simplify(project(flat_rdom.predicate())))
     sched.rvars()=flat_rdom.domain(); sched.dims()=project(dims); sched.splits()=var_splits.

## Bugs found & fixed during bring-up (all in Phase A)
1. Predicate carried transient flat_rdom back-pointers -> "predicate may only
   refer to its own RVars". Fix: strip reduction_domain from predicate vars
   (bare, like a normal def predicate) so factoring phase can re-home it.
2. Skipping bounds promises dropped the ORIGINAL tight rvar bound; reconstructed
   split index (4*rxo+rxi in [0,11]) over-estimated input read region (extent 12
   vs 10) -> out-of-bounds. Fix: apply rdom_promises with ORIGINAL bounds so the
   composite index is promise_clamped(...,0,9); nested promise in Phase B is
   harmless (outer bound dominates). Verified: input bounds query 0..10 in both.
3. internal_assert on illegal fuse -> SIGABRT(134) in error test. Fix: bail early.

## Results
- build/exp (Debug, LLVM21). correctness_rfactor: baseline=0, two-phase=0.
- error_rfactor_after_var_and_rvar_fusion: baseline=1, two-phase=1.
- Lowered-stmt parity verified on rfactor_precise_bounds_test (tile+rfactor).
- Default path (env unset) is inert: only a get_env_variable check added.

## UNIFICATION (done): rfactor + hoist_invariants share Phase A
- Env var renamed HL_RFACTOR_TWO_PHASE -> HL_MATERIALIZE_REDUCTION_DOMAIN (neutral).
- Shared entry maybe_materialize_reduction_domain(Definition&) at Func.cpp:912;
  called by BOTH Stage::rfactor (1209) and Stage::hoist_invariants (1432).
- hoist_invariants over a flat domain: intermediates reduce over the materialized
  leaf rvars directly (schedule copy is now split-free); write-back eliminates
  them via the flat reduction_bounds. Verified split-before-hoist:
    baseline    C_intm rvars: k$x   splits=1
    materialized C_intm rvars: r11 r10  splits=0   (Phase A fired)
- Results (build/exp): correctness_rfactor 0/0, correctness_hoist_invariants 0/0,
  error_rfactor_after_var_and_rvar_fusion 1/1 (baseline/materialized).

## Follow-ups (not done)
- Broader: run full correctness suite in materialized mode (heavy). Env-gated, low risk.
- Promote to default / delete the split-projection path (project_rdom's split replay
  and rfactor_validate_args' partition) once confident, making Phase A the only
  place splits are lowered for reductions.

## Key refs (all src/Func.cpp unless noted)
- Stage::rfactor: 1053 ; rfactor_validate_args: 959 ; split partition loop: 1006-1049
- project_rdom: 708 ; RFactorProjection: 686 ; add_let: 669 ; substitute_self_reference: 611
- apply_split / compute_loop_bounds_after_split: src/ApplySplit.h
- Split struct: src/Schedule.h ; Definition: src/Definition.h (args/values/predicate + schedule)

---
# (Earlier task) Understand Halide scheduling directives

Task: download Halide (done) and explain how these schedule-time directives work:
1. rfactor()
2. hoist_invariants()  (hinted "branch/PR" — dev history on origin/alexreinking/rfactor-hoisting; merged to main via PR #9190)
3. clone_in() / in()

## Status
- [x] Clone Halide to /Users/teoc/projects/halides (full clone, all remote branches). HEAD=main d33fd254f
- [x] Locate implementations (all in src/Func.cpp + supporting passes)
- [x] rfactor research DONE (agent report received)
- [x] hoist_invariants research DONE (agent report received)
- [x] in/clone_in research DONE (agent report received + verified WrapCalls removal)
- [x] Synthesize explanation for user — DONE

## Headline finding
This tree (halide/Halide main @ d33fd254f, ~Sept 2026) has REMOVED the WrapCalls lowering pass.
in()/clone_in() custom wrappers are applied EAGERLY at schedule time (Function::substitute_calls
+ freeze); the global wrapper in() is a lazy FunctionContents::global_wrapper link resolved via
follow_global_wrappers in FunctionPtr::get(), materialized at Lower.cpp:165 build_environment+deep_copy.
Verified: src/WrapCalls.* absent, no refs; eager loop at Func.cpp:2705-2726.

## Concrete usage examples found (main)
- test/correctness/rfactor.cpp
- test/correctness/hoist_invariants.cpp — catalogs: sum*scale, min+c, max+c, or&&, and||;
  scattered factors (signed/unsigned); multi-term -> multiple intermediates; tuple flattening
  (FuncVec ordered by tuple idx then term); strict_float preserved; predicated rdom;
  composes-with-rfactor; error cases (invalid law e.g. unsigned-min wrap; nothing-to-hoist).
- test/correctness/func_wrapper.cpp (in), func_clone.cpp (clone_in), in_place.cpp
- test/performance/tiled_matmul_arm_neon.cpp — THE composition example: eager_inline() folds
  Wt/VecDq producers into Acc's update so scale factors surface as leaves -> hoist_invariants()
  lifts WtScale(i)*VecScale out -> rfactor() splits scale-free reduction by block ->
  change_type(Int(32)) retypes inner dot product so CodeGen_ARM emits SDOT.

## Key locations (main branch)
- rfactor: src/Func.cpp:600 (RVar,Var overload), :959 (rfactor_validate_args), :1053 (main impl)
  - RFactorProjection helper struct: src/Func.cpp:686
  - substitute_self_reference: src/Func.cpp:611
  - associativity proving: src/AssociativeOpsTable.cpp, src/Associativity.cpp
- hoist_invariants: src/Func.cpp:1269
  - semiring distributive law table: src/Func.cpp:772
  - HoistedFactor/HoistedTerm: src/Func.cpp:837/846
  - extract_hoisted_terms: src/Func.cpp:900
- in()/clone_in(): src/Func.cpp:2734 / :2753 ; wrap helper ~2546-2730
  - Function::add_wrapper: src/Function.cpp:1311 ; FuncSchedule::add_wrapper: src/Schedule.cpp:498
  - lowering pass that applies wrappers: src/WrapCalls.cpp

## Notes
- eager_inline() (PR #9256) is a sibling schedule-time-rewrite directive, related to rfactor/hoist use.
- Branch origin/alexreinking/rfactor-hoisting = granular dev history of the merged work.
