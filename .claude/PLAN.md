# PLAN — Two-phase rfactor experiment (branch: experiment/rfactor-two-phase)

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
