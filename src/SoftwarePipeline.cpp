#include "SoftwarePipeline.h"

#include <vector>

#include "Func.h"
#include "Function.h"
#include "IR.h"
#include "IREquality.h"
#include "IRMutator.h"
#include "IROperator.h"
#include "Schedule.h"
#include "Simplify.h"
#include "Substitute.h"
#include "UniquifyVariableNames.h"
#include "Util.h"

namespace Halide {
namespace Internal {

namespace {

// Rebuild a chain of LetStmts (outermost first) around a body.
Stmt wrap_lets(const std::vector<std::pair<std::string, Expr>> &lets, Stmt body) {
    for (auto it = lets.rbegin(); it != lets.rend(); ++it) {
        body = LetStmt::make(it->first, it->second, body);
    }
    return body;
}

// Substitute the loop variable `var` with `value` throughout a hoisted unit.
// Because the ring-buffer slot index and the data address are both pure
// functions of `var` at this (post-flatten) stage, this single substitution
// moves the produce/consume to the correct iteration AND the correct ring slot.
Stmt at(const std::string &var, const Expr &value, const Stmt &unit) {
    return simplify(substitute(var, value, unit));
}

class SoftwarePipeline : public IRMutator {
    const std::map<std::string, Function> &env;

    using IRMutator::visit;

    // Is `name` a producer the user asked to software-pipeline?
    bool is_pipelined(const std::string &name) const {
        auto it = env.find(name);
        return it != env.end() && it->second.schedule().software_pipeline();
    }

    // The pipeline depth Q (ring_buffer extent) for producer `name`, or 0 if not
    // a positive compile-time constant (we only pipeline statically-sized rings).
    int ring_depth(const std::string &name) const {
        auto it = env.find(name);
        if (it == env.end()) {
            return 0;
        }
        Expr q = it->second.schedule().ring_buffer();
        if (!q.defined()) {
            return 0;
        }
        auto qi = as_const_int(simplify(q));
        return (qi && *qi > 0) ? (int)*qi : 0;
    }

    Stmt visit(const For *op) override {
        // Rewrite children first, then try to recognize/transform this loop.
        Stmt body = mutate(op->body);

        // Walk the LEADING RUN of pipelined producers, peeling per-iteration LetStmts as we go
        // (e.g. each producer's base-address let). A GEMM mainloop is
        //   produce As; produce Bs; consume Bs { consume As { compute } }
        // -- two flat leading produces, then one nested consumer. We collect every pipelined
        // producer at the front and treat the remainder as the single shared consumer.
        std::vector<std::pair<std::string, Expr>> lets;
        std::vector<const ProducerConsumer *> producers;
        Stmt inner = body;
        while (true) {
            while (const LetStmt *l = inner.as<LetStmt>()) {
                lets.emplace_back(l->name, l->value);
                inner = l->body;
            }
            const Block *blk = inner.as<Block>();
            if (!blk) {
                break;
            }
            const ProducerConsumer *p = blk->first.as<ProducerConsumer>();
            if (!p || !p->is_producer || !is_pipelined(p->name)) {
                break;  // a non-pipelined producer (or the consumer) ends the run
            }
            producers.push_back(p);
            inner = blk->rest;
        }
        if (producers.empty()) {
            return rebuild(op, body);
        }
        Stmt consumer_stmt = inner;  // the shared consumer of all collected producers

        // --- Legality preconditions (refuse -> leave synchronous, still correct) ---
        // 1) Must be a serial loop: a skew across a parallel/vectorized/GPU axis is
        //    meaningless (no in-order relationship to lead).
        if (op->for_type != ForType::Serial) {
            user_warning << "software_pipeline() ignored on loop \"" << op->name
                         << "\": not a serial loop.\n";
            return rebuild(op, body);
        }
        // 2) Every collected producer must share one positive compile-time ring depth Q
        //    (so a single lead D = Q-1 leads them together). Mixed depths -> fall back.
        const int Q = ring_depth(producers[0]->name);
        for (const ProducerConsumer *p : producers) {
            if (ring_depth(p->name) != Q) {
                user_warning << "software_pipeline() ignored on loop \"" << op->name
                             << "\": producers have mismatched ring_buffer depths.\n";
                return rebuild(op, body);
            }
        }
        const int D = Q - 1;
        if (D < 1) {
            return rebuild(op, body);
        }
        // 3) For lead D > 1 the prologue produces indices [min, min+D-1]; they must be
        //    in-range, i.e. the loop runs at least D times. (D == 1 is always safe.)
        Expr n_iters = simplify(op->max - op->min + 1);
        if (D > 1 && !can_prove(n_iters >= D)) {
            user_warning << "software_pipeline() ignored on loop \"" << op->name
                         << "\": cannot prove it runs >= " << D << " times.\n";
            return rebuild(op, body);
        }

        const std::string &v = op->name;
        Expr vv = Variable::make(Int(32), v);

        // GPU ring producers (GPUShared) need the mbarrier COMPLETION carried by the rotation: the
        // TMA-loaded tile is consumed D iters after it is issued, so the wgmma must wait on the ring
        // slot's full_mbar with the tile's parity. We emit TILE-INDEXED markers -- an async_issue in
        // each produce (so inject_tma takes the RING path: it writes the ring slot and arrives
        // full_mbar[slot]) and an async_wait before each consume -- both as functions of the loop var,
        // so substitution moves slot AND parity exactly. The selector lowers them to mbarrier ops, the
        // SAME mechanism the fork uses (shared selector, per-topology emission). CPU producers get no
        // markers (the register/MLP data dependency is the completion). NFC: this never runs for the
        // .async() fork path (those producers are not software_pipeline()'d).
        bool gpu = false;
        {
            auto it = env.find(producers[0]->name);
            gpu = it != env.end() && it->second.schedule().memory_type() == MemoryType::GPUShared;
        }
        // The explicit empty (WAR / buffer-reuse) mbarrier edge -- the CUTLASS PipelineTmaAsync dual of
        // the full edge. Gated (default OFF) so the default uniform pipeline output stays byte-identical;
        // it pairs with HL_WG_MEMBAR, which (once the empty edge makes correctness independent of the
        // conservative bar.sync wall) may remove that wall. See research/empty_mbarrier_subproject.md.
        const bool emit_empty = gpu && get_env_variable("HL_WG_SP_EMPTY") == "1";

        // Build one hoisted unit per producer + one for the shared consumer. The lets are pure
        // address math, so duplicating them into each unit is safe (the simplifier drops unused
        // ones); substituting the loop var then moves both the data address AND the ring slot.
        std::vector<Stmt> produce_units;
        produce_units.reserve(producers.size());
        for (const ProducerConsumer *p : producers) {
            Stmt pbody = p->body;
            if (emit_empty) {
                // Empty (WAR) edge -- producer_acquire: BEFORE overwriting ring slot v%Q with this
                // tile's TMA, wait until the consumer has finished reading the slot's PREVIOUS occupant
                // (tile v-Q). The consumer signals that via async_release on empty_mbar[v%Q]. Slot s is
                // released after consuming tiles s, s+Q, ...; the producer of tile v (=s+kQ, k=v/Q)
                // needs release #(k-1), which completed empty-phase (k-1) -- parity (v/Q+1)%2 (== (k-1)
                // mod 2, kept non-negative). SKIP the first Q tiles per slot (v < min+Q): the slots
                // start free, so there is no prior occupant to wait on (matches mbarrier_init phase 0).
                Stmt acquire = Evaluate::make(Call::make(
                    Int(32), Call::async_acquire,
                    {empty_mbar_ref(p->name, vv % Q), (vv / Q + 1) % 2}, Call::Intrinsic));
                acquire = IfThenElse::make(vv >= op->min + Q, acquire);
                pbody = Block::make(acquire, pbody);
            }
            if (gpu) {
                // After the cooperative store (match_tile_copy reads the store), arm the ring slot:
                // async_issue(CpAsyncBulk, full_mbar[v%Q], bytes) -- find_ring_mbar grabs args[1].
                Stmt issue = Evaluate::make(Call::make(
                    Int(32), Call::async_issue,
                    {Expr((int)CompletionKind::CpAsyncBulk), mbar_ref(p->name, vv % Q), Expr(0)},
                    Call::Intrinsic));
                pbody = Block::make(pbody, issue);
            }
            produce_units.push_back(wrap_lets(lets, ProducerConsumer::make(p->name, true, pbody)));
        }
        Stmt consumer_body = consumer_stmt;
        if (gpu) {
            // Before the shared consumer, wait on every producer's tile: async_wait(CpAsyncBulk, Block,
            // full_mbar[v%Q], parity=(v/Q)%2). The selector lowers to mbarrier_try_wait.
            Expr parity = (vv / Q) % 2;
            Stmt waits;
            for (const ProducerConsumer *p : producers) {
                Stmt w = Evaluate::make(Call::make(
                    Int(32), Call::async_wait,
                    {Expr((int)CompletionKind::CpAsyncBulk), Expr((int)SyncScope::Block),
                     mbar_ref(p->name, vv % Q), parity},
                    Call::Intrinsic));
                waits = waits.defined() ? Block::make(waits, w) : w;
            }
            consumer_body = Block::make(waits, consumer_stmt);
        }
        if (emit_empty) {
            // Empty (WAR) edge -- consumer_release: AFTER the consumer drains its wgmma read of slot
            // v%Q (the wgmma.wait_group is emitted at the tail of the consumer's collective, so any
            // statement sequenced after consumer_stmt runs once the async read has retired), signal the
            // slot free so the producer Q iterations ahead may reuse it. Every consumer thread arrives
            // (a plain mbarrier.arrive, +1 each); empty_mbar's expected count is the block thread total,
            // patched once it is known (FuseGPUThreadLoops). One release per producer ring slot.
            for (const ProducerConsumer *p : producers) {
                Stmt rel = Evaluate::make(Call::make(
                    Int(32), Call::async_release, {empty_mbar_ref(p->name, vv % Q)}, Call::Intrinsic));
                consumer_body = Block::make(consumer_body, rel);
            }
        }
        Stmt consume_unit = wrap_lets(lets, consumer_body);

        // Produce EVERY pipelined producer at iteration `idx` (preserving original order).
        auto produce_all = [&](const Expr &idx) {
            Stmt s;
            for (const Stmt &pu : produce_units) {
                Stmt one = at(v, idx, pu);
                s = s.defined() ? Block::make(s, one) : one;
            }
            return s;
        };

        // Prologue: produce all, for the first D iterations [min, min+D-1].
        Stmt prologue;
        for (int d = 0; d < D; d++) {
            Stmt p = produce_all(op->min + d);
            prologue = prologue.defined() ? Block::make(prologue, p) : p;
        }

        // Steady state: for v in [min, max-D], produce all at v+D, then consume at v.
        Stmt steady_body = Block::make(produce_all(Variable::make(Int(32), v) + D), consume_unit);
        Stmt steady = For::make(v, op->min, op->max - D, op->for_type, op->partition_policy,
                                op->device_api, steady_body, op->realization, op->warps_per_group);

        // Epilogue: consume the last D iterations [max-D+1, max].
        Stmt epilogue;
        for (int d = 0; d < D; d++) {
            Stmt c = at(v, op->max - D + 1 + d, consume_unit);
            epilogue = epilogue.defined() ? Block::make(epilogue, c) : c;
        }

        Stmt result = Block::make({prologue, steady, epilogue});
        if (gpu) {
            // Allocate + init one depth-Q full_mbar per producer, wrapping the rotated region. count=1:
            // a TMA producer arrives exactly once (expect_tx). PatchTmaMbarCounts keeps it at 1.
            for (const ProducerConsumer *p : producers) {
                Stmt init = Evaluate::make(Call::make(Int(32), "mbarrier_init",
                                                      {mbar_ref(p->name, 0), Expr(Q), Expr(1)},
                                                      Call::Intrinsic));
                result = Block::make(init, result);
                result = Allocate::make(p->name + ".full_mbar", UInt(64), MemoryType::GPUShared,
                                        {Expr(Q)}, const_true(), result);
                if (emit_empty) {
                    // The dual empty_mbar (buffer-reuse/WAR edge). Its expected arrival count = the
                    // number of consumer threads that arrive (every thread in the block), unknown until
                    // the thread loops are fused -- emit a 0 placeholder; PatchEmptyMbarCounts sets it to
                    // the block thread total in FuseGPUThreadLoops. Phase 0 + "slots start free" is
                    // realized by the skip-first-Q guard on producer_acquire (no arrive precedes the
                    // first Q waits).
                    Stmt einit = Evaluate::make(Call::make(Int(32), "mbarrier_init",
                                                           {empty_mbar_ref(p->name, 0), Expr(Q), Expr(0)},
                                                           Call::Intrinsic));
                    result = Block::make(einit, result);
                    result = Allocate::make(p->name + ".empty_mbar", UInt(64), MemoryType::GPUShared,
                                            {Expr(Q)}, const_true(), result);
                }
            }
        }
        return result;
    }

    // A reference to producer `prod`'s ring full_mbar at ring slot `slot` (a UInt64 shared array of Q).
    static Expr mbar_ref(const std::string &prod, Expr slot) {
        return Load::make(UInt(64), prod + ".full_mbar", std::move(slot), Buffer<>{}, Parameter{},
                          const_true(), ModulusRemainder{});
    }

    // The dual: producer `prod`'s ring empty_mbar at ring slot `slot` (a UInt64 shared array of Q).
    static Expr empty_mbar_ref(const std::string &prod, Expr slot) {
        return Load::make(UInt(64), prod + ".empty_mbar", std::move(slot), Buffer<>{}, Parameter{},
                          const_true(), ModulusRemainder{});
    }

    // Rebuild the For with a (possibly) mutated body, preserving all fork fields.
    Stmt rebuild(const For *op, const Stmt &body) {
        if (body.same_as(op->body)) {
            return op;
        }
        return For::make(op->name, op->min, op->max, op->for_type, op->partition_policy,
                         op->device_api, body, op->realization, op->warps_per_group);
    }

public:
    using IRMutator::mutate;
    explicit SoftwarePipeline(const std::map<std::string, Function> &env)
        : env(env) {
    }
};

}  // namespace

Stmt software_pipeline(Stmt s, const std::map<std::string, Function> &env) {
    // Fast out if nothing requested it.
    bool any = false;
    for (const auto &kv : env) {
        if (kv.second.schedule().software_pipeline()) {
            any = true;
            break;
        }
    }
    if (!any) {
        return s;
    }
    s = SoftwarePipeline(env).mutate(s);
    // Peeling/rotating duplicates produce/consume bodies, so their bound let names (e.g. a TMA
    // descriptor `As.tma_map`) now repeat -- which violates Halide's global name-uniqueness
    // invariant that Simplify enforces. Re-uniquify, exactly as UnrollLoops does after it duplicates
    // loop bodies. GPU-safe: unroll_loops runs the same call on post-canonicalize GPU code.
    s = uniquify_variable_names(s);
    return s;
}

}  // namespace Internal
}  // namespace Halide
