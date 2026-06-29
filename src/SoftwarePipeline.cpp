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

        // Build one hoisted unit per producer + one for the shared consumer. The lets are pure
        // address math, so duplicating them into each unit is safe (the simplifier drops unused
        // ones); substituting the loop var then moves both the data address AND the ring slot.
        std::vector<Stmt> produce_units;
        produce_units.reserve(producers.size());
        for (const ProducerConsumer *p : producers) {
            produce_units.push_back(wrap_lets(lets, ProducerConsumer::make(p->name, true, p->body)));
        }
        Stmt consume_unit = wrap_lets(lets, consumer_stmt);

        const std::string &v = op->name;
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

        return Block::make({prologue, steady, epilogue});
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
