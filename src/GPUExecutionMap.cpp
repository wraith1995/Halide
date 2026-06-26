#include "GPUExecutionMap.h"

#include <functional>

#include "CanonicalizeGPUVars.h"  // gpu_thread_name
#include "ExprUsesVar.h"
#include "IROperator.h"
#include "IRVisitor.h"
#include "Simplify.h"

namespace Halide {
namespace Internal {

const char *exec_scope_name(ExecScope s) {
    switch (s) {
    case ExecScope::Thread:
        return "Thread";
    case ExecScope::Warp:
        return "Warp";
    case ExecScope::WarpGroup:
        return "WarpGroup";
    case ExecScope::Block:
        return "Block";
    case ExecScope::Cluster:
        return "Cluster";
    case ExecScope::Grid:
        return "Grid";
    }
    return "?";
}

namespace {

// Discover the thread/lane/warp-group axes of a kernel (the block geometry). Every
// For with for_type GPUThread/GPULane is an axis; a warps_per_group>0 loop is a
// warp-group collective axis (its single unit spans warps_per_group*warp_size lanes,
// the threads implicit). Axes are deduped by loop-variable name, taking the max extent
// (matching ExtractBlockSize), so sibling branches that reuse a canonical thread name
// collapse to one axis.
class DiscoverAxes : public IRVisitor {
    int warp_size;
    using IRVisitor::visit;

    void visit(const For *op) override {
        if (op->for_type == ForType::GPUThread || op->for_type == ForType::GPULane) {
            int weight = op->warps_per_group > 0 ? op->warps_per_group * warp_size : 1;
            record(op->name, op->min, op->max, weight, op->warps_per_group > 0);
        }
        IRVisitor::visit(op);
    }

    void record(const std::string &var, const Expr &mn, const Expr &mx, int weight, bool is_wg) {
        for (ExecAxis &a : axes) {
            if (a.var == var) {
                a.extent = simplify(Max::make(a.extent, Add::make(Sub::make(mx, mn), 1)));
                return;
            }
        }
        ExecAxis a;
        a.var = var;
        a.min = mn;
        a.extent = simplify(Add::make(Sub::make(mx, mn), 1));
        a.lane_weight = weight;
        a.is_warp_group = is_wg;
        axes.push_back(a);
    }

public:
    std::vector<ExecAxis> axes;
    explicit DiscoverAxes(int ws)
        : warp_size(ws) {
    }
};

// Is this a simple comparison of a single Variable against a constant-ish bound?
// Returns the variable name and narrows `iv` to the half-line it implies.
bool single_var_cmp(const Expr &cond, std::string *var, Interval *iv) {
    auto match = [&](const Expr &a, const Expr &b, char op) -> bool {
        const Variable *v = a.as<Variable>();
        if (!v || expr_uses_var(b, v->name)) {
            return false;
        }
        *var = v->name;
        switch (op) {
        case '=':  // a == b
            *iv = Interval::single_point(b);
            return true;
        case '<':  // a < b  -> [-inf, b-1]
            *iv = Interval(Interval::neg_inf(), simplify(b - 1));
            return true;
        case 'l':  // a <= b -> [-inf, b]
            *iv = Interval(Interval::neg_inf(), b);
            return true;
        case '>':  // a > b  -> [b+1, inf]
            *iv = Interval(simplify(b + 1), Interval::pos_inf());
            return true;
        case 'g':  // a >= b -> [b, inf]
            *iv = Interval(b, Interval::pos_inf());
            return true;
        }
        return false;
    };
    if (const EQ *e = cond.as<EQ>()) {
        return match(e->a, e->b, '=') || match(e->b, e->a, '=');
    } else if (const LT *e = cond.as<LT>()) {
        // a < b, or flip b > a
        return match(e->a, e->b, '<') || match(e->b, e->a, '>');
    } else if (const LE *e = cond.as<LE>()) {
        return match(e->a, e->b, 'l') || match(e->b, e->a, 'g');
    } else if (const GT *e = cond.as<GT>()) {
        return match(e->a, e->b, '>') || match(e->b, e->a, '<');
    } else if (const GE *e = cond.as<GE>()) {
        return match(e->a, e->b, 'g') || match(e->b, e->a, 'l');
    }
    return false;
}

}  // namespace

ExecMap::ExecMap(const Stmt &kernel, int warp_size)
    : warp_size_(warp_size) {
    DiscoverAxes d(warp_size_);
    kernel.accept(&d);
    axes_ = std::move(d.axes);
    stack_.push_back(ActiveSet::whole_block());
}

const ExecAxis *ExecMap::axis_for_var(const std::string &var) const {
    for (const ExecAxis &a : axes_) {
        if (a.var == var) {
            return &a;
        }
    }
    // 1-D flat-id decode: a guard on an unknown var (e.g. partition_warp_groups' flat
    // warp id) maps onto the single thread axis when the block is 1-D.
    if (axes_.size() == 1) {
        return &axes_[0];
    }
    return nullptr;
}

Interval ExecMap::axis_interval(const ActiveSet &a, const ExecAxis &ax) const {
    auto it = a.narrowed.find(ax.var);
    if (it != a.narrowed.end()) {
        return it->second;
    }
    return Interval(ax.min, simplify(ax.min + ax.extent - 1));
}

ActiveSet ExecMap::narrow(const ActiveSet &a, const std::string &var, const Interval &iv) const {
    const ExecAxis *ax = axis_for_var(var);
    if (!ax) {
        return a;  // unrecognized axis: conservatively do not narrow
    }
    ActiveSet r = a;
    Interval cur = axis_interval(a, *ax);
    Interval next = Interval::make_intersection(cur, iv);
    // Clamp to the axis's own extent.
    next = Interval::make_intersection(next, Interval(ax->min, simplify(ax->min + ax->extent - 1)));
    next.min = simplify(next.min);
    next.max = simplify(next.max);
    if (next.is_empty() ||
        (next.has_lower_bound() && next.has_upper_bound() && can_prove(next.min > next.max))) {
        r.empty = true;
    }
    r.narrowed[ax->var] = next;
    return r;
}

void ExecMap::enter_for(const For *op) {
    ActiveSet cur = current();
    if (op->for_type == ForType::GPUThread || op->for_type == ForType::GPULane) {
        cur = narrow(cur, op->name, Interval(op->min, op->max));
    }
    stack_.push_back(cur);
}

bool ExecMap::recognizes_guard(const Expr &condition) const {
    std::string var;
    Interval iv;
    if (const And *a = condition.as<And>()) {
        return recognizes_guard(a->a) && recognizes_guard(a->b);
    }
    if (single_var_cmp(condition, &var, &iv)) {
        return axis_for_var(var) != nullptr;
    }
    return false;
}

void ExecMap::enter_if(const Expr &condition) {
    ActiveSet cur = current();
    std::function<ActiveSet(const Expr &, ActiveSet)> apply =
        [&](const Expr &cond, ActiveSet a) -> ActiveSet {
        if (const And *aa = cond.as<And>()) {
            return apply(aa->b, apply(aa->a, a));
        }
        std::string var;
        Interval iv;
        if (single_var_cmp(cond, &var, &iv)) {
            return narrow(a, var, iv);
        }
        return a;  // unrecognized: conservatively no narrowing (more sync, never less)
    };
    stack_.push_back(apply(condition, cur));
}

void ExecMap::pop() {
    internal_assert(stack_.size() > 1) << "ExecMap::pop underflow\n";
    stack_.pop_back();
}

Expr ExecMap::block_lanes() const {
    Expr p = 1;
    for (const ExecAxis &a : axes_) {
        p = simplify(p * a.extent * a.lane_weight);
    }
    return p;
}

Expr ExecMap::elected_lane(const ActiveSet &a) const {
    if (get_env_variable("HL_ELECT_DEBUG") == "1") {
        std::cerr << "ELECT axes:";
        for (const ExecAxis &ax : axes_) {
            Interval iv = axis_interval(a, ax);
            std::cerr << " {" << ax.var << " min=" << ax.min << " ext=" << ax.extent
                      << " w=" << ax.lane_weight << " | active=[" << iv.min << "," << iv.max << "]}";
        }
        std::cerr << "\n";
    }
    // The hardware thread index of the region's first lane = its lane OFFSET from the block's
    // first lane, summed over axes: (each axis's narrowed min - its base) * lane_weight. This is
    // correct whether the warp-spec split narrowed a wide thread axis (weight 1) or a warp-group
    // axis (weight = warps_per_group*warp_size), and ignores degenerate (ext-1) fused axes since
    // they sit at their base. Whole block => every axis at its base => 0 (== legacy global-tid==0).
    Expr lane = 0;
    for (const ExecAxis &ax : axes_) {
        Interval iv = axis_interval(a, ax);
        lane = simplify(lane + (iv.min - ax.min) * ax.lane_weight);
    }
    return lane;
}

Expr ExecMap::count(const ActiveSet &a) const {
    if (a.empty) {
        return 0;
    }
    Expr p = 1;
    for (const ExecAxis &ax : axes_) {
        Interval iv = axis_interval(a, ax);
        Expr size = simplify(iv.max - iv.min + 1);
        p = simplify(p * size * ax.lane_weight);
    }
    return p;
}

ExecScope ExecMap::scope(const ActiveSet &a) const {
    if (a.empty) {
        return ExecScope::Thread;
    }
    Expr c = count(a);
    if (is_const_one(c)) {
        return ExecScope::Thread;
    }
    // Whole block? Every axis is at its full extent.
    bool whole = true;
    for (const ExecAxis &ax : axes_) {
        Interval iv = axis_interval(a, ax);
        bool full = is_const_zero(simplify(iv.min - ax.min)) &&
                    is_const_zero(simplify(iv.max - (ax.min + ax.extent - 1)));
        if (!full) {
            whole = false;
            break;
        }
    }
    if (whole) {
        return ExecScope::Block;
    }
    // A proper, contiguous sub-block: a named (warp-group) barrier with an arbitrary
    // participant count covers it. (Warp/Cluster classification deferred — the selector
    // lowers only Thread/WarpGroup/Block today.)
    return ExecScope::WarpGroup;
}

ActiveSet ExecMap::join(const ActiveSet &a, const ActiveSet &b) {
    if (a.empty) {
        return b;
    }
    if (b.empty) {
        return a;
    }
    ActiveSet r;
    // Only axes narrowed in BOTH can stay narrowed; an axis full in either is full in
    // the hull. (axis_interval would need the model; here we operate on narrowed maps
    // directly — a missing key == full, so union with full == full == drop the key.)
    for (const auto &kv : a.narrowed) {
        auto it = b.narrowed.find(kv.first);
        if (it != b.narrowed.end()) {
            Interval h = Interval::make_union(kv.second, it->second);
            h.min = simplify(h.min);
            h.max = simplify(h.max);
            r.narrowed[kv.first] = h;
        }
    }
    return r;
}

bool ExecMap::same_single_lane(const ActiveSet &a, const ActiveSet &b) const {
    if (a.empty || b.empty) {
        return false;
    }
    if (!is_const_one(count(a)) || !is_const_one(count(b))) {
        return false;
    }
    // Same single lane: identical interval on every axis.
    for (const ExecAxis &ax : axes_) {
        Interval ia = axis_interval(a, ax);
        Interval ib = axis_interval(b, ax);
        if (!(is_const_zero(simplify(ia.min - ib.min)) &&
              is_const_zero(simplify(ia.max - ib.max)))) {
            return false;
        }
    }
    return true;
}

}  // namespace Internal
}  // namespace Halide
