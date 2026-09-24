#include "Inductive.h"

#include "Bounds.h"
#include "ConciseCasts.h"
#include "Error.h"
#include "ExprUsesVar.h"
#include "Function.h"
#include "IR.h"
#include "IREquality.h"
#include "IRVisitor.h"
#include "Simplify.h"
#include "Substitute.h"

namespace Halide {
namespace Internal {

using std::string;
using std::vector;

namespace {

class BaseCaseSolver : public IRVisitor {
    using IRVisitor::visit;
    const vector<string> &vars;
    const string &func;

    const vector<Interval> &start_box;
    const vector<bool> &is_inductive_var;
    const bool is_update;

    vector<Interval> condition_intervals;

    // The region of each var over which the recursive branch being visited
    // is taken.
    vector<Interval> branch_intervals;

    Scope<Interval> bounds;

    Scope<> inductive_vars;

    int nested_select = 0;

    void visit(const Call *op) override {
        if (op->is_intrinsic(Call::if_then_else)) {

            // A select whose condition doesn't depend on any inductive var
            // can be traversed normally. So can one whose condition reads
            // the function itself: that is a data-dependent step of the
            // recurrence, not a base-case guard, and both of its branches
            // may recurse (each self-reference in them is still checked).
            bool cond_is_inductive = expr_uses_vars(op->args[0], inductive_vars);
            bool cond_recurses = false;
            visit_with(op->args[0], [&](auto *self, const Call *inner_op) {
                if (inner_op->name == func) {
                    cond_recurses = true;
                }
                self->visit_base(inner_op);
            });
            if (!cond_is_inductive || cond_recurses) {
                IRVisitor::visit(op);
                return;
            }

            // Theoretically there is no need to check op->args[0].
            // Select nodes are only converted to if_then_else when the condition is pure,
            // which means the condition cannot have any recursive calls.
            // std::cout<<"cond is" << op->args[0];
            op->args[0].accept(this);

            bool left_recurse = false, right_recurse = false;
            visit_with(op->args[1], [&](auto *self, const Call *inner_op) {
                if (inner_op->name == func) {
                    left_recurse = true;
                }
                self->visit_base(inner_op);
            });
            visit_with(op->args[2], [&](auto *self, const Call *inner_op) {
                if (inner_op->name == func) {
                    right_recurse = true;
                }
                self->visit_base(inner_op);
            });
            // Again, this check is theoretically unnecessary
            user_assert(!(left_recurse && right_recurse)) << "Select node in inductive function " << func << " does not have a base case";

            nested_select += 1;
            vector<Interval> old_intervals = condition_intervals;
            if (left_recurse) {
                for (size_t i = 0; i < vars.size(); i++) {
                    Interval inter = Interval::make_intersection(old_intervals[i], solve_for_outer_interval(simplify(op->args[0]), vars[i]));
                    condition_intervals[i] = Interval(inter.min, Interval::pos_inf());
                    branch_intervals[i] = inter;
                    bounds.push(vars[i], condition_intervals[i]);
                }
                op->args[1].accept(this);
                for (const auto &var : vars) {
                    bounds.pop(var);
                }
            }
            if (right_recurse) {
                for (size_t i = 0; i < vars.size(); i++) {
                    Interval inter = Interval::make_intersection(old_intervals[i], solve_for_outer_interval(simplify(!op->args[0]), vars[i]));
                    condition_intervals[i] = Interval(inter.min, Interval::pos_inf());
                    branch_intervals[i] = inter;
                    bounds.push(vars[i], condition_intervals[i]);
                }
                op->args[2].accept(this);
                for (const auto &var : vars) {
                    bounds.pop(var);
                }
            }
            condition_intervals = old_intervals;
            branch_intervals = vector<Interval>(vars.size(), Interval::everything());
            nested_select -= 1;
        } else if (op->name == func) {
            // An update definition's self-reference whose inductive args match the
            // pure vars exactly is acceptable.
            if (is_update) {
                bool exact_pure_self_reference = true;
                for (size_t position = 0; position < vars.size() && exact_pure_self_reference; position++) {
                    if (is_inductive_var[position] &&
                        (position >= op->args.size() || !equal(op->args[position], Variable::make(op->args[position].type(), vars[position])))) {
                        exact_pure_self_reference = false;
                    }
                }
                if (exact_pure_self_reference) {
                    IRVisitor::visit(op);
                    return;
                }
            }

            user_assert(nested_select > 0) << "Function " << func << " contains an inductive function reference outside of a select operation value.\n";
            user_assert(nested_select == 1) << "Function " << func << " contains an inductive function reference inside a nested select operation.\n";
            bool found_inductive = false;
            for (size_t position = 0; position < vars.size(); position++) {
                const Expr inductive_expr = op->args[position];
                if (!is_inductive_var[position]) {
                    continue;
                }
                const Expr new_v = Variable::make(inductive_expr.type(), vars[position]);
                const Expr gets_lower = simplify(new_v - inductive_expr > 0, bounds);
                const Interval i_lower = solve_for_inner_interval(gets_lower, vars[position]);

                Interval new_interval;
                if (equal(new_v, inductive_expr)) {
                    new_interval = start_box[position];
                } else if (i_lower.is_everything()) {
                    found_inductive = true;
                    new_interval = Interval(Interval::neg_inf(), start_box[position].max);
                } else {
                    std::ostringstream err;
                    err << "Inductive variable " << vars[position] << " in inductive function " << func << " is not provably monotonically decreasing outside of the base case.";
                    user_error << err.str() << "\n";
                }
                new_interval = Interval::make_intersection(new_interval, condition_intervals[position]);
                Scope<Interval> i_scope;
                i_scope.push(vars[position], new_interval);
                result_intervals[position] = Interval::make_union(result_intervals[position], Interval::make_union(new_interval, bounds_of_expr_in_scope(inductive_expr, i_scope)));
            }
            user_assert(found_inductive) << "Unable to prove in inductive function " << func << " that the inductive step is monotonically decreasing.\n";

            // A non-inductive var the step reads at an offset (a stencil over
            // the previous slice) widens by that offset at every step back.
            // Record how far each step reaches, and where the reads can land
            // given the region the recursive branch is taken over.
            for (size_t position = 0; position < vars.size() && !is_update; position++) {
                const Expr &arg = op->args[position];
                const Expr v = Variable::make(arg.type(), vars[position]);
                if (is_inductive_var[position] || equal(arg, v)) {
                    continue;
                }
                Interval offset = bounds_of_expr_in_scope(simplify(arg - v), bounds);
                user_assert(offset.is_bounded())
                    << "Unable to bound how far inductive function " << func
                    << " reads itself along " << vars[position] << ".\n";
                neighbour_reach[position] = Interval::make_union(neighbour_reach[position], offset);
                const Interval &branch = branch_intervals[position];
                Interval lands(branch.has_lower_bound() ? branch.min + offset.min : Interval::neg_inf(),
                               branch.has_upper_bound() ? branch.max + offset.max : Interval::pos_inf());
                neighbour_lands[position] = Interval::make_union(neighbour_lands[position], lands);
            }

            IRVisitor::visit(op);

        } else {
            IRVisitor::visit(op);
        }
    }

public:
    vector<Interval> result_intervals;

    // For each non-inductive var, the offsets at which the recursive steps
    // read it, and the region those reads land in.
    vector<Interval> neighbour_reach, neighbour_lands;

    BaseCaseSolver(const vector<string> &v, const vector<bool> &is_inductive_var, const string &func, const vector<Interval> &con, bool is_update)
        : vars(v), func(func), start_box(con), is_inductive_var(is_inductive_var), is_update(is_update) {
        condition_intervals = vector<Interval>(start_box.size());
        branch_intervals = vector<Interval>(start_box.size(), Interval::everything());
        result_intervals = vector<Interval>(start_box.size(), Interval::nothing());
        neighbour_reach = vector<Interval>(start_box.size(), Interval::nothing());
        neighbour_lands = vector<Interval>(start_box.size(), Interval::nothing());
        for (size_t i = 0; i < vars.size(); i++) {
            if (is_inductive_var[i]) {
                inductive_vars.push(vars[i]);
            }
        }
    }
};

}  // anonymous namespace

Box expand_to_include_base_case(const vector<string> &vars, const vector<bool> &is_inductive_var,
                                const Expr &RHS, const string &func, const Box &box_required,
                                bool is_update) {
    Expr substed = substitute_in_all_lets(RHS);
    Box box2 = box_required;
    BaseCaseSolver b(vars, is_inductive_var, func, box_required.bounds, is_update);
    substed.accept(&b);

    Scope<Interval> non_inductive_scope;
    for (size_t i = 0; i < vars.size(); i++) {
        if (!is_inductive_var[i]) {
            non_inductive_scope.push(vars[i], box_required[i]);
        }
    }

    // Each step back along an inductive var moves at least one down it, so
    // the number of steps from the region required down to the base case
    // is at most the sum of how far each inductive var falls.
    Expr steps = 0;
    for (size_t i = 0; i < vars.size(); i++) {
        user_assert(b.result_intervals[i].is_bounded() || b.result_intervals[i].is_empty()) << "Unable to prove that the inductive function " << func << " uses a bounded interval";
        if (!b.result_intervals[i].is_empty()) {
            Interval new_interval(min(b.result_intervals[i].min, box_required[i].min), box_required[i].max);

            new_interval.min = bounds_of_expr_in_scope(new_interval.min, non_inductive_scope).min;
            box2[i] = new_interval;
            steps += box_required[i].max - new_interval.min;
        }
    }
    steps = simplify(max(steps, 0));

    Scope<> all_vars;
    for (const auto &v : vars) {
        all_vars.push(v);
    }

    // A non-inductive var read at an offset widens by that offset at every
    // step, but never past where the recursive branch's reads can land.
    for (size_t i = 0; i < vars.size(); i++) {
        const Interval &reach = b.neighbour_reach[i];
        if (is_inductive_var[i] || reach.is_empty() || !box_required[i].is_bounded()) {
            continue;
        }
        Expr lo = bounds_of_expr_in_scope(min(reach.min, 0), non_inductive_scope).min;
        Expr hi = bounds_of_expr_in_scope(max(reach.max, 0), non_inductive_scope).max;
        Expr new_min = box_required[i].min + steps * lo;
        Expr new_max = box_required[i].max + steps * hi;
        const Interval &lands = b.neighbour_lands[i];
        if (lands.has_lower_bound() && !expr_uses_vars(lands.min, all_vars)) {
            new_min = max(new_min, lands.min);
        }
        if (lands.has_upper_bound() && !expr_uses_vars(lands.max, all_vars)) {
            new_max = min(new_max, lands.max);
        }
        box2[i] = Interval(simplify(min(box_required[i].min, new_min)),
                           simplify(max(box_required[i].max, new_max)));
    }

    return box2;
}

}  // namespace Internal
}  // namespace Halide
