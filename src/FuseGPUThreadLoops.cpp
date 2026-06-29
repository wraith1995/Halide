#include <algorithm>
#include <cmath>
#include <tuple>
#include <utility>

#include "AsyncProducers.h"
#include "Bounds.h"
#include "CSE.h"
#include "CanonicalizeGPUVars.h"
#include "CodeGen_GPU_Dev.h"
#include "CompilerLogger.h"
#include "ExprUsesVar.h"
#include "Function.h"
#include "FuseGPUThreadLoops.h"
#include "GPUExecutionMap.h"
#include "IR.h"
#include "IREquality.h"
#include "IRMutator.h"
#include "IROperator.h"
#include "IRPrinter.h"
#include "Monotonic.h"
#include "Simplify.h"
#include "Solve.h"
#include "Substitute.h"
#include "Target.h"
#include "Util.h"

namespace Halide {
namespace Internal {

using std::map;
using std::pair;
using std::sort;
using std::string;
using std::vector;

namespace {

class ExtractBlockSize : public IRVisitor {
protected:
    Expr block_extent[3], block_count[3];
    string block_var_name[3];

    using IRVisitor::visit;

    void found_thread_for(int dim, const string &name, const Expr &extent) {
        internal_assert(dim >= 0 && dim < 3);
        if (!block_extent[dim].defined()) {
            block_extent[dim] = simplify(extent);
        } else {
            block_extent[dim] = simplify(Max::make(extent, block_extent[dim]));
        }
    }

    void found_block_for(int dim, const string &name, Expr extent) {
        internal_assert(dim >= 0 && dim < 3);
        internal_assert(!block_count[dim].defined());
        block_count[dim] = std::move(extent);
        block_var_name[dim] = name;
    }

    void visit(const For *op) override {
        for (int i = 0; i < 3; i++) {
            if (ends_with(op->name, gpu_thread_name(i))) {
                found_thread_for(i, op->name, op->extent());
            } else if (ends_with(op->name, gpu_block_name(i))) {
                found_block_for(i, op->name, op->extent());
            }
        }

        IRVisitor::visit(op);

        Scope<Interval> scope;
        scope.push(op->name, Interval(op->min, op->max));
        // For non-rectangular thread loops, use a bounding box. We'll inject if statements later.
        for (Expr &e : block_extent) {
            if (e.defined() && expr_uses_var(e, op->name)) {
                e = simplify(common_subexpression_elimination(e));
                e = simplify(bounds_of_expr_in_scope(e, scope).max);
            }
        }
    }

    void visit(const LetStmt *op) override {
        IRVisitor::visit(op);
        for (Expr &e : block_extent) {
            if (e.defined() &&
                expr_uses_var(e, op->name)) {
                e = simplify(Let::make(op->name, op->value, e));
            }
        }
    }

public:
    int blocks_dimensions() const {
        for (int i = 0; i < 3; i++) {
            if (!block_count[i].defined()) {
                return i;
            }
        }
        return 3;
    }

    int threads_dimensions() const {
        for (int i = 0; i < 3; i++) {
            if (!block_extent[i].defined()) {
                return i;
            }
        }
        return 3;
    }

    Expr num_threads(int d) const {
        return block_extent[d];
    }

    Expr num_blocks(int d) const {
        return block_count[d];
    }

    Expr block_var(int d) const {
        // The name of the actual for loop
        return Variable::make(Int(32), block_var_name[d]);
    }

    Expr thread_var(int d) const {
        // Thread variables get canonical names
        return Variable::make(Int(32), gpu_thread_name(d));
    }
};

class NormalizeDimensionality : public IRMutator {
protected:
    using IRMutator::visit;

    const ExtractBlockSize &block_size;
    const DeviceAPI device_api;

    int depth = 0;
    int max_depth = 0;

    Stmt wrap(Stmt s) {
        if (depth != 0) {
            return mutate(s);
        }
        max_depth = 0;
        s = mutate(s);
        if (is_no_op(s)) {
            return s;
        }
        while (max_depth < block_size.threads_dimensions()) {
            s = For::make(gpu_thread_name(max_depth), 0, 0, ForType::GPUThread,
                          Partition::Never, device_api, s, GPUVectorScope::Register, -1);
            max_depth++;
        }
        return s;
    }

    Stmt visit(const Block *op) override {
        Stmt first = wrap(op->first);

        Stmt rest;
        if (op->rest.defined()) {
            rest = wrap(op->rest);
        }

        if (first.same_as(op->first) &&
            rest.same_as(op->rest)) {
            return op;
        } else {
            return Block::make(first, rest);
        }
    }

    Stmt visit(const For *op) override {
        if (op->for_type == ForType::GPUThread ||
            op->for_type == ForType::GPULane) {
            depth++;
            max_depth = std::max(max_depth, depth);
            Stmt stmt = IRMutator::visit(op);
            depth--;
            return stmt;
        } else {
            return IRMutator::visit(op);
        }
    }

public:
    NormalizeDimensionality(const ExtractBlockSize &e, DeviceAPI device_api)
        : block_size(e), device_api(device_api) {
    }
};

class ReplaceForWithIf : public IRMutator {
protected:
    using IRMutator::visit;

    const ExtractBlockSize &block_size;

    Stmt visit(const For *op) override {
        if (op->for_type == ForType::GPUThread ||
            op->for_type == ForType::GPULane) {
            int dim;
            for (dim = 0; dim < 3; dim++) {
                if (ends_with(op->name, gpu_thread_name(dim))) {
                    break;
                }
            }

            internal_assert(dim >= 0 && dim < block_size.threads_dimensions());

            Stmt body = mutate(op->body);

            Expr var = Variable::make(Int(32), gpu_thread_name(dim));
            body = substitute(op->name, var + op->min, body);

            if (can_prove(op->extent() == block_size.num_threads(dim))) {
                return body;
            } else {
                Expr cond = var <= op->max;
                return IfThenElse::make(cond, body, Stmt());
            }
        } else {
            return IRMutator::visit(op);
        }
    }

public:
    ReplaceForWithIf(const ExtractBlockSize &e)
        : block_size(e) {
    }
};

class ExtractSharedAndHeapAllocations : public IRMutator {
protected:
    using IRMutator::visit;

    struct IntInterval {
        IntInterval()
            : IntInterval(0, 0) {
        }
        IntInterval(int min, int max)
            : min(min), max(max) {
        }
        int min;
        int max;
    };

    struct SharedAllocation {
        string name;
        Type type;
        Expr size;
        IntInterval liveness;    // Start and end of the barrier stage at which this allocation is used.
        MemoryType memory_type;  // Should be GPUShared or Heap
        bool striped_over_threads;
        bool size_computed_on_host;
        // A swizzled allocation is kept as its own standalone Allocate (never
        // coalesced or type-clustered), so the codegen swizzle hook can key on
        // its name and the XOR applies to its own logical index. See SwizzleLayout.
        SwizzleLayout swizzle;
    };

    struct AllocGroup {
        AllocGroup() = default;
        AllocGroup(const SharedAllocation &alloc)
            : name(alloc.name),
              widest_type(alloc.type),
              max_size(alloc.size),
              memory_type(alloc.memory_type),
              swizzle(alloc.swizzle) {
            group.push_back(alloc);
        }

        void insert(const SharedAllocation &alloc) {
            internal_assert(alloc.memory_type == memory_type);
            // Swizzled allocations are isolated upstream (find_best_fit returns -1
            // for them and skips swizzled groups), so a group should never mix them.
            internal_assert(!alloc.swizzle.defined() && !swizzle.defined())
                << "Swizzled shared allocations must not be coalesced.\n";
            if (alloc.type.bytes() == widest_type.bytes()) {
                max_size = max(max_size, alloc.size);
            } else if (alloc.type.bytes() > widest_type.bytes()) {
                // Change units of max_size
                int size_ratio = alloc.type.bytes() / widest_type.bytes();
                max_size = max(max_size / size_ratio, alloc.size);
                widest_type = alloc.type;
            } else {
                int size_ratio = widest_type.bytes() / alloc.type.bytes();
                max_size = max(max_size, alloc.size / size_ratio);
            }
            group.push_back(alloc);
            name += "_" + alloc.name;
        }

        // Only need to check the back of the vector since we always insert
        // the most recent allocation at the back.
        bool is_free(int stage) const {
            return group.back().liveness.max < stage;
        }

        string name;
        Type widest_type;
        Expr max_size;                   // In units of the widest type
        vector<SharedAllocation> group;  // Groups of allocs that should be coalesced together
        MemoryType memory_type;          // All allocations in the group have this memory type
        SwizzleLayout swizzle;           // Non-identity only for isolated (uncoalesced) swizzled allocs
    };

public:
    vector<SharedAllocation> allocations;

protected:
    map<string, SharedAllocation *> shared;

    bool in_threads = false;

    int barrier_stage = 0;

    const DeviceAPI device_api;

    string thread_id_var_name, num_threads_var_name;

    const bool may_merge_allocs_of_different_type =
        device_api != DeviceAPI::D3D12Compute &&
        device_api != DeviceAPI::Vulkan &&
        device_api != DeviceAPI::WebGPU;

    // A loop on the host used to compute the shared memory size
    Stmt host_side_preamble;

    void precompute_allocation_size(SharedAllocation &s) {
        Expr val = Load::make(Int(32), s.name + ".shared_size", 0,
                              Buffer<>{}, Parameter{}, const_true(), ModulusRemainder{});
        Stmt update_size = Store::make(s.name + ".shared_size", max(s.size, val), 0,
                                       Parameter{}, const_true(), ModulusRemainder{});

        if (host_side_preamble.defined()) {
            host_side_preamble = Block::make(host_side_preamble, update_size);
        } else {
            host_side_preamble = update_size;
        }
        s.size_computed_on_host = true;
        s.size = Variable::make(Int(32), s.name + ".shared_size_var");
    }

    Stmt visit(const For *op) override {
        bool is_thread_loop = op->for_type == ForType::GPUThread || op->for_type == ForType::GPULane;
        ScopedValue<bool> old_in_threads(in_threads, in_threads || is_thread_loop);

        // Set aside the allocations we've found so far.
        vector<SharedAllocation> old;
        old.swap(allocations);

        // And any preamble
        Stmt old_preamble = host_side_preamble;
        host_side_preamble = Stmt();

        // Find allocations inside the loop body
        Stmt body = mutate(op->body);

        // Expand any new shared allocations found in the body using the loop bounds.
        Scope<Interval> scope;
        scope.push(op->name, Interval(op->min, op->max));
        for (SharedAllocation &s : allocations) {
            // If the size depends on the loop variable, take the max
            // over all loop iterations
            if (expr_uses_var(s.size, op->name) && !s.size_computed_on_host) {
                s.size = simplify(common_subexpression_elimination(s.size));
                // It's worth working extra hard to remove any
                // repeated dependence on the block var
                s.size = solve_expression(s.size, op->name).result;
                s.size = simplify(common_subexpression_elimination(s.size));
                switch (is_monotonic(s.size, op->name)) {
                case Monotonic::Unknown:
                    // TODO: if bounds_of_expr_in_scope becomes more
                    // powerful than is_monotonic, it might be better
                    // to call it here. That would be risky though, as
                    // it's not exact.
                    debug(1)
                        << "Shared allocation for " << s.name
                        << " has a size that is non-monotonic in the gpu block variable " << op->name
                        << ": " << s.size << "\n";
                    if (get_compiler_logger()) {
                        get_compiler_logger()->record_non_monotonic_loop_var(op->name, s.size);
                    }
                    precompute_allocation_size(s);
                    break;
                case Monotonic::Increasing:
                    s.size = substitute(op->name, op->max, s.size);
                    break;
                case Monotonic::Constant:
                    // The size expression used the variable, but we
                    // may have successfully eliminated it above, or
                    // is_monotonic might have detected that the
                    // dependence is false somehow. Just treat it as
                    // decreasing...
                case Monotonic::Decreasing:
                    s.size = substitute(op->name, op->min, s.size);
                    break;
                }
            }
            if (in_threads && op->is_parallel()) {
                // For parallel inner loops, make a separate slice per loop iteration
                s.size *= op->extent();
            }
        }

        // Add back on the allocations we set aside.
        if (!allocations.empty()) {
            allocations.insert(allocations.end(), old.begin(), old.end());
        } else {
            allocations.swap(old);
        }

        Expr new_min = mutate(op->min);
        Expr new_max = mutate(op->max);

        if (host_side_preamble.defined()) {
            string loop_name = unique_name('t');
            Expr v = Variable::make(Int(32), loop_name);
            host_side_preamble = substitute(op->name, v, host_side_preamble);
            host_side_preamble = For::make(loop_name, new_min, new_max,
                                           ForType::Serial, Partition::Never, DeviceAPI::None, host_side_preamble, GPUVectorScope::Register, -1);
            if (old_preamble.defined()) {
                host_side_preamble = Block::make(old_preamble, host_side_preamble);
            }
        } else {
            host_side_preamble = old_preamble;
        }

        return For::make(op->name, new_min, new_max,
                         op->for_type, op->partition_policy,
                         op->device_api, body, op->realization, op->warps_per_group);
    }

    Stmt visit(const Block *op) override {
        if (!in_threads && op->rest.defined()) {
            Stmt first = mutate(op->first);
            barrier_stage++;
            Stmt rest = mutate(op->rest);

            if (first.same_as(op->first) &&
                rest.same_as(op->rest)) {
                return op;
            } else {
                return Block::make(first, rest);
            }
        } else {
            return IRMutator::visit(op);
        }
    }

    Stmt visit(const IfThenElse *op) override {
        Expr condition = mutate(op->condition);
        Stmt before_preamble = host_side_preamble;
        host_side_preamble = Stmt();
        Stmt then_case = mutate(op->then_case);
        Stmt then_preamble = host_side_preamble;
        host_side_preamble = Stmt();
        Stmt else_case = mutate(op->else_case);
        Stmt else_preamble = host_side_preamble;

        if (then_preamble.defined()) {
            host_side_preamble = IfThenElse::make(condition, then_preamble, else_preamble);
        } else if (else_preamble.defined()) {
            host_side_preamble = IfThenElse::make(!condition, else_preamble);
        }
        if (before_preamble.defined() && host_side_preamble.defined()) {
            host_side_preamble = Block::make(before_preamble, host_side_preamble);
        } else if (before_preamble.defined()) {
            host_side_preamble = before_preamble;
        }
        return IfThenElse::make(condition, then_case, else_case);
    }

    int alloc_node_counter = 0;

    Stmt visit(const Allocate *op) override {
        user_assert(!op->new_expr.defined())
            << "Allocate node inside GPU kernel has custom new expression.\n"
            << "(Memoization is not supported inside GPU kernels at present.)\n";

        bool fixed_size_thread_allocation = (op->constant_allocation_size() != 0) && in_threads;

        if ((fixed_size_thread_allocation &&
             op->memory_type != MemoryType::Heap &&
             op->memory_type != MemoryType::GPUShared &&
             op->memory_type != MemoryType::GPUTexture) ||
            op->memory_type == MemoryType::Register ||
            op->memory_type == MemoryType::Stack) {
            // These allocations go in register or local memory
            return IRMutator::visit(op);
        }

        user_assert(op->memory_type == MemoryType::Auto ||
                    op->memory_type == MemoryType::GPUShared ||
                    op->memory_type == MemoryType::GPUTexture ||
                    op->memory_type == MemoryType::Heap)
            << "Allocation " << op->name << " must live in shared or heap memory, "
            << "but is scheduled to live in " << op->memory_type << " memory.\n";

        SharedAllocation alloc;
        alloc.name = op->name + "." + std::to_string(alloc_node_counter++);
        alloc.type = op->type;
        alloc.liveness = IntInterval(barrier_stage, barrier_stage);
        alloc.size = 1;
        for (const auto &extent : op->extents) {
            alloc.size *= extent;
        }
        alloc.size = simplify(alloc.size);
        alloc.memory_type = op->memory_type;
        alloc.size_computed_on_host = false;
        alloc.striped_over_threads = in_threads;
        alloc.swizzle = op->swizzle;

        if (alloc.memory_type == MemoryType::Auto) {
            if (in_threads) {
                // Dynamic allocation within the threads loop go on
                // the heap by default.
                alloc.memory_type = MemoryType::Heap;
            } else {
                // Allocations at the blocks level go in shared by
                // default.
                alloc.memory_type = MemoryType::GPUShared;
            }
        }

        // Updates the liveness by checking for all uses
        shared.emplace(op->name, &alloc);
        Stmt stmt = IRMutator::visit(op);
        op = stmt.as<Allocate>();
        internal_assert(op);

        allocations.push_back(alloc);
        shared.erase(op->name);
        return op->body;
    }

    Expr mutate_index(SharedAllocation *alloc, const Expr &index) {
        Expr idx = mutate(index);
        if (alloc->striped_over_threads) {
            idx *= Variable::make(Int(32), num_threads_var_name);
            idx += Variable::make(Int(32), thread_id_var_name);
        }
        return idx;
    }

    Expr visit(const Load *op) override {
        auto it = shared.find(op->name);
        if (it != shared.end()) {
            SharedAllocation *alloc = it->second;
            alloc->liveness.max = barrier_stage;
            Expr predicate = mutate(op->predicate);
            Expr index = mutate_index(alloc, op->index);
            return Load::make(op->type, alloc->name,
                              index, op->image, op->param, predicate, op->alignment);
        } else {
            return IRMutator::visit(op);
        }
    }

    Stmt visit(const Store *op) override {
        auto it = shared.find(op->name);
        if (it != shared.end()) {
            SharedAllocation *alloc = it->second;
            alloc->liveness.max = barrier_stage;
            Expr predicate = mutate(op->predicate);
            Expr index = mutate_index(alloc, op->index);
            Expr value = mutate(op->value);
            return Store::make(alloc->name, value, index,
                               op->param, predicate, op->alignment);
        } else {
            return IRMutator::visit(op);
        }
    }

    Stmt visit(const LetStmt *op) override {
        Expr value = mutate(op->value);

        // Set aside the allocations we've found so far.
        Stmt old_preamble = host_side_preamble;
        host_side_preamble = Stmt();
        vector<SharedAllocation> old;
        old.swap(allocations);

        Stmt body = mutate(op->body);

        // Wrap let expression for any allocations found within
        for (SharedAllocation &s : allocations) {
            if (expr_uses_var(s.size, op->name) && !s.size_computed_on_host) {
                s.size = Let::make(op->name, op->value, s.size);
                s.size = simplify(s.size);
            }
        }

        if (host_side_preamble.defined() &&
            stmt_uses_var(host_side_preamble, op->name)) {
            host_side_preamble = LetStmt::make(op->name, op->value, host_side_preamble);
        }

        if (old_preamble.defined()) {
            if (host_side_preamble.defined()) {
                host_side_preamble = Block::make(old_preamble, host_side_preamble);
            } else {
                host_side_preamble = old_preamble;
            }
        }

        // Add back on the allocations we set aside.
        if (!allocations.empty()) {
            allocations.insert(allocations.end(), old.begin(), old.end());
        } else {
            allocations.swap(old);
        }

        if (op->body.same_as(body) && value.same_as(op->value)) {
            return op;
        } else {
            return LetStmt::make(op->name, value, body);
        }
    }

    // Return index to free_spaces where 'alloc' should be coalesced. Return -1
    // if there isn't any.
    int find_best_fit(const vector<AllocGroup> &mem_allocs,
                      const vector<int> &free_spaces,
                      const SharedAllocation &alloc, int stage) {
        int free_idx = -1;

        // Swizzled allocations are never coalesced -- they must reach codegen as
        // their own named Allocate so the swizzle hook can key on the name.
        if (alloc.swizzle.defined()) {
            return -1;
        }

        Expr alloc_size = simplify(alloc.size);

        // We prefer to coalesce dynamic-sized allocation with a dynamic-sized one and
        // constant-sized alloc with a constant-sized one. If we can't find any free
        // space with a matching type, we pick the most-recently freed space of the
        // other type (e.g. pick constant-sized free space for a dynamic-sized allocation
        // and vice versa). We prefer the most-recently freed space as stages that are
        // close together usually have relatively similar allocation size. For
        // constant-sized allocation, we prioritize free space which size differs
        // the least with 'alloc' (can be smaller or larger; it does not really
        // matter since we take the max of the two as the new size).

        if (!is_const(alloc_size)) {  // dynamic-sized alloc
            for (int i = free_spaces.size() - 1; i >= 0; --i) {
                internal_assert(free_spaces[i] >= 0 && free_spaces[i] < (int)mem_allocs.size());
                internal_assert(mem_allocs[free_spaces[i]].is_free(stage));

                if (mem_allocs[free_spaces[i]].memory_type != alloc.memory_type) {
                    continue;
                }

                // Never coalesce anything into a swizzled allocation's space.
                if (mem_allocs[free_spaces[i]].swizzle.defined()) {
                    continue;
                }

                if (!may_merge_allocs_of_different_type &&
                    mem_allocs[free_spaces[i]].group[0].type != alloc.type) {
                    continue;
                }

                if (!is_const(mem_allocs[free_spaces[i]].max_size)) {
                    return i;
                } else if (free_idx == -1) {
                    free_idx = i;
                }
            }
        } else {  // constant-sized alloc
            int64_t diff = -1;
            for (int i = free_spaces.size() - 1; i >= 0; --i) {
                internal_assert(free_spaces[i] >= 0 && free_spaces[i] < (int)mem_allocs.size());
                internal_assert(mem_allocs[free_spaces[i]].is_free(stage));

                if (mem_allocs[free_spaces[i]].memory_type != alloc.memory_type) {
                    continue;
                }

                // Never coalesce anything into a swizzled allocation's space.
                if (mem_allocs[free_spaces[i]].swizzle.defined()) {
                    continue;
                }

                if (!may_merge_allocs_of_different_type &&
                    mem_allocs[free_spaces[i]].group[0].type != alloc.type) {
                    continue;
                }

                if (is_const(mem_allocs[free_spaces[i]].max_size)) {
                    const auto &candidate_group = mem_allocs[free_spaces[i]];
                    Expr size = alloc_size * alloc.type.bytes();
                    Expr dist = candidate_group.max_size * candidate_group.widest_type.bytes() - size;
                    auto current_diff = as_const_int(simplify(dist));
                    internal_assert(current_diff);
                    int64_t abs_diff = std::abs(*current_diff);
                    if ((free_idx == -1) || (abs_diff < diff)) {
                        diff = abs_diff;
                        free_idx = i;
                    }
                } else if (free_idx == -1) {
                    free_idx = i;
                }
            }
        }

        return free_idx;
    }

    // Given some allocations, return a vector of allocation group where each group
    // consists of a number of allocations which should be coalesced together
    // in the shared memory.
    vector<AllocGroup> allocate_funcs(vector<SharedAllocation> &allocations) {
        // Sort based on the ascending order of the min liveness stage,
        // then sort based on the ascending order of the max liveness stage.
        sort(allocations.begin(), allocations.end(),
             [](const SharedAllocation &lhs, const SharedAllocation &rhs) {
                 if (lhs.liveness.min < rhs.liveness.min) {
                     return true;
                 } else if (lhs.liveness.min == rhs.liveness.min) {
                     return lhs.liveness.max < rhs.liveness.max;
                 }
                 return false;
             });

        vector<AllocGroup> mem_allocs;
        vector<int> free_spaces;  // Contains index to free spaces in mem_allocs
        int start_idx = 0;

        for (int stage = 0; stage <= barrier_stage; ++stage) {
            for (int i = start_idx; i < (int)allocations.size(); ++i) {
                if (allocations[i].liveness.min > stage) {
                    break;
                } else if (allocations[i].liveness.min == stage) {  // Allocate
                    int free_idx = find_best_fit(mem_allocs, free_spaces, allocations[i], stage);
                    if (free_idx != -1) {
                        mem_allocs[free_spaces[free_idx]].insert(allocations[i]);
                        free_spaces.erase(free_spaces.begin() + free_idx);
                    } else {
                        mem_allocs.emplace_back(allocations[i]);
                    }
                } else if (allocations[i].liveness.max == stage - 1) {  // Free
                    int free_idx = -1;
                    for (int j = 0; j < (int)mem_allocs.size(); ++j) {  // Find the index of the space to free
                        if (mem_allocs[j].group.back().name == allocations[i].name) {
                            free_idx = j;
                            break;
                        }
                    }
                    internal_assert(free_idx >= 0 && free_idx < (int)mem_allocs.size());
                    free_spaces.push_back(free_idx);
                    start_idx = i + 1;
                }
            }
        }

        return mem_allocs;
    }

    Expr get_block_id(const ExtractBlockSize &bs) const {
        Expr block_id = 0;
        for (int d = bs.blocks_dimensions() - 1; d >= 0; d--) {
            block_id *= bs.num_blocks(d);
            block_id += bs.block_var(d);
        }
        return block_id;
    }

    Expr max_over_blocks(const Expr &e, const ExtractBlockSize &bs) const {
        Scope<Interval> scope;
        for (int d = 0; d < bs.blocks_dimensions(); d++) {
            scope.push(bs.block_var(d).as<Variable>()->name,
                       Interval(0, bs.num_blocks(d) - 1));
        }
        Interval in = bounds_of_expr_in_scope(simplify(e), scope);
        if (in.has_upper_bound()) {
            return in.max;
        } else {
            return Expr();
        }
    }

    struct GlobalAllocation {
        string name;
        Expr size;
        Type type;
    };
    vector<GlobalAllocation> global_allocations;

public:
    Stmt rewrap_block(Stmt s, const ExtractBlockSize &bs) {

        // Shared buffers that are the destination of a TMA bulk load (cp.async.bulk.tensor)
        // need a 128-byte-aligned base; round their packed offset up below. Empty unless the
        // TMA recognizer fired (HL_WG_TMA), so this is NFC for every other path.
        std::set<std::string> tma_targets;
        {
            class FindTmaDsts : public IRVisitor {
                using IRVisitor::visit;
                void visit(const Call *op) override {
                    if (op->is_intrinsic() && op->name == "tma_load_2d" && !op->args.empty()) {
                        if (const Load *l = op->args[0].as<Load>()) {
                            names.insert(l->name);
                        }
                    }
                    IRVisitor::visit(op);
                }

            public:
                std::set<std::string> names;
            } f;
            s.accept(&f);
            tma_targets = std::move(f.names);
        }

        // Combine the allocations into groups that have disjoint
        // lifetimes, and then cluster the groups according to which
        // ones can share a single allocation. For cuda, opencl, and
        // similar we get one big combined allocation per memory
        // type. For vulkan and direct3d, we also separate by
        // element type.
        // Key is (memory type, element type, discriminator). The discriminator is
        // empty for normal allocations (so they cluster together as before) and the
        // allocation's unique name for swizzled allocations (so each lands in its
        // own cluster -> its own named Allocate, which the codegen swizzle hook keys on).
        map<std::tuple<MemoryType, Type, string>, vector<AllocGroup>> clustered_allocs;

        {
            vector<AllocGroup> mem_allocs = allocate_funcs(allocations);

            // Every allocation must belong to one group
            internal_assert(allocations.size() >= mem_allocs.size());

            // Sort the allocations by the max size in bytes of the primitive
            // types in the group. Because the type sizes are then decreasing powers of
            // two, doing this guarantees that all allocations are aligned
            // to then element type as long as the original one is aligned
            // to the widest type.
            sort(mem_allocs.begin(), mem_allocs.end(),
                 [](const AllocGroup &lhs, const AllocGroup &rhs) {
                     return lhs.widest_type.bytes() > rhs.widest_type.bytes();
                 });

            for (const auto &alloc : mem_allocs) {
                // Swizzled allocations are kept as their own (uncoalesced) GROUP, but allocations
                // with the SAME swizzle share one CLUSTER (disc keyed on the swizzle params, not the
                // name): the existing per-group cumulative offsets then give them distinct,
                // non-overlapping bases (two swizzled operands As/Bs no longer collide at offset 0),
                // and the cluster keeps its element type (not the UInt8 merge) so the swizzle-hook
                // params stay in element units. The group-offset alignment below rounds each group to
                // the swizzle super-period so swizzle(off+i) == off+swizzle(i) stays phase-correct.
                bool swz = alloc.swizzle.defined();
                Type t = (swz || !may_merge_allocs_of_different_type) ? alloc.widest_type : UInt(8);
                string disc = swz ? ("swz_" + std::to_string(alloc.swizzle.bits) + "_" +
                                     std::to_string(alloc.swizzle.base) + "_" +
                                     std::to_string(alloc.swizzle.shift))
                                  : string();
                std::tuple<MemoryType, Type, string> key{alloc.memory_type, t, disc};
                clustered_allocs[key].push_back(alloc);
            }
        }

        for (auto &p : clustered_allocs) {
            vector<AllocGroup> &cluster = p.second;
            // Heap or shared?
            MemoryType memory_type = std::get<0>(p.first);
            // Type of the combined Allocate node
            Type alloc_type = std::get<1>(p.first);

            // Figure out a name for the cluster, the total size of
            // the cluster (in terms of the alloc_type), and the
            // widest type in the cluster (which may be wider than the
            // alloc_type).
            string name;
            Expr total_size = 0;
            Type widest_type;
            int number_of_allocs = 0;
            // Swizzled allocs are isolated into their own singleton cluster above.
            SwizzleLayout cluster_swizzle;
            for (const auto &alloc : cluster) {
                number_of_allocs += alloc.group.size();
                if (alloc.swizzle.defined()) {
                    cluster_swizzle = alloc.swizzle;
                }
            }
            for (const auto &alloc : cluster) {
                if (name.empty()) {
                    widest_type = alloc.widest_type;
                    if (number_of_allocs > 1) {
                        name = "allocgroup__" + alloc.name;
                    } else {
                        name = alloc.name;
                    }
                } else {
                    if (alloc.widest_type.bytes() > widest_type.bytes()) {
                        widest_type = alloc.widest_type;
                    }
                    name += "__" + alloc.name;
                }
                int ratio = alloc.widest_type.bytes() / alloc_type.bytes();
                internal_assert(ratio != 0)
                    << "alloc_type should have been at most as wide as the widest type in group\n";
                total_size += alloc.max_size * ratio;
            }

            // Upgrade the alloc type to the widest type found, and
            // downgrade total_size accordingly.
            int ratio = widest_type.bytes() / alloc_type.bytes();
            internal_assert(ratio != 0)
                << "alloc_type should have been at most as wide as the widest type in cluster\n";
            if (ratio != 1) {
                total_size += ratio - 1;
                total_size /= ratio;
            }
            alloc_type = widest_type;

            // Remove any dependence on the block vars by taking a max
            {
                Expr size = max_over_blocks(total_size, bs);
                internal_assert(size.defined())
                    << memory_type
                    << " memory used by GPU kernel varies with the block index in an unbounded way: "
                    << total_size << "\n";
                total_size = size;
            }

            // A swizzle's XOR can push an index up to (period - granule) higher, so
            // the allocation must be padded up to a whole number of swizzle periods
            // (1 << (base + bits) elements of alloc_type) or swizzle(i) could go OOB.
            if (cluster_swizzle.defined()) {
                int period = 1 << (cluster_swizzle.base + cluster_swizzle.bits);
                total_size = simplify(((total_size + period - 1) / period) * period);
            }

            const string total_size_name = name + ".size";
            Expr total_size_var = Variable::make(Int(32), total_size_name);

            // Make the allocation
            if (memory_type == MemoryType::Heap) {
                global_allocations.push_back(GlobalAllocation{name, total_size, alloc_type});
            } else {
                s = Allocate::make(name, alloc_type, memory_type,
                                   {total_size_var}, const_true(), s,
                                   Expr(), std::string(), 0, cluster_swizzle);
            }

            // Define a group offset for each group in the
            // cluster. The group offsets are in elements of
            // widest_type across the entire cluster. Using that,
            // define an individual offset for each allocation in the
            // group, using units of that allocation's type.
            for (int i = (int)(cluster.size()) - 1; i >= 0; i--) {
                Expr group_offset = Variable::make(Int(32), name + "." + std::to_string(i) + ".offset");

                for (const SharedAllocation &alloc : cluster[i].group) {
                    // Change units, as described above.
                    Expr offset = group_offset;
                    internal_assert(alloc.type.bytes() <= widest_type.bytes());
                    if (alloc.type.bytes() < widest_type.bytes()) {
                        offset *= (widest_type.bytes() / alloc.type.bytes());
                    }
                    offset = simplify(offset);

                    // Rewrite all loads and stores to point to the allocation
                    // cluster they belong to with the appropriate offset into it.
                    class RewriteGroupAccess : public IRMutator {
                        using IRMutator::visit;
                        Expr visit(const Load *op) override {
                            if (op->name == alloc_name) {
                                return Load::make(op->type, cluster_name, mutate(op->index) + offset,
                                                  op->image, op->param, mutate(op->predicate),
                                                  op->alignment);
                            } else {
                                return IRMutator::visit(op);
                            }
                        }

                        Stmt visit(const Store *op) override {
                            if (op->name == alloc_name) {
                                return Store::make(cluster_name, mutate(op->value), mutate(op->index) + offset,
                                                   op->param, mutate(op->predicate), op->alignment);
                            } else {
                                return IRMutator::visit(op);
                            }
                        }
                        const string &alloc_name;
                        const string &cluster_name;
                        const Expr &offset;

                    public:
                        RewriteGroupAccess(const string &alloc_name,
                                           const string &cluster_name,
                                           const Expr &offset)
                            : alloc_name(alloc_name), cluster_name(cluster_name), offset(offset) {
                        }
                    } rewriter{alloc.name, name, offset};
                    s = rewriter(s);
                }

                // Define the group offset in terms of the previous group in the cluster
                Expr offset;
                if (i > 0) {
                    // Build off the last offset
                    offset = Variable::make(Int(32), name + "." + std::to_string(i - 1) + ".offset");
                    int ratio = (widest_type.bytes() / cluster[i - 1].widest_type.bytes());
                    internal_assert(ratio != 0);
                    offset += simplify((cluster[i - 1].max_size + ratio - 1) / ratio);
                } else {
                    if (memory_type == MemoryType::Heap) {
                        // One slice of a larger global allocation
                        offset = get_block_id(bs) * total_size_var;
                    } else {
                        // Base address for shared memory is zero
                        offset = 0;
                    }
                }

                // A TMA bulk-load destination (cp.async.bulk.tensor) OR a swizzled wgmma operand
                // must start on a 128-byte boundary: TMA writes / the wgmma descriptor's swizzle
                // mode reads relative to the tile base, and base_offset=0 assumes swizzle-atom
                // alignment. Round this group's offset up; the next group packs after the gap.
                bool is_tma = false;
                for (const SharedAllocation &a : cluster[i].group) {
                    if (tma_targets.count(a.name)) {
                        is_tma = true;
                    }
                }
                if (cluster[i].swizzle.defined() || is_tma) {
                    // Swizzled groups align to the swizzle SUPER-period (2^(shift+bits) elements of
                    // the swizzle's own type): only a base that is a multiple of it leaves the XOR
                    // field of every access undisturbed, so swizzle(off+i) == off + swizzle(i). A
                    // plain TMA destination just needs 128B. (offset is in widest_type units.)
                    int align_units;
                    if (cluster[i].swizzle.defined()) {
                        const SwizzleLayout &s = cluster[i].swizzle;
                        align_units = 1 << (s.shift + s.bits);
                    } else {
                        align_units = std::max(1, 128 / widest_type.bytes());
                    }
                    offset = simplify(((offset + (align_units - 1)) / align_units) * align_units);
                }

                s = LetStmt::make(group_offset.as<Variable>()->name, simplify(offset), s);
            }
            s = LetStmt::make(total_size_name, total_size, s);
        }

        // Resolve thread_id and threads_per_block variables, uses of
        // which were injected above if any allocation was striped
        // over the threads.
        Expr thread_id = 0, num_threads = 1;
        for (int d = bs.threads_dimensions() - 1; d >= 0; d--) {
            num_threads *= bs.num_threads(d);
            thread_id *= bs.num_threads(d);
            thread_id += bs.thread_var(d);
        }
        if (stmt_uses_var(s, thread_id_var_name)) {
            s = LetStmt::make(thread_id_var_name, thread_id, s);
        }
        if (stmt_uses_var(s, num_threads_var_name)) {
            s = LetStmt::make(num_threads_var_name, num_threads, s);
        }

        return s;
    }

    Stmt rewrap_kernel_launch(Stmt s, const ExtractBlockSize &bs, DeviceAPI device_api) {

        for (const auto &alloc : global_allocations) {
            Expr total_size = alloc.size;

            Expr device_interface = make_device_interface_call(device_api);
            string buffer_name = alloc.name + ".buffer";
            Expr buffer_var = Variable::make(type_of<halide_buffer_t *>(), buffer_name);

            BufferBuilder builder;
            builder.mins.emplace_back(0);
            builder.extents.push_back(total_size);
            builder.strides.emplace_back(1);
            builder.type = alloc.type;
            builder.dimensions = 1 + bs.blocks_dimensions();

            for (int d = 0; d < bs.blocks_dimensions(); d++) {
                Expr next_stride =
                    builder.strides.back() *
                    builder.extents.back();
                builder.strides.push_back(next_stride);
                builder.extents.emplace_back(bs.num_blocks(d));
            }
            Expr buffer = builder.build();
            Expr allocate_heap_call = Call::make(Int(32), "halide_device_malloc",
                                                 {buffer_var, device_interface}, Call::Extern);
            string allocate_heap_result_var_name = unique_name('t');
            Expr allocate_heap_result_var = Variable::make(Int(32), allocate_heap_result_var_name);
            Stmt check_allocated =
                AssertStmt::make(allocate_heap_result_var == 0, allocate_heap_result_var);
            Expr device_field = Call::make(Handle(), Call::buffer_get_device, {buffer_var}, Call::Extern);
            s = LetStmt::make(alloc.name, device_field, s);
            s = Block::make(check_allocated, s);
            s = LetStmt::make(allocate_heap_result_var_name, allocate_heap_call, s);
            s = Allocate::make(buffer_name, alloc.type,
                               MemoryType::Auto, {}, const_true(), s,
                               buffer, "halide_device_free_as_destructor");
        }

        s = compute_shared_memory_sizes_on_host(s);

        return s;
    }

    Stmt compute_shared_memory_sizes_on_host(Stmt result) {
        if (!host_side_preamble.defined()) {
            return result;
        }

        // Make all the let stmts that define the size vars
        for (auto &alloc : allocations) {
            if (alloc.size_computed_on_host) {
                string alloc_name = alloc.name + ".shared_size";
                string var_name = alloc.name + ".shared_size_var";
                Expr val = Load::make(Int(32), alloc_name, 0,
                                      Buffer<>{}, Parameter{}, const_true(), ModulusRemainder{});
                result = LetStmt::make(var_name, val, result);
                alloc.size = Variable::make(Int(32), var_name);
            }
        }

        // Prefix the preamble
        result = Block::make(host_side_preamble, result);

        // Wrap the preamble in all the allocation nodes
        for (auto &alloc : allocations) {
            if (alloc.size_computed_on_host) {
                string alloc_name = alloc.name + ".shared_size";
                Stmt init = Store::make(alloc_name, 0, 0,
                                        Parameter{}, const_true(), ModulusRemainder{});
                result = Block::make(init, result);
                result = Allocate::make(alloc_name, Int(32), MemoryType::Stack, {1}, const_true(), result);
            }
        }

        return result;
    }

    ExtractSharedAndHeapAllocations(DeviceAPI d)
        : device_api(d),
          thread_id_var_name(unique_name('t')),
          num_threads_var_name(unique_name('t')) {
    }
};  // namespace Internal

// Pull out any allocate node outside of the innermost thread
// block. Should only be run after shared allocations have already
// been extracted.
class ExtractRegisterAllocations : public IRMutator {
protected:
    using IRMutator::visit;

    struct RegisterAllocation {
        string name;
        string loop_var;  // The nearest enclosing loop over threads. Empty if it's at block level.
        Type type;
        Expr size;
        MemoryType memory_type;  // Should be Auto, Stack, or Register
    };

    bool in_lane_loop = false;

    Stmt visit(const For *op) override {
        ScopedValue<string> old_loop_var(loop_var);

        if (op->for_type == ForType::GPULane) {
            loop_var = op->name;
            internal_assert(!in_lane_loop);
            ScopedValue<bool> old_in_lane_loop(in_lane_loop, true);
            has_lane_loop = true;
            return IRMutator::visit(op);
        } else {
            if (op->for_type == ForType::GPUThread) {
                has_thread_loop = true;
                loop_var = op->name;
            }

            // Hoisting an allocation out of a vectorized for loop
            // would break here. We should already have hoisted
            // vectorized allocations.
            internal_assert(op->for_type != ForType::Vectorized);

            // Set aside the allocations we've found so far.
            vector<RegisterAllocation> old;
            old.swap(allocations);

            // Find allocations inside the loop body
            Stmt body = mutate(op->body);

            // Expand any new register allocations found in the body using the loop bounds.
            Scope<Interval> scope;
            scope.push(op->name, Interval(op->min, op->max));

            // Expand the inner allocations using the loop bounds.
            for (RegisterAllocation &s : allocations) {
                if (expr_uses_var(s.size, op->name)) {
                    s.size = bounds_of_expr_in_scope(s.size, scope).max;
                }
            }

            // Add back on the allocations we set aside.
            if (!allocations.empty()) {
                allocations.insert(allocations.end(), old.begin(), old.end());
            } else {
                allocations.swap(old);
            }

            return For::make(op->name, mutate(op->min), mutate(op->max), op->for_type, op->partition_policy, op->device_api, body, op->realization, op->warps_per_group);
        }
    }

    int alloc_node_counter = 0;
    Scope<string> alloc_renaming;

    Stmt visit(const Allocate *op) override {
        if (in_lane_loop) {
            return IRMutator::visit(op);
        }

        user_assert(op->memory_type == MemoryType::Stack ||
                    op->memory_type == MemoryType::Register ||
                    op->memory_type == MemoryType::Heap ||
                    op->memory_type == MemoryType::Auto)
            << "Allocation " << op->name << " is scheduled inside a loop over GPU threads, so "
            << "it must live in stack memory, heap memory, or registers. "
            << "Shared allocations at this loop level are not yet supported.\n";

        ScopedBinding<int> p(register_allocations, op->name, 0);

        RegisterAllocation alloc;
        alloc.name = op->name + "." + std::to_string(alloc_node_counter++);
        alloc.type = op->type;
        alloc.size = 1;
        alloc.loop_var = loop_var;
        for (const auto &extent : op->extents) {
            alloc.size *= extent;
        }
        alloc.size = simplify(mutate(alloc.size));
        alloc.memory_type = op->memory_type;

        allocations.push_back(alloc);
        {
            ScopedBinding<string> bind(alloc_renaming, op->name, alloc.name);
            return mutate(op->body);
        }
    }

    Expr visit(const Load *op) override {
        const string *new_name = alloc_renaming.find(op->name);
        if (!new_name) {
            new_name = &(op->name);
        }
        return Load::make(op->type, *new_name, mutate(op->index),
                          op->image, op->param, mutate(op->predicate),
                          op->alignment);
    }

    Stmt visit(const Store *op) override {
        const string *new_name = alloc_renaming.find(op->name);
        if (!new_name) {
            new_name = &(op->name);
        }
        return Store::make(*new_name, mutate(op->value), mutate(op->index),
                           op->param, mutate(op->predicate), op->alignment);
    }

    template<typename LetOrLetStmt>
    auto visit_let(const LetOrLetStmt *op) -> decltype(op->body) {
        auto body = op->body;

        body = mutate(op->body);
        Expr value = mutate(op->value);

        for (RegisterAllocation &s : allocations) {
            if (expr_uses_var(s.size, op->name)) {
                s.size = simplify(Let::make(op->name, value, s.size));
            }
        }

        if (op->body.same_as(body) && op->value.same_as(value)) {
            return op;
        } else {
            return LetOrLetStmt::make(op->name, value, body);
        }
    }

    Expr visit(const Let *op) override {
        return visit_let(op);
    }

    Stmt visit(const LetStmt *op) override {
        return visit_let(op);
    }

    Scope<int> register_allocations;
    string loop_var;

public:
    vector<RegisterAllocation> allocations;

    Stmt rewrap(Stmt body, const string &loop_var) {
        for (RegisterAllocation &alloc : allocations) {
            if ((!loop_var.empty() && ends_with(alloc.loop_var, loop_var)) ||
                (loop_var.empty() && alloc.loop_var.empty())) {
                body = Allocate::make(alloc.name, alloc.type, alloc.memory_type, {alloc.size}, const_true(), body);
            }
        }
        return body;
    }

    bool has_lane_loop = false;
    bool has_thread_loop = false;
};

class InjectThreadBarriers : public IRMutator {
protected:
    bool in_threads = false, injected_barrier;

    using IRMutator::visit;

    const ExtractSharedAndHeapAllocations &block_allocs;
    const ExtractRegisterAllocations &register_allocs;

    // The execution-mapping model (M1): barrier scope = join(active(producer),
    // active(consumer)) instead of the old unconditional Block. `exec` tracks the
    // current active set as this mutator descends (enter_for / enter_if / pop). See
    // research/exec_mapping_model.md, fusegpu_rearch_plan.md C.0/M2.
    ExecMap &exec;

    std::set<std::string> shared_stores;
    std::set<std::string> device_stores;
    std::set<std::string> shared_loads;
    std::set<std::string> device_loads;
    // Redundant-barrier elimination (Triton membar principle: an async-wait is a sync point). A TMA
    // shared store whose completion mbarrier is try_wait'd (by all threads) before the consumer load
    // is ALREADY ordered store->load, so no block barrier is needed for it. Track each TMA shared
    // dst -> its completion mbar buffer, and the set of mbar buffers that are try_wait'd.
    std::map<std::string, std::string> tma_dst_mbar;
    std::set<std::string> mbar_waited;
    // The buffer a (possibly nested) Load addresses, or "" if none.
    static std::string load_buffer(const Expr &e) {
        class NameOf : public IRVisitor {
            using IRVisitor::visit;
            void visit(const Load *l) override { if (name.empty()) name = l->name; IRVisitor::visit(l); }
        public: std::string name;
        } n;
        e.accept(&n);
        return n.name;
    }
    // Active set of the producing store(s) / consuming load(s) per name (joined over
    // all occurrences), captured alongside the name sets above.
    std::map<std::string, ActiveSet> store_active;
    std::map<std::string, ActiveSet> load_active;

    void record_active(std::map<std::string, ActiveSet> &m, const std::string &name) {
        auto it = m.find(name);
        m[name] = (it == m.end()) ? exec.current() : ExecMap::join(it->second, exec.current());
    }

    MemoryType memory_type_for_name(const std::string &name) {
        for (const auto &x : register_allocs.allocations) {
            if (x.name == name) {
                return x.memory_type;
            }
        }
        for (const auto &x : block_allocs.allocations) {
            if (x.name == name) {
                return x.memory_type;
            }
        }
        // Not allocated here, so must assume it's input/output
        // of shader
        return MemoryType::Auto;
    }

    Stmt make_barrier(int mask) {
        // Emit a target-independent Block-scope synchronization requirement (the
        // produce/consume dependency spans the whole CTA here). lower_sync_requirements
        // picks the mechanism — today a whole-CTA gpu_thread_barrier, byte-identical to
        // emitting it directly. See research/gpu_sync_model.md.
        return Evaluate::make(Call::make(Int(32), Call::sync_requirement,
                                         {IntImm::make(Int(32), (int)SyncScope::Block),
                                          IntImm::make(Int(32), mask)},
                                         Call::Intrinsic));
    }

    Stmt visit(const For *op) override {
        ScopedValue<bool> old_in_threads(in_threads,
                                         (in_threads ||
                                          op->for_type == ForType::GPUThread ||
                                          op->for_type == ForType::GPULane));

        ScopedValue<bool> old_injected_barrier(injected_barrier, false);

        // Track the active set: descending into a GPU thread/lane/warp-group loop
        // narrows which lanes execute the body (full extent => no change).
        exec.enter_for(op);
        Stmt result;
        if (!is_parallel(op->for_type)) {
            Stmt body = mutate(op->body);
            // Serial for loops at the block level with internal
            // synchronization also need synchronization after each
            // loop iteration.
            if (!in_threads && injected_barrier) {
                // Any memory access fences should be handled by the
                // synchronizations within the block
                body = Block::make(body, make_barrier(0));
            }
            result = For::make(op->name, op->min, op->max,
                               op->for_type, op->partition_policy, op->device_api, body, op->realization, op->warps_per_group);
        } else {
            result = IRMutator::visit(op);
        }
        exec.pop();
        return result;
    }

    Stmt visit(const IfThenElse *op) override {
        Expr condition = mutate(op->condition);
        // The then-branch executes only on the lanes satisfying the guard.
        exec.enter_if(op->condition);
        Stmt then_case = mutate(op->then_case);
        exec.pop();
        // The else-branch executes on the COMPLEMENT lanes: narrow by the negated guard (the
        // warp-spec fork emits `if tid<32 {As} else if tid<64 {Bs} ...`, so Bs lives in the else
        // and must narrow to [32,64) -- without this its elected lane / scope would be wrong).
        exec.enter_if(simplify(!op->condition));
        Stmt else_case = op->else_case.defined() ? mutate(op->else_case) : Stmt();
        exec.pop();
        if (condition.same_as(op->condition) &&
            then_case.same_as(op->then_case) &&
            else_case.same_as(op->else_case)) {
            return op;
        }
        return IfThenElse::make(condition, then_case, else_case);
    }

    Stmt visit(const Store *op) override {
        debug(4) << "Encountered store to " << op->name << "\n";
        auto mem_type = memory_type_for_name(op->name);
        switch (mem_type) {
        case MemoryType::GPUShared:
            debug(4) << "   memory type is shared\n";
            shared_stores.insert(op->name);
            record_active(store_active, op->name);
            break;
        case MemoryType::Auto:
        case MemoryType::Heap:
        case MemoryType::GPUTexture:
            debug(4) << "   memory type is heap or auto\n";
            device_stores.insert(op->name);
            record_active(store_active, op->name);
            break;
        case MemoryType::Stack:
        case MemoryType::Register:
        case MemoryType::LockedCache:
        case MemoryType::VTCM:
        case MemoryType::AMXTile:
            break;
        }

        return IRMutator::visit(op);
    }

    Expr visit(const Load *op) override {
        debug(4) << "Encountered load from " << op->name << "\n";
        auto mem_type = memory_type_for_name(op->name);
        switch (mem_type) {
        case MemoryType::GPUShared:
            debug(4) << "   memory type is shared\n";
            shared_loads.insert(op->name);
            record_active(load_active, op->name);
            break;
        case MemoryType::Auto:
        case MemoryType::Heap:
        case MemoryType::GPUTexture:
            debug(4) << "   memory type is heap or auto\n";
            device_loads.insert(op->name);
            record_active(load_active, op->name);
            break;
        case MemoryType::Stack:
        case MemoryType::Register:
        case MemoryType::LockedCache:
        case MemoryType::VTCM:
        case MemoryType::AMXTile:
            break;
        }

        return IRMutator::visit(op);
    }

    // Append the model's ELECTED LANE (the thread codegen should single-thread-issue this op on) as
    // the op's last arg, when it isn't already there (base_argc = the op's normal arg count). Replaces
    // codegen's hardcoded global tid==0: a sub-region (warp-spec) producer elects its OWN first lane.
    Expr with_elected_lane(const Call *op, size_t base_argc) {
        if (op->args.size() != base_argc) {
            return op;  // already carries the elected lane (idempotent)
        }
        std::vector<Expr> args = op->args;
        args.push_back(exec.elected_lane(exec.current()));
        return Call::make(op->type, op->name, args, op->call_type);
    }

    Expr visit(const Call *op) override {
        // F3: the mbarrier_init intrinsic WRITES the mbarrier shared state (via inline asm); its
        // args are Load carriers (base_ref) that would otherwise register as shared READS. Register
        // the mbar buffer(s) as shared STORES instead, so the produce->consume scan inserts a
        // correctly-ordered all-threads barrier before the arrive/try_wait reads (CUTLASS's prologue
        // __syncthreads, emitted by Halide's own mechanism). Don't recurse -> don't double-count the
        // carriers as loads.
        if (op->is_intrinsic() && op->name == "mbarrier_init") {
            for (const Expr &a : op->args) {
                if (const Load *l = a.as<Load>()) {
                    if (memory_type_for_name(l->name) == MemoryType::GPUShared) {
                        shared_stores.insert(l->name);
                        record_active(store_active, l->name);
                    }
                }
            }
            return op;
        }
        if (op->is_intrinsic() && op->name == "tma_load_2d") {
            // tma_load_2d WRITES its shared destination (arg 0, a Load carrier) -- the bulk copy
            // lands the tile in shared. Register it as a shared STORE so the produce->consume scan
            // GENERATES the CTA broadcast barrier (and lets the mbarrier args register as reads).
            // Mirrors the mbarrier_init case. See fusegpu_rearch_plan.md C.4b/M4.
            if (const Load *dst = op->args[0].as<Load>()) {
                if (memory_type_for_name(dst->name) == MemoryType::GPUShared) {
                    shared_stores.insert(dst->name);
                    record_active(store_active, dst->name);
                    // Remember which mbarrier completes this TMA dst (arg 4). If the consumer
                    // try_waits that mbarrier (all threads) the store->load edge is already CTA-
                    // ordered and the block barrier is redundant -- see the consider() skip below.
                    if (op->args.size() > 4) {
                        std::string m = load_buffer(op->args[4]);
                        if (!m.empty()) {
                            tma_dst_mbar[dst->name] = m;
                        }
                    }
                }
            }
            return with_elected_lane(op, 5);
        }
        if (op->is_intrinsic() && op->name == "mbarrier_arrive_expect_tx") {
            return with_elected_lane(op, 2);
        }
        if (op->is_intrinsic() && op->name == "mbarrier_try_wait") {
            // An all-threads mbarrier wait: record the mbar buffer as a CTA-wide sync point so the
            // RAW it covers (TMA store -> this consumer's load) needs no extra block barrier.
            if (!op->args.empty()) {
                std::string m = load_buffer(op->args[0]);
                if (!m.empty()) {
                    mbar_waited.insert(m);
                }
            }
            return IRMutator::visit(op);
        }
        return IRMutator::visit(op);
    }

    Stmt visit(const Block *op) override {
        if (!in_threads && op->rest.defined()) {
            // First, we record which loads from shared/device memory occur
            // in the rest block
            Stmt rest = mutate(op->rest);

            // Capture REST's stores (for WAR/WAW vs first) before they're cleared, and snapshot the
            // accumulated loads so we can isolate FIRST's own loads. Used only by membar mode below.
            std::set<std::string> rest_shared_stores = shared_stores;
            std::set<std::string> rest_device_stores = device_stores;
            std::set<std::string> shared_loads_before = shared_loads;
            std::set<std::string> device_loads_before = device_loads;

            // Now, record which stores occur in the first stmt
            // of this block
            shared_stores.clear();
            device_stores.clear();
            store_active.clear();
            Stmt first = mutate(op->first);

            // If there are any loads in the rest part that load from something stored in
            // first, insert the appropriate fence type AND accumulate the combined active
            // set of the matched producer stores and consumer loads (M2).
            int mask = 0;
            ActiveSet combined;
            bool any_match = false;
            const bool membar = get_env_variable("HL_WG_MEMBAR") == "1";
            auto consider = [&](const std::set<std::string> &stores,
                                const std::set<std::string> &loads, int fence) {
                for (const auto &st : stores) {
                    if (loads.count(st)) {
                        // Redundant-barrier elimination (membar mode only, so the default path is
                        // byte-identical NFC): if `st` is a TMA shared dst whose completion mbarrier is
                        // try_wait'd (all threads) by the consumer, that wait is the CTA-wide ordering
                        // point (Hopper mbarrier.try_wait.parity) and the block barrier is redundant.
                        auto mi = tma_dst_mbar.find(st);
                        if (membar && mi != tma_dst_mbar.end() && mbar_waited.count(mi->second)) {
                            continue;
                        }
                        mask |= fence;
                        auto si = store_active.find(st);
                        auto li = load_active.find(st);
                        ActiveSet s = (si != store_active.end()) ? si->second : ActiveSet::whole_block();
                        ActiveSet l = (li != load_active.end()) ? li->second : ActiveSet::whole_block();
                        ActiveSet pair = ExecMap::join(s, l);
                        combined = any_match ? ExecMap::join(combined, pair) : pair;
                        any_match = true;
                    }
                }
            };
            consider(shared_stores, shared_loads, CodeGen_GPU_Dev::MemoryFenceType::Shared);
            consider(device_stores, device_loads, CodeGen_GPU_Dev::MemoryFenceType::Device);

            // Triton-membar mode (opt-in via HL_WG_MEMBAR): emit a barrier ONLY on a real shared
            // hazard not already ordered by an mbarrier sync point. consider() above is RAW (first
            // store -> rest load), minus mbarrier-covered TMA edges. We must ALSO keep WAR/WAW edges
            // (first load/store of a buffer that REST stores) -- e.g. the ring slot-reuse edge: the
            // next iteration's TMA store vs the previous (async) wgmma read. Default (flag off) keeps
            // the conservative unconditional barrier (byte-identical NFC).
            if (membar) {
                auto intersects = [](const std::set<std::string> &a, const std::set<std::string> &b) {
                    for (const auto &x : a) {
                        if (b.count(x)) return true;
                    }
                    return false;
                };
                // FIRST's own loads = accumulated loads minus what was already there before first.
                std::set<std::string> first_shared_loads, first_device_loads;
                for (const auto &x : shared_loads) {
                    if (!shared_loads_before.count(x)) first_shared_loads.insert(x);
                }
                for (const auto &x : device_loads) {
                    if (!device_loads_before.count(x)) first_device_loads.insert(x);
                }
                bool war_waw = false;
                if (intersects(first_shared_loads, rest_shared_stores) ||
                    intersects(shared_stores, rest_shared_stores)) {
                    war_waw = true;
                    mask |= CodeGen_GPU_Dev::MemoryFenceType::Shared;
                }
                if (intersects(first_device_loads, rest_device_stores) ||
                    intersects(device_stores, rest_device_stores)) {
                    war_waw = true;
                    mask |= CodeGen_GPU_Dev::MemoryFenceType::Device;
                }
                if (!any_match && !war_waw) {
                    // No within-block hazard needing a barrier HERE. But keep injected_barrier set so
                    // the enclosing serial loop still emits its once-per-iteration end barrier, which
                    // (after the consume's wgmma.wait_group) orders the CROSS-iteration ring slot reuse
                    // -- the next iteration's TMA store vs this iteration's wgmma read. Dropping the
                    // per-op barriers (the win) while keeping the one loop barrier (correctness).
                    injected_barrier = true;
                    return Block::make(first, rest);
                }
            }
            injected_barrier = true;
            // M2: barrier scope = scope(join(active(producer), active(consumer))). When
            // that scope is Thread (producer and consumer are the SAME single lane) the
            // dependency is satisfied by program order and NO barrier is needed -- this is
            // where the old scope-blindness (make_barrier always Block) dies. Every CURRENT
            // producer/consumer is cooperative (active = whole block => Block), so this is
            // byte-identical until an explicit single-thread guard appears (M4 TMA). The
            // WarpGroup-scope named-barrier (count) refinement lands with the ring (M3).
            if (any_match && exec.scope(combined) == ExecScope::Thread) {
                return Block::make(first, rest);
            }
            return Block::make({first, make_barrier(mask), rest});
        } else {
            return IRMutator::visit(op);
        }
    }

public:
    InjectThreadBarriers(ExtractSharedAndHeapAllocations &sha, ExtractRegisterAllocations &ra,
                         ExecMap &exec)
        : block_allocs(sha),
          register_allocs(ra),
          exec(exec) {
    }
};

class FuseGPUThreadLoopsSingleKernel : public IRMutator {
protected:
    using IRMutator::visit;
    const ExtractBlockSize &block_size;
    ExtractSharedAndHeapAllocations &block_allocations;

    Stmt visit(const For *op) override {
        if (ends_with(op->name, gpu_block_name(0))) {
            Stmt body = op->body;

            // This is the innermost loop over blocks.
            debug(3) << "Fusing thread block:\n"
                     << body << "\n\n";

            NormalizeDimensionality n(block_size, op->device_api);
            body = n(body);

            debug(3) << "Normalized dimensionality:\n"
                     << body << "\n\n";

            Expr block_size_x = block_size.threads_dimensions() ? block_size.num_threads(0) : 1;
            ExtractRegisterAllocations register_allocs;
            ForType innermost_loop_type = ForType::GPUThread;
            if (block_size.threads_dimensions()) {
                body = register_allocs(body);
                if (register_allocs.has_lane_loop) {
                    innermost_loop_type = ForType::GPULane;
                }
            }

            debug(3) << "Extracted register-level allocations:\n"
                     << body << "\n\n";

            if (register_allocs.has_thread_loop) {
                // If there's no loop over threads, everything is already synchronous.
                // The execution-mapping model (M1) over this block body supplies the
                // active set of each producer/consumer so barrier scope is a query, not
                // an unconditional Block (research/exec_mapping_model.md). The thread
                // loops are still explicit For GPUThread here (ReplaceForWithIf runs
                // after), so the active sets are well-defined.
                ExecMap exec(body);
                InjectThreadBarriers i{block_allocations, register_allocs, exec};
                body = i(body);
            }

            debug(3) << "Injected synchronization:\n"
                     << body << "\n\n";

            ReplaceForWithIf f(block_size);
            body = f(body);

            debug(3) << "Replaced for with if:\n"
                     << body << "\n\n";

            // There is always a loop over the innermost thread dimension
            string thread_id = gpu_thread_name(0);
            // Add back in any register-level allocations
            body = register_allocs.rewrap(body, thread_id);
            body = For::make(thread_id, 0, block_size_x - 1, innermost_loop_type, op->partition_policy, op->device_api, body, GPUVectorScope::Register, -1);

            // Rewrap the whole thing in other loops over threads
            for (int i = 1; i < block_size.threads_dimensions(); i++) {
                thread_id = gpu_thread_name(i);
                body = register_allocs.rewrap(body, thread_id);
                body = For::make(thread_id, 0, block_size.num_threads(i) - 1,
                                 ForType::GPUThread, op->partition_policy, op->device_api, body, GPUVectorScope::Register, -1);
            }
            thread_id.clear();
            body = register_allocs.rewrap(body, thread_id);

            debug(3) << "Rewrapped in for loops:\n"
                     << body << "\n\n";

            // Add back in the shared allocations
            body = block_allocations.rewrap_block(body, block_size);
            debug(3) << "Add back in shared allocations:\n"
                     << body << "\n\n";

            if (body.same_as(op->body)) {
                return op;
            } else {
                return For::make(op->name, op->min, op->max, op->for_type, op->partition_policy, op->device_api, body, op->realization, op->warps_per_group);
            }
        } else {
            return IRMutator::visit(op);
        }
    }

public:
    FuseGPUThreadLoopsSingleKernel(const ExtractBlockSize &bs,
                                   ExtractSharedAndHeapAllocations &sm)
        : block_size(bs), block_allocations(sm) {
    }
};

// Part 2 of the Fork-aware lowering: rewrite any device warp-spec Fork in this
// statement into a flat 1D thread partition (defined below, after ThreadExtents).
// No Fork => returns the statement unchanged (the identity invariant).
Stmt flatten_warp_spec_forks(const Stmt &s, int warp_size,
                             const std::map<std::string, Function> &env);

class FuseGPUThreadLoops : public IRMutator {
    const int warp_size;
    const std::map<std::string, Function> &env;

public:
    FuseGPUThreadLoops(int warp_size, const std::map<std::string, Function> &env)
        : warp_size(warp_size), env(env) {
    }

protected:
    using IRMutator::visit;

    Stmt visit(const For *op) override {
        user_assert(!(op->for_type == ForType::GPUThread ||
                      op->for_type == ForType::GPULane))
            << "Loops over GPU thread variable: \"" << op->name
            << "\" is outside of any loop over a GPU block variable. "
            << "This schedule is malformed. There must be a GPU block "
            << "variable, and it must reordered to be outside all GPU "
            << "thread variables.\n";

        if (op->for_type == ForType::GPUBlock) {
            // Warp-spec forks become a flat thread partition before block-size
            // analysis, so ExtractBlockSize sums the groups (one flat dim) instead
            // of maxing them. A kernel with no fork is returned unchanged.
            Stmt loop = flatten_warp_spec_forks(op, warp_size, env);

            // Do the analysis of thread block size and shared memory usage.
            ExtractBlockSize block_size;
            loop.accept(&block_size);

            ExtractSharedAndHeapAllocations block_allocations(op->device_api);
            loop = block_allocations(loop);

            debug(3) << "Pulled out shared allocations:\n"
                     << loop << "\n\n";

            // Mutate the inside of the kernel
            loop = FuseGPUThreadLoopsSingleKernel(block_size, block_allocations)(loop);

            loop = block_allocations.rewrap_kernel_launch(loop, block_size, op->device_api);

            return loop;
        } else {
            return IRMutator::visit(op);
        }
    }
};

class ZeroGPULoopMins : public IRMutator {
protected:
    bool in_non_glsl_gpu = false;
    using IRMutator::visit;

    Stmt visit(const For *op) override {
        ScopedValue<bool> old_in_non_glsl_gpu(in_non_glsl_gpu);

        in_non_glsl_gpu = (in_non_glsl_gpu && op->device_api == DeviceAPI::None) ||
                          (op->device_api == DeviceAPI::CUDA) || (op->device_api == DeviceAPI::OpenCL) ||
                          (op->device_api == DeviceAPI::Metal) ||
                          (op->device_api == DeviceAPI::D3D12Compute) ||
                          (op->device_api == DeviceAPI::Vulkan);

        Stmt stmt = IRMutator::visit(op);
        if (is_gpu(op->for_type) && !is_const_zero(op->min)) {
            op = stmt.as<For>();
            internal_assert(op);
            Expr adjusted = Variable::make(Int(32), op->name) + op->min;
            Stmt body = substitute(op->name, adjusted, op->body);
            stmt = For::make(op->name, 0, simplify(op->max - op->min), op->for_type, op->partition_policy, op->device_api, body, op->realization, op->warps_per_group);
        }
        return stmt;
    }

public:
    ZeroGPULoopMins() = default;
};

}  // namespace

// Also used by InjectImageIntrinsics
Stmt zero_gpu_loop_mins(const Stmt &s) {
    return ZeroGPULoopMins()(s);
}

namespace {

// Find the inner most GPU block of a statement.
class FindInnermostGPUBlock : public IRVisitor {
protected:
    using IRVisitor::visit;

    void visit(const For *op) override {
        if (op->for_type == ForType::GPUBlock) {
            // Set the last found GPU block to found_gpu_block.
            found_gpu_block = op;
        }
        IRVisitor::visit(op);
    }

public:
    const For *found_gpu_block = nullptr;
};

// Given a condition and a loop, add the condition
// to the loop body.
class AddConditionToALoop : public IRMutator {
protected:
    using IRMutator::visit;

    Stmt visit(const For *op) override {
        if (op != loop) {
            return IRMutator::visit(op);
        }

        return For::make(op->name, op->min, op->max, op->for_type, op->partition_policy, op->device_api,
                         IfThenElse::make(condition, op->body, Stmt()), op->realization, op->warps_per_group);
    }

public:
    AddConditionToALoop(const Expr &condition, const For *loop)
        : condition(condition), loop(loop) {
    }
    const Expr &condition;
    const For *loop;
};

// Push if statements between GPU blocks through all GPU blocks.
// Throw error if the if statement has an else clause.
class NormalizeIfStatements : public IRMutator {
protected:
    using IRMutator::visit;

    bool inside_gpu_blocks = false;

    Stmt visit(const For *op) override {
        if (op->for_type != ForType::GPUBlock) {
            return IRMutator::visit(op);
        }
        ScopedValue<bool> old_inside_gpu_blocks(inside_gpu_blocks, true);
        return IRMutator::visit(op);
    }

    Stmt visit(const IfThenElse *op) override {
        if (!inside_gpu_blocks) {
            return IRMutator::visit(op);
        }
        FindInnermostGPUBlock find;
        find(op);
        if (find.found_gpu_block != nullptr) {
            internal_assert(!op->else_case.defined()) << "Found an if statement with else case between two GPU blocks.\n";
            return AddConditionToALoop(op->condition, find.found_gpu_block)(op->then_case);
        }
        return IRMutator::visit(op);
    }
};

// The highest GPU thread/lane dimension (0..2) used in this statement, or -1 if
// none. The warp-group split goes one dimension above this; if that would exceed
// dim 2 there's no room and we fall back to synchronous staging.
class MaxThreadDim : public IRVisitor {
    using IRVisitor::visit;
    void visit(const For *op) override {
        if (op->for_type == ForType::GPUThread || op->for_type == ForType::GPULane) {
            for (int i = 0; i < 3; i++) {
                if (ends_with(op->name, gpu_thread_name(i))) {
                    max_dim = std::max(max_dim, i);
                }
            }
        }
        IRVisitor::visit(op);
    }

public:
    int max_dim = -1;
};

// Does this statement contain a Realize of a warp-specialized producer other
// than `self`? Nested warp specialization isn't handled yet.
class ContainsNestedWarpSpec : public IRVisitor {
    const std::map<std::string, Function> &env;
    const std::string &self;
    using IRVisitor::visit;
    void visit(const Realize *op) override {
        if (op->name != self) {
            auto it = env.find(op->name);
            if (it != env.end() && is_gpu_warp_specialized(it->second)) {
                found = true;
            }
        }
        IRVisitor::visit(op);
    }

public:
    bool found = false;
    ContainsNestedWarpSpec(const std::map<std::string, Function> &env, const std::string &self)
        : env(env), self(self) {
    }
};

// For a single warp-specialized producer, wrap its produce body so it runs on
// warp group 0 and its consume body so it runs on warp group 1, by injecting an
// outer GPU-thread loop over the (otherwise unused) dim-2 thread dimension and
// guarding each side. fuse_gpu_thread_loops then fuses this into a launch with
// disjoint producer/consumer warp groups and a whole-block barrier between them.
class WrapWarpGroups : public IRMutator {
    const std::string &name;
    DeviceAPI device_api;
    int wg_dim;
    using IRMutator::visit;

    Stmt wrap(const Stmt &body, int group) {
        // Single producer/consumer pair -> two warp groups total, placed on the
        // lowest free thread dimension.
        const std::string wg = unique_name("warp_group") + gpu_thread_name(wg_dim);
        Expr v = Variable::make(Int(32), wg);
        Stmt guarded = IfThenElse::make(v == group, body);
        // min 0, max 1 => extent 2 (group 0 = producer, group 1 = consumer).
        return For::make(wg, 0, 1, ForType::GPUThread, Partition::Never, device_api, guarded, GPUVectorScope::Register, -1);
    }

    Stmt visit(const ProducerConsumer *op) override {
        if (op->name == name) {
            int group = op->is_producer ? 0 : 1;
            return ProducerConsumer::make(op->name, op->is_producer, wrap(op->body, group));
        }
        return IRMutator::visit(op);
    }

public:
    WrapWarpGroups(const std::string &name, DeviceAPI device_api, int wg_dim)
        : name(name), device_api(device_api), wg_dim(wg_dim) {
    }
};

class InjectGPUWarpSpecialization : public IRMutator {
    const std::map<std::string, Function> &env;
    DeviceAPI device_api = DeviceAPI::None;
    // When true, do not specialize producers in this subtree. Set when a producer
    // falls back while other warp-spec producers are present in the same region,
    // so multiple producers in one block fall back together (rather than one
    // specializing while the others run on all warp groups).
    bool disable = false;
    using IRMutator::visit;

    Stmt visit(const For *op) override {
        ScopedValue<DeviceAPI> d(device_api,
                                 op->device_api != DeviceAPI::None ? op->device_api : device_api);
        return IRMutator::visit(op);
    }

    Stmt visit(const Realize *op) override {
        auto it = env.find(op->name);
        if (disable || it == env.end() || !is_gpu_warp_specialized(it->second) ||
            it->second.schedule().ring_buffer().defined()) {
            // ring warp-spec producers go through lower_gpu_warp_async (fork mapping);
            // this whole-CTA pass only handles the non-ring (depth-1) case.
            return IRMutator::visit(op);
        }

        // Conservative guards: only handle the simple, safe case. Otherwise
        // leave it unchanged (synchronous shared-memory staging, still correct).
        MaxThreadDim mtd;
        op->body.accept(&mtd);
        int wg_dim = mtd.max_dim + 1;  // warp-group split goes one dim above.
        ContainsNestedWarpSpec nested(env, op->name);
        op->body.accept(&nested);
        if (mtd.max_dim < 0 || wg_dim > 2 || nested.found || device_api == DeviceAPI::None) {
            // Multiple warp-spec producers in one region (nested.found) all fall
            // back together: a single specialized producer alongside a fallen-back
            // one would double-write the fallen-back producer's shared buffer.
            ScopedValue<bool> d(disable, disable || nested.found);
            return IRMutator::visit(op);
        }

        // Exactly one warp-spec producer here; specialize it. Don't specialize
        // anything further down this subtree.
        ScopedValue<bool> d(disable, true);
        Stmt body = WrapWarpGroups(op->name, device_api, wg_dim)(op->body);
        body = mutate(body);
        return Realize::make(op->name, op->types, op->memory_type,
                             op->bounds, op->condition, body);
    }

public:
    InjectGPUWarpSpecialization(const std::map<std::string, Function> &env)
        : env(env) {
    }
};

// Max GPU-thread extent per dim, used to size the warp-group split + barrier count.
class ThreadExtents : public IRVisitor {
    using IRVisitor::visit;
    void visit(const For *op) override {
        if (op->for_type == ForType::GPUThread || op->for_type == ForType::GPULane) {
            for (int i = 0; i < 3; i++) {
                if (ends_with(op->name, gpu_thread_name(i))) {
                    max_dim = std::max(max_dim, i);
                    Expr e = op->extent();
                    extent[i] = extent[i].defined() ? simplify(Max::make(extent[i], e)) : e;
                }
            }
        }
        IRVisitor::visit(op);
    }

public:
    int max_dim = -1;
    Expr extent[3];
};

// Replace one fork branch's gpu_thread loops with their reconstruction from a flat
// thread id. The branch occupies flat ids [base, base+size); `local` = flat - base.
// Thread dim d is recovered as (local / stride[d]) and, for all but the top dim,
// % max_extent[d]; a per-dim guard `idx < actual_extent` masks tails and warp
// padding (the named ring barriers sit outside these guards, so all `size` lanes
// still reach them — matching the per-edge counts).
class FlattenBranchThreads : public IRMutator {
    Expr local;
    const Expr *stride;
    const Expr *max_extent;
    int max_dim;
    using IRMutator::visit;

    Stmt visit(const For *op) override {
        int d = -1;
        if (op->for_type == ForType::GPUThread || op->for_type == ForType::GPULane) {
            for (int i = 0; i < 3; i++) {
                if (ends_with(op->name, gpu_thread_name(i))) {
                    d = i;
                    break;
                }
            }
        }
        if (d < 0) {
            return IRMutator::visit(op);
        }
        Expr idx = simplify(local / stride[d]);
        if (d < max_dim) {
            idx = simplify(idx % max_extent[d]);  // top dim keeps the full quotient
        }
        Stmt body = mutate(op->body);
        body = substitute(op->name, op->min + idx, body);
        return IfThenElse::make(idx < op->extent(), body);
    }

public:
    FlattenBranchThreads(const Expr &local, const Expr *stride,
                         const Expr *max_extent, int max_dim)
        : local(local), stride(stride), max_extent(max_extent), max_dim(max_dim) {
    }
};

// Partition a block's flat thread-id space across an ordered list of warp groups
// (`branches`): group g runs on a contiguous, warp-aligned range
// [base_g, base_g+size_g), with base summed across the list (sum-between groups,
// max-within a group via ThreadExtents). Each branch's gpu_thread loops are
// reconstructed from (flat_id - base_g) and guarded to its range. Returns one
// gpu_thread loop of extent Sum(size_g) on dim 0; ExtractBlockSize reads that sum
// as blockDim.x.
//
// Source-agnostic by design (the Step-2 fuser redesign): the branch list may come
// from an async Fork (role-asymmetric, different bodies) or — once gpu_warps lands —
// a data-symmetric group split (identical bodies, different group index). The
// per-group sizing/guard/flat-id math is identical for both; only how the branches
// are produced differs.
// The largest warps_per_group (warp-group collective scope, e.g. wgmma) anywhere in a branch,
// or 0 if none. Used to warp-group-align the flat thread partition so a wgmma consumer's 4-warp
// group lands on an aligned boundary.
// The name of the first producing Func in a warp-spec branch (its ProducerConsumer), or "".
// Used to look up a branch's explicit gpu_warp_group assignment in the environment.
std::string producer_name_of(const Stmt &s) {
    class V : public IRVisitor {
        using IRVisitor::visit;
        void visit(const ProducerConsumer *op) override {
            if (op->is_producer && name.empty()) {
                name = op->name;
            }
            IRVisitor::visit(op);
        }

    public:
        std::string name;
    } v;
    s.accept(&v);
    return v.name;
}

int max_warps_per_group(const Stmt &s) {
    class V : public IRVisitor {
        using IRVisitor::visit;
        void visit(const For *op) override {
            if (op->warps_per_group > 0) {
                m = std::max(m, op->warps_per_group);
            }
            IRVisitor::visit(op);
        }

    public:
        int m = 0;
    } v;
    s.accept(&v);
    return v.m;
}

// The warp-group participant count of a fork branch / warp-group peel: its collective-
// aware, warp-rounded hardware-lane assignment. THE single source of truth -- both
// partition_warp_groups (the flat lane partition) and LowerGPUWarpAsyncFork's per-edge
// barrier count query it, so the "count computed three ways" deadlock (a rounding mismatch
// between the partition's lane ranges and the barrier's arriving-lane count) cannot recur.
// This is the execution-mapping model's count(scope = WarpGroup) realized for the
// PRE-partition Fork: the branch is not yet a thread-guard region (the Fork is split into
// warp groups later, in fuse), so it is measured by its ThreadExtents. A collective branch
// (warps_per_group = N, e.g. the wgmma consumer) occupies exactly its N warps; an explicit
// peel override fixes N warps; a non-collective branch (producer/DMA) is its thread tile
// rounded up to a whole warp. See research/exec_mapping_model.md, fusegpu_rearch_plan.md
// C.0/M3 (the model's count query table).
Expr warp_group_lane_count(const Stmt &branch, int warp_size, int warps_per_group_override = 0) {
    if (warps_per_group_override > 0) {
        return Expr(warps_per_group_override * warp_size);  // explicit symmetric peel
    }
    ThreadExtents te;
    branch.accept(&te);
    Expr prod = 1;
    for (int d = 0; d <= te.max_dim; d++) {
        if (te.extent[d].defined()) {
            prod = prod * te.extent[d];
        }
    }
    prod = simplify(prod);
    if (int wpg = max_warps_per_group(branch); wpg > 0) {
        // Collective branch (a warp-group scope, e.g. the wgmma consumer): its lane count is the FULL
        // thread extent rounded up to whole warp groups. The full extent already includes the
        // warp-group AXIS multiplicity -- an M-split consumer (R5's 128x256 = 2 m64n256 groups) is
        // gpu_warps(wg=2)*gpu_threads(tx=128) = 256 lanes = 2 groups, NOT one. The old per-group
        // shortcut (wpg*warp_size = 4*32 = 128) sized it as a SINGLE group, so partition_warp_groups
        // gave the consumer only 128 lanes -> only group 0 ran -> rows 0-63 (the other M-half + the
        // group's epilogue thread_id_y reconstruction were dropped). NFC for 1 group (extent == 128).
        int g = wpg * warp_size;
        return simplify(((prod + (g - 1)) / g) * g);
    }
    return simplify(((prod + (warp_size - 1)) / warp_size) * warp_size);
}

Stmt partition_warp_groups(const std::vector<Stmt> &branches, int warp_size,
                           DeviceAPI device_api, int warps_per_group_override = 0,
                           const std::vector<int> &group_index = {},
                           const std::vector<std::pair<int, bool>> &reg_budget = {}) {
    const std::string ftid = unique_name("warp_flat") + gpu_thread_name(0);
    Expr fv = Variable::make(Int(32), ftid);
    Expr base = 0;
    Stmt body;
    // PLACEMENT ORDER (F2 explicit warp-group assignment): lay the branches out by ascending
    // gpu_warp_group index (stable; default = the given branch order). This is pure placement --
    // it reorders WHICH warp range each branch occupies (same bodies/handshakes/barrier counts),
    // so e.g. assigning the wgmma consumer index 0 puts it on the first, naturally-aligned warp
    // group (warps 0-3) with the producers packed after (no alignment gap). NOT folding: same-index
    // branches are still laid out at distinct (adjacent) bases -- co-residency of two ring
    // producers on one group needs producer fusion, not shared placement (it deadlocks otherwise).
    std::vector<int> order(branches.size());
    for (int i = 0; i < (int)branches.size(); i++) {
        order[i] = i;
    }
    if (!group_index.empty()) {
        std::stable_sort(order.begin(), order.end(),
                         [&](int a, int b) { return group_index[a] < group_index[b]; });
    }
    // COLLECTIVE-AWARE PLACEMENT (the explicit-warp-group-assignment mechanism, F2). A branch
    // that carries a warp-group collective (a warps_per_group=N scope, e.g. the wgmma consumer)
    // must start at a warp-GROUP boundary and occupy N warps -- wgmma faults ("Illegal Instruction
    // Encoding") if its 4-warp group is misaligned. Only such a branch is pinned to a group
    // boundary + group-sized; non-collective branches (producer/DMA warps) need only WARP
    // alignment and pack tightly. A gap before an aligned collective base is fine: those lanes are
    // in no branch's guard, so they idle and are excluded from the per-edge barrier counts.
    for (int oi : order) {
        const Stmt &branch = branches[oi];
        ThreadExtents te;
        branch.accept(&te);
        Expr stride[3], maxe[3], prod = 1;
        for (int i = 0; i < 3; i++) {
            stride[i] = prod;
            maxe[i] = te.extent[i].defined() ? te.extent[i] : Expr(1);
            prod = simplify(prod * maxe[i]);
        }
        int branch_wpg = max_warps_per_group(branch);  // >0 => this branch has a warp-group collective
        // Size = the warp-group participant count (the single source of truth shared with the ring's
        // per-edge barrier count): explicit peel override -> N warps; collective -> its N warps;
        // non-collective -> the tile rounded to a warp. See warp_group_lane_count.
        Expr size = warp_group_lane_count(branch, warp_size, warps_per_group_override);
        // Pin a collective branch's base to a warp-group boundary (round up; a gap may precede it).
        if (warps_per_group_override == 0 && branch_wpg > 0) {
            int align = branch_wpg * warp_size;
            base = simplify(((base + (align - 1)) / align) * align);
        }
        Stmt fb = FlattenBranchThreads(simplify(fv - base), stride, maxe, te.max_dim)(branch);
        // Hopper per-warp-group register reallocation (setmaxnreg): if the Func placed on this
        // branch set a register budget, emit it at the ENTRY of the branch body, INSIDE the group
        // guard below so the whole warp group (and only it) executes the warp-group-collective
        // .sync.aligned instruction. arg[0] = reg count (compile-time immediate), arg[1] = inc/dec.
        if (!reg_budget.empty() && reg_budget[oi].first >= 0) {
            Stmt smn = Evaluate::make(Call::make(Int(32), "setmaxnreg",
                                                 {Expr(reg_budget[oi].first),
                                                  Expr((int)reg_budget[oi].second)},
                                                 Call::Intrinsic));
            fb = Block::make(smn, fb);
        }
        // Group range guard: only this group's warp range runs the branch (incl. its
        // cross-group barriers), so per-edge barrier counts (= sum of two groups) hold.
        fb = IfThenElse::make(fv >= base && fv < simplify(base + size), fb);
        body = body.defined() ? Block::make(body, fb) : fb;
        base = simplify(base + size);
    }
    // For stores an inclusive max; max = total-1 gives extent total.
    return For::make(ftid, 0, simplify(base - 1), ForType::GPUThread,
                     Partition::Never, device_api, body, GPUVectorScope::Register, -1);
}

// Loop-fuse two co-resident producer branches placed on the same warp. Each is `lets...; For(ko, ...)`
// over the SAME ring loop. INTERLEAVE them -> `lets_a; lets_b; For(ko, ..., {body_a; body_b})` so one
// elected lane issues BOTH operands' transfers PER ko iteration (the matmul_4 shape). Sequential
// concatenation (`{For(ko){A}; For(ko){B}}`) instead deadlocks the empty/WAR edge at ring wrap: the A
// loop's empty-wait at ko=Q blocks because the consumer can't free slot 0 without B[0], which the B loop
// hasn't produced yet. Falls back to Block (sequential) if the two branches aren't matching For loops.
Stmt fuse_coresident_producers(const Stmt &a, const Stmt &b) {
    std::vector<const LetStmt *> a_lets, b_lets;
    Stmt sa = a, sb = b;
    while (const LetStmt *l = sa.as<LetStmt>()) { a_lets.push_back(l); sa = l->body; }
    while (const LetStmt *l = sb.as<LetStmt>()) { b_lets.push_back(l); sb = l->body; }
    const For *fa = sa.as<For>();
    const For *fb = sb.as<For>();
    if (fa && fb && fa->name == fb->name && equal(fa->min, fb->min) && equal(fa->max, fb->max)) {
        Stmt merged = For::make(fa->name, fa->min, fa->max, fa->for_type, fa->partition_policy,
                                fa->device_api, Block::make(fa->body, fb->body),
                                fa->realization, fa->warps_per_group);
        for (auto it = b_lets.rbegin(); it != b_lets.rend(); ++it) {
            merged = LetStmt::make((*it)->name, (*it)->value, merged);
        }
        for (auto it = a_lets.rbegin(); it != a_lets.rend(); ++it) {
            merged = LetStmt::make((*it)->name, (*it)->value, merged);
        }
        return merged;
    }
    return Block::make(a, b);
}

// Convert a device warp-spec Fork into a flat 1D thread partition by collecting its
// branches and handing them to partition_warp_groups (the source-agnostic core).
class FlattenWarpSpecForks : public IRMutator {
    DeviceAPI device_api = DeviceAPI::None;
    const int warp_size;
    const std::map<std::string, Function> &env;
    using IRMutator::visit;

public:
    FlattenWarpSpecForks(int warp_size, const std::map<std::string, Function> &env)
        : warp_size(warp_size), env(env) {
    }

private:
    // The explicit gpu_warp_group placement index of a branch (from its producing Func's
    // schedule), or -1 if unset. Used to order the flat partition (F2).
    int branch_group_index(const Stmt &s) const {
        std::string pname = producer_name_of(s);
        if (!pname.empty()) {
            auto it = env.find(pname);
            if (it != env.end()) {
                const std::vector<int> &g = it->second.schedule().gpu_warp_group();
                if (!g.empty()) {
                    return g[0];
                }
            }
        }
        return -1;
    }

    // The Hopper register budget (setmaxnreg) of a branch, from its producing Func's schedule:
    // {regs, increase} with regs >= 0, or {-1, false} if unset. Mirrors branch_group_index.
    std::pair<int, bool> branch_reg_budget(const Stmt &s) const {
        std::string pname = producer_name_of(s);
        if (!pname.empty()) {
            auto it = env.find(pname);
            if (it != env.end()) {
                const auto &sched = it->second.schedule();
                if (sched.gpu_register_budget() >= 0) {
                    return {sched.gpu_register_budget(), sched.gpu_register_increase()};
                }
            }
        }
        return {-1, false};
    }

    static void flatten(const Stmt &s, std::vector<Stmt> &branches) {
        if (const Fork *f = s.as<Fork>()) {
            branches.push_back(f->first);
            flatten(f->rest, branches);
        } else {
            branches.push_back(s);
        }
    }

    Stmt visit(const For *op) override {
        ScopedValue<DeviceAPI> d(device_api,
                                 op->device_api != DeviceAPI::None ? op->device_api : device_api);
        if (op->warps_per_group > 0) {
            // EXPLICIT wgmma-scope sizing: each group is exactly N warps regardless of
            // the thread tile (idle lanes masked). That padding can't be a plain thread
            // dimension, so we peel into N symmetric warp-group branches (group index
            // substituted in) and flat-partition them via partition_warp_groups (the
            // same core the async Fork uses). NOTE: the guarded branches can't yet carry
            // a correct shared-memory barrier (that needs the WarpGroup-scope scoped
            // barrier — research/gpu_sync_model.md step 4); explicit-size groups are
            // pointwise-only until then.
            Stmt body = mutate(op->body);
            auto n = as_const_int(simplify(op->extent()));
            user_assert(n && *n > 0)
                << "gpu_warps axis " << op->name
                << " must have a constant positive extent (the number of warp groups), got: "
                << op->extent() << "\n";
            std::vector<Stmt> branches;
            branches.reserve((size_t)*n);
            for (int64_t g = 0; g < *n; g++) {
                branches.push_back(substitute(op->name, op->min + (int)g, body));
            }
            // Hopper register budget (setmaxnreg): the peeled groups share one producing Func, so
            // all symmetric branches carry its budget (if any). Empty => no setmaxnreg (NFC).
            std::pair<int, bool> rb = branch_reg_budget(body);
            std::vector<std::pair<int, bool>> rbudget;
            if (rb.first >= 0) {
                rbudget.assign(branches.size(), rb);
            }
            return partition_warp_groups(branches, warp_size, device_api, op->warps_per_group,
                                         {}, rbudget);
        }
        // A DERIVED-size (warps_per_group == 0) symmetric gpu_warps axis lowers as a
        // normal thread sub-dimension: the hardware decodes the group index from
        // threadIdx (group = high thread dim, within-group lane = low dims), so the
        // group's produce/consume stay separate thread loops at block level and the
        // standard fusion + barrier pipeline applies — a shared-memory consumer gets a
        // correct whole-CTA barrier reached by every lane (the "decode model",
        // research/gpu_sync_model.md §3). The role-asymmetric Fork still peels (visit
        // Fork). warps_per_group < 0 (not a gpu_warps axis) is also unchanged here.
        return IRMutator::visit(op);
    }

    Stmt visit(const Fork *op) override {
        std::vector<Stmt> branches;
        flatten(op, branches);
        // F2 explicit warp-group assignment: order the flat partition by each branch's
        // gpu_warp_group index. Unset branches sort AFTER assigned ones (in their flatten order),
        // so assigning only the consumer index 0 places it first (wgmma on warps 0-3, aligned) with
        // the producers packed after. All unset => empty => NFC (the flatten order is preserved).
        const int UNSET = 1 << 30;
        std::vector<int> gidx(branches.size());
        // Hopper register budget (setmaxnreg) per branch, keyed on the same producing Func as the
        // warp-group placement. Empty unless some branch sets it (opt-in, NFC otherwise).
        std::vector<std::pair<int, bool>> rbudget(branches.size(), {-1, false});
        bool any = false, any_rb = false;
        for (int i = 0; i < (int)branches.size(); i++) {
            int g = branch_group_index(branches[i]);
            gidx[i] = (g < 0) ? UNSET : g;
            any = any || (g >= 0);
            rbudget[i] = branch_reg_budget(branches[i]);
            any_rb = any_rb || (rbudget[i].first >= 0);
        }
        // PLACEMENT (async_storage_model.md S4): CO-RESIDENT producers. By default each async producer
        // gets its own warp (spread). When several NON-collective producers are placed on the SAME warp
        // group (same gpu_warp_group index), fuse their branches into ONE warp branch so a SINGLE
        // elected lane issues all their transfers serially (the matmul_4 shape: one thread issues both
        // operand TMAs). The two-warp (spread) choice stays available by giving the producers DIFFERENT
        // group indices. Gated while bringing up; collective (wgmma) branches are never fused.
        if (any && get_env_variable("HL_WG_FUSE_PRODUCERS") == "1") {
            std::vector<Stmt> nb;
            std::vector<int> ng;
            std::vector<std::pair<int, bool>> nrb;
            std::map<int, int> fused_at;  // group index -> index of its fused producer branch in nb
            for (int i = 0; i < (int)branches.size(); i++) {
                bool collective = max_warps_per_group(branches[i]) > 0;
                if (!collective && gidx[i] != UNSET && fused_at.count(gidx[i])) {
                    int j = fused_at[gidx[i]];
                    nb[j] = fuse_coresident_producers(nb[j], branches[i]);
                    if (nrb[j].first < 0) {
                        nrb[j] = rbudget[i];  // the fused producer inherits a register budget
                    }
                } else {
                    if (!collective && gidx[i] != UNSET) {
                        fused_at[gidx[i]] = (int)nb.size();
                    }
                    nb.push_back(branches[i]);
                    ng.push_back(gidx[i]);
                    nrb.push_back(rbudget[i]);
                }
            }
            branches = std::move(nb);
            gidx = std::move(ng);
            rbudget = std::move(nrb);
            any_rb = false;
            for (auto &rb : rbudget) any_rb = any_rb || (rb.first >= 0);
        }
        return partition_warp_groups(branches, warp_size, device_api, 0,
                                     any ? gidx : std::vector<int>{},
                                     any_rb ? rbudget : std::vector<std::pair<int, bool>>{});
    }
};

Stmt flatten_warp_spec_forks(const Stmt &s, int warp_size,
                             const std::map<std::string, Function> &env) {
    return FlattenWarpSpecForks(warp_size, env)(s);
}

// Is `name` a host semaphore of a warp-specialized ring producer? (Those are
// "<prod>.semaphore_<i>" / "<prod>.folding_semaphore.ring_buffer".)
bool sema_belongs_to_warpspec_ring(const std::string &name,
                                   const std::map<std::string, Function> &env) {
    std::string prod;
    const std::string ring_suffix = ".folding_semaphore.ring_buffer";
    if (ends_with(name, ring_suffix)) {
        prod = name.substr(0, name.size() - ring_suffix.size());
    } else {
        size_t pos = name.rfind(".semaphore_");
        if (pos == std::string::npos) {
            return false;
        }
        prod = name.substr(0, pos);
    }
    auto it = env.find(prod);
    return it != env.end() && is_gpu_warp_specialized(it->second) &&
           it->second.schedule().ring_buffer().defined();
}

// Collect the names of every warp-specialized ring producer in a statement, in
// pre-order (outermost HoistedStorage first). Used to assign each producer a warp
// group and a sequential block of named-barrier ids.
class CollectWarpSpecRing : public IRVisitor {
    const std::map<std::string, Function> &env;
    using IRVisitor::visit;
    void visit(const HoistedStorage *op) override {
        auto it = env.find(op->name);
        if (it != env.end() && is_gpu_warp_specialized(it->second) &&
            it->second.schedule().ring_buffer().defined()) {
            producers.push_back(op->name);
        }
        IRVisitor::visit(op);
    }

public:
    std::vector<std::string> producers;
    CollectWarpSpecRing(const std::map<std::string, Function> &env)
        : env(env) {
    }
};

// (a)-1/(a)-3: lower a device async Fork of N warp-specialized ring producers to an
// (N+1)-way warp-group split coordinated by per-slot named barriers. Producer i runs
// on warp group wg==i, the consumer on wg>=N (an injected outer GPU-thread dim).
// acquire/release become gpu_named_barrier with per-slot ids; each producer owns a
// block full=[2N*i, 2N*i+N), empty=[2N*i+N, 2N*i+2N) keyed by its semaphore names.
// Each producer's first N empty-waits are primed out (slots start free); host
// semaphores are stripped. Barrier wait/arrive is emitted via emit_barrier() so
// mbarrier can later drop in at the same seam.

class LowerGPUWarpAsyncFork : public IRMutator {
    const std::map<std::string, Function> &env;
    DeviceAPI device_api = DeviceAPI::None;
    bool active = false;
    std::string ring_loop;
    int wg_dim = -1;
    int num_producers = 0;  // producers occupy wg [0, num_producers); consumer is wg >= num_producers
    Expr thread_count;
    // The Fork-aware (flat-partition) lowering is the default: leave the Fork for the
    // Fork-aware fuser and use per-edge barrier counts. Set HL_GPU_WARP_FORK_FUSE=0 to
    // fall back to the older rectangular wg-dim split (escape hatch).
    bool fork_fuse = true;
    // Per-group warp-aligned thread counts (fork_fuse path), set per fork: producer i's
    // threads and the consumer's. A producer<->consumer edge barrier is reached only by
    // those two groups (flat partition), so its count is the sum of the two.
    std::vector<Expr> producer_threads;
    Expr consumer_threads;
    // Each warp-spec ring semaphore maps to a block of per-slot named-barrier ids.
    struct BarrierInfo {
        int base;       // first barrier id for this edge
        int ring_n;     // ring depth (slot = ring_loop % ring_n)
        bool is_empty;  // empty edge (producer waits) vs full edge (consumer waits)
        int producer;   // producing warp-group index (selects its thread count, fork_fuse)
        std::string mbar_name;  // F3: non-empty on the full (data) edge when HL_WG_MBAR is on ->
                                // the edge is realized by an mbarrier (cp.async-completion arrive +
                                // parity try_wait) instead of a named barrier. See §5e.
    };
    std::map<std::string, BarrierInfo> sema_map;
    const int warp_size;
    // F3: realize the full (cp.async data) edge as an mbarrier completion handshake (deep pipeline,
    // no loop skew) instead of the synchronous named-barrier + producer wait_group. Empty edge stays
    // a named barrier. Off by default (NFC); on with HL_WG_MBAR=1.
    const bool mbar;
    using IRMutator::visit;

    // mode 0 = wait, 1 = arrive. id = base + (ring_loop % ring_n) selects the slot.
    Stmt emit_barrier(const BarrierInfo &b, int mode) {
        Expr slot = Variable::make(Int(32), ring_loop) % b.ring_n;
        Expr id = b.base == 0 ? slot : (b.base + slot);
        // Rectangular: every blockDim lane of both groups hits the barrier (2*max).
        // Flat partition: only the producing + consuming groups' lanes are in range.
        Expr count = fork_fuse ? simplify(producer_threads[b.producer] + consumer_threads)
                               : thread_count;
        // Emit a WarpGroup-scope sync requirement (only the producing + consuming
        // groups rendezvous); LowerSyncRequirements lowers it to a partial named
        // barrier — or, at sm_90, an mbarrier — at the single sync seam. mode 0 =
        // wait, 1 = arrive. See research/gpu_sync_model.md.
        return Evaluate::make(Call::make(Int(32), Call::sync_requirement,
                                         {IntImm::make(Int(32), (int)SyncScope::WarpGroup),
                                          id, count, IntImm::make(Int(32), mode)},
                                         Call::Intrinsic));
    }

    // F3: a Load carrier addressing full mbarrier slot (ring_loop % ring_n). codegen derives the
    // addrspace(3) pointer from it (and ExtractSharedAndHeapAllocations folds the shared offset).
    Expr mbar_slot_ref(const BarrierInfo &b) {
        Expr slot = Variable::make(Int(32), ring_loop) % b.ring_n;
        return Load::make(UInt(64), b.mbar_name, slot, Buffer<>{}, Parameter{}, const_true(),
                          ModulusRemainder{});
    }

    Stmt visit(const LetStmt *op) override {
        if (sema_belongs_to_warpspec_ring(op->name, env)) {
            return mutate(op->body);  // drop the host-semaphore alloca
        }
        return IRMutator::visit(op);
    }

    Stmt visit(const HoistedStorage *op) override {
        auto it = env.find(op->name);
        bool is_wsr = it != env.end() && is_gpu_warp_specialized(it->second) &&
                      it->second.schedule().ring_buffer().defined();
        if (!is_wsr || active) {
            // Not a warp-spec ring producer, or a nested one already covered by the
            // outermost context — just recurse.
            return IRMutator::visit(op);
        }
        ThreadExtents te;
        op->body.accept(&te);
        int wgd = te.max_dim + 1;
        if (wgd > 2) {
            // No free thread dim for the warp-group split; leave it (will error later).
            return IRMutator::visit(op);
        }
        // Collect every warp-spec ring producer in this nest and give each a sequential
        // block of barrier ids; key both its semaphores so acquire/release map by name.
        CollectWarpSpecRing collector(env);
        op->accept(&collector);
        std::map<std::string, BarrierInfo> smap;
        int base = 0;
        for (int pi = 0; pi < (int)collector.producers.size(); pi++) {
            const std::string &prod = collector.producers[pi];
            auto n = as_const_int(env.at(prod).schedule().ring_buffer());
            internal_assert(n) << "ring_buffer extent must be a constant for warp specialization\n";
            int rn = (int)*n;
            // Full (data) edge -> mbarrier when enabled; empty (slot-reuse control) edge stays named.
            smap[prod + ".semaphore_0"] = {base, rn, /*is_empty*/ false, pi,
                                           mbar ? (prod + ".full_mbar") : std::string()};
            smap[prod + ".folding_semaphore.ring_buffer"] = {base + rn, rn, /*is_empty*/ true, pi, {}};
            base += 2 * rn;
        }
        // Per-edge participant count: producer warp group + consumer warp group. With
        // equal one-warp groups this is 2x the per-dim thread extent, independent of the
        // producer count. Asymmetric sizing (plan §9.1) will make this per-edge.
        Expr tc = 2;
        for (int i = 0; i < wgd; i++) {
            if (te.extent[i].defined()) {
                tc = tc * te.extent[i];
            }
        }
        ScopedValue<bool> a(active, true);
        ScopedValue<std::map<std::string, BarrierInfo>> sm(sema_map, smap);
        ScopedValue<int> np(num_producers, (int)collector.producers.size());
        ScopedValue<int> wd(wg_dim, wgd);
        ScopedValue<Expr> t(thread_count, simplify(tc));
        return HoistedStorage::make(op->name, mutate(op->body));
    }

    Stmt visit(const For *op) override {
        ScopedValue<DeviceAPI> d(device_api,
                                 op->device_api != DeviceAPI::None ? op->device_api : device_api);
        ScopedValue<std::string> r(ring_loop,
                                   (active && op->for_type == ForType::Serial) ? op->name : ring_loop);
        return IRMutator::visit(op);
    }

    // Halide builds an N-ary fork as right-nested binary Forks; flatten to an ordered
    // branch list [producer_0, ..., producer_{P-1}, consumer].
    static void flatten_fork(const Stmt &s, std::vector<Stmt> &branches) {
        if (const Fork *f = s.as<Fork>()) {
            branches.push_back(f->first);
            flatten_fork(f->rest, branches);
        } else {
            branches.push_back(s);
        }
    }

    // Peel leading HoistedStorage nodes off a fork branch. compute_with-fused cluster
    // members (which are not async themselves, so their storage isn't lifted by the
    // async fork) land here, duplicated inside each branch. They must instead live at
    // block level, around the warp groups: otherwise the producer and consumer warp
    // groups get distinct shared allocations and the consumer reads unwritten memory.
    static Stmt peel_hoisted(Stmt s, std::vector<std::string> &order,
                             std::set<std::string> &seen) {
        while (const HoistedStorage *h = s.as<HoistedStorage>()) {
            if (seen.insert(h->name).second) {
                order.push_back(h->name);
            }
            s = h->body;
        }
        return s;
    }

    Stmt visit(const Fork *op) override {
        if (!active) {
            return IRMutator::visit(op);
        }
        std::vector<Stmt> branches;
        flatten_fork(op, branches);
        int num_groups = (int)branches.size();  // P producers + 1 consumer
        user_assert(num_groups == num_producers + 1)
            << "Found " << num_producers << " async GPUShared ring-buffered producer(s)"
            << " but the async fork has " << num_groups << " branch(es). This happens when"
            << " async shared producers are combined with compute_with: the fusion merges"
            << " their producer bodies into one, so they can no longer each own a warp"
            << " group. Mark only one producer in a compute_with cluster as .async() — the"
            << " others are brought into its warp group by compute_with and share its"
            << " synchronization (still give them .ring_buffer(N) for double buffering).\n";

        if (fork_fuse) {
            // Leave the Fork for the Fork-aware fuser (sum-between, max-within). Here we
            // only inject the cross-group ring barriers (with per-edge counts) and lift
            // shared storage to block level; the fuser sizes/partitions the thread space.
            // The per-edge barrier count is producer_threads[i] + consumer_threads, which MUST match
            // the actual lane ranges partition_warp_groups assigns. Both now query the SAME
            // warp_group_lane_count, so the rounding can no longer disagree (the count-three-ways
            // deadlock dissolves). NFC for plain async (no collective -> per-warp rounding, as before).
            auto branch_warp_threads = [&](const Stmt &s) {
                return warp_group_lane_count(s, warp_size);
            };
            // NOTE (F2 consumption WIP): the gpu_warp_group directive is parsed/stored, but the
            // producer-fold consumption is NOT wired here. Merging co-grouped producers' BODIES
            // into one branch deadlocks (two ring producers' barrier handshakes run sequentially on
            // the shared warps don't compose). The correct mechanism is to keep the producer
            // branches SEPARATE but place same-group branches at the SAME warp-group base (overlap)
            // in partition_warp_groups -- i.e. thread the group index to the partitioner, not merge
            // bodies. Until then each producer keeps its own group (sequential), as before.
            std::vector<Expr> ptv(num_producers);
            for (int i = 0; i < num_producers; i++) {
                ptv[i] = branch_warp_threads(branches[i]);
            }
            ScopedValue<std::vector<Expr>> pt(producer_threads, ptv);
            // NOTE (R5 multi-consumer WIP): with >1 consumer warp group (M-split 128x256), the empty
            // (WAR) edge must be arrived by ALL consumer lanes before the producer refills a slot. The
            // count here is producer + ONE consumer group (branches[num_groups-1]); raising it to the
            // consumers' full span (G*128) makes the named barrier HANG -- empirically only ~1 consumer
            // group's lanes actually execute the empty-arrive after partition (count=160 races, 288
            // hangs). So the real fix is STRUCTURAL: make every consumer group execute the empty-arrive
            // (and match the count), not just bump the count. Use HL_WG_EMPTY_COUNT below to probe the
            // true participant count on H100 while pinning the topology. NFC for R4 (1 consumer).
            ScopedValue<Expr> ct(consumer_threads, branch_warp_threads(branches[num_groups - 1]));
            std::vector<std::string> lifted;
            std::set<std::string> lifted_seen;
            std::vector<Stmt> out(num_groups);
            for (int i = 0; i < num_groups; i++) {
                out[i] = mutate(peel_hoisted(branches[i], lifted, lifted_seen));
            }
            Stmt result = out.back();  // right-nested fork, consumer innermost
            for (int i = num_groups - 2; i >= 0; i--) {
                result = Fork::make(out[i], result);
            }
            // F3: per-producer full-edge mbarrier. Allocate the N-slot array at block level and arm
            // it ONCE before the fork (thread-0-guarded + CTA barrier in codegen). EXPECTED arrivals
            // = the producer branch's thread count (producer_threads[pi]) -- exactly the threads that
            // run the cp.async.mbarrier.arrive; the consumer only polls, it does not arrive. The
            // branch arrive/wait reference the array by name (a Load carrier whose shared offset
            // ExtractSharedAndHeapAllocations folds). The array is live across the whole mainloop
            // (referenced every ko), so it never coalesces with the As/Bs operand slots.
            if (mbar) {
                bool mdbg = get_env_variable("HL_WG_MBAR_DEBUG") == "1";
                if (mdbg) {
                    std::cerr << "[mbar] num_producers=" << num_producers
                              << " consumer_threads=" << simplify(consumer_threads) << "\n";
                }
                // ONE combined init for ALL producers' mbarriers (flattened triples
                // base_ref,ring_n,count) -> a single opaque init+bar.sync asm in codegen. A single,
                // outermost init lands in the uniform entry region; a second (nested) init asm got
                // placed inside a tid==0 block by the scheduler, deadlocking its internal barrier.
                std::vector<Expr> init_args;
                std::vector<BarrierInfo> mbar_allocs;
                for (const auto &kv : sema_map) {
                    const BarrierInfo &b = kv.second;
                    if (b.mbar_name.empty()) {
                        continue;  // full (data) edges only
                    }
                    if (mdbg) {
                        ThreadExtents te;
                        branches[b.producer].accept(&te);
                        Expr unr = 1;
                        for (int d = 0; d <= te.max_dim; d++) {
                            if (te.extent[d].defined()) unr = unr * te.extent[d];
                        }
                        std::cerr << "[mbar] " << b.mbar_name << " producer=" << b.producer
                                  << " ring_n=" << b.ring_n
                                  << " EXPECTED(rounded)=" << simplify(producer_threads[b.producer])
                                  << " extent(unrounded)=" << simplify(unr) << "\n";
                    }
                    Expr base_ref = Load::make(UInt(64), b.mbar_name, 0, Buffer<>{}, Parameter{},
                                               const_true(), ModulusRemainder{});
                    // EXPECTED arrivals = the number of ACTIVE producer lanes that execute the
                    // arrive. branch_warp_threads warp-ROUNDS (fine for bar.sync, which converges a
                    // whole warp), but mbarrier.arrive only fires on active lanes -> must match the
                    // actual count. HL_WG_MBAR_COUNT overrides it for diagnosis.
                    // NB: a TMA producer's expected arrival count is 1 (the elected thread's
                    // expect_tx), not the cp.async per-lane warp count -- but TMA is injected LATER
                    // (inject_tma_copies, after this pass), so that count is patched there
                    // (patch_tma_mbar_counts), not here.
                    Expr count = producer_threads[b.producer];
                    std::string cov = get_env_variable("HL_WG_MBAR_COUNT");
                    if (!cov.empty()) count = Expr(std::atoi(cov.c_str()));
                    init_args.push_back(base_ref);
                    init_args.push_back(Expr(b.ring_n));
                    init_args.push_back(count);
                    mbar_allocs.push_back(b);
                }
                if (!init_args.empty()) {
                    // CUTLASS-style uniform prologue: ONE opaque tid==0-predicated init asm (no
                    // internal barrier, no tid exposed to LLVM). NO explicit barrier here -- the init
                    // registers as a shared STORE (see InjectThreadBarriers::visit(Call)), so Halide
                    // inserts a correctly-ordered all-threads barrier between this block-level init
                    // and the arrive/try_wait reads. (My own explicit barrier got tail-duplicated by
                    // the compiler across the warp-select diamond into a tid==0 block -> deadlock.)
                    Stmt init = Evaluate::make(Call::make(Int(32), "mbarrier_init", init_args,
                                                          Call::Intrinsic));
                    result = Block::make(init, result);
                    for (const BarrierInfo &b : mbar_allocs) {
                        result = Allocate::make(b.mbar_name, UInt(64), MemoryType::GPUShared,
                                                {Expr(b.ring_n)}, const_true(), result);
                    }
                }
            }
            for (auto it = lifted.rbegin(); it != lifted.rend(); ++it) {
                result = HoistedStorage::make(*it, result);
            }
            return result;
        }

        std::string wg = unique_name("warp_group") + gpu_thread_name(wg_dim);
        Expr wgv = Variable::make(Int(32), wg);
        // Producer i takes wg == i; the consumer (last branch) takes the rest. The
        // barrier ids are keyed by semaphore name, so the wg a producer lands on is
        // independent of which barriers coordinate it.
        std::vector<std::string> lifted;
        std::set<std::string> lifted_seen;
        Stmt body;
        for (int i = 0; i < num_groups; i++) {
            Stmt branch = peel_hoisted(branches[i], lifted, lifted_seen);
            Expr cond = (i < num_producers) ? (wgv == i) : (wgv >= num_producers);
            Stmt guarded = IfThenElse::make(cond, mutate(branch));
            body = body.defined() ? Block::make(body, guarded) : guarded;
        }
        // For stores an inclusive max, so max = num_groups - 1 gives extent num_groups.
        Stmt result = For::make(wg, 0, num_groups - 1, ForType::GPUThread,
                                Partition::Never, device_api, body, GPUVectorScope::Register, -1);
        // Re-emit the lifted storage at block level (outermost peeled first).
        for (auto it = lifted.rbegin(); it != lifted.rend(); ++it) {
            result = HoistedStorage::make(*it, result);
        }
        return result;
    }

    Stmt visit(const Acquire *op) override {
        if (active) {
            const Variable *v = op->semaphore.as<Variable>();
            auto it = v ? sema_map.find(v->name) : sema_map.end();
            if (it != sema_map.end()) {
                const BarrierInfo &b = it->second;
                Stmt body = mutate(op->body);
                if (!b.mbar_name.empty()) {
                    // F3 full-edge consumer wait: spin on the slot's mbarrier until the producer's
                    // cp.async copies complete (parity = (ko/N)&1, since the slot is reused every N
                    // iters and each completion flips the phase). No producer drain -> deep overlap.
                    // The try_wait.parity convention is ISA-ambiguous; HL_WG_MBAR_PFLIP toggles it
                    // empirically (start phase 0 vs 1) without a rebuild.
                    int pflip = get_env_variable("HL_WG_MBAR_PFLIP") == "1" ? 1 : 0;
                    Expr parity = (Variable::make(Int(32), ring_loop) / b.ring_n + pflip) % 2;
                    // Portable completion WAIT (M3b): the ring no longer hand-picks the mbarrier; it
                    // emits async_wait(CpAsyncGroup, WarpGroup, token, parity) and the S2 selector
                    // (lower_async_completions) lowers it to mbarrier_try_wait. The mbar slot array +
                    // init are still materialized in visit(Fork) below (a shared store -> the
                    // visibility barrier is generated by InjectThreadBarriers).
                    Stmt wait = Evaluate::make(Call::make(
                        Int(32), Call::async_wait,
                        {Expr((int)CompletionKind::CpAsyncGroup), Expr((int)SyncScope::WarpGroup),
                         mbar_slot_ref(b), parity},
                        Call::Intrinsic));
                    return Block::make(wait, body);
                }
                Stmt wait = emit_barrier(b, /*wait*/ 0);
                if (b.is_empty) {
                    // Slots start free: skip the first N empty-waits or iter 0 deadlocks.
                    wait = IfThenElse::make(Variable::make(Int(32), ring_loop) >= b.ring_n, wait);
                }
                return Block::make(wait, body);
            }
        }
        return IRMutator::visit(op);
    }

    Stmt visit(const Evaluate *op) override {
        const Call *c = op->value.as<Call>();
        if (c && c->name == "halide_semaphore_init" && !c->args.empty()) {
            const Variable *v = c->args[0].as<Variable>();
            if (v && sema_belongs_to_warpspec_ring(v->name, env)) {
                return Evaluate::make(0);
            }
        }
        if (active && c && c->name == "halide_semaphore_release" && !c->args.empty()) {
            const Variable *v = c->args[0].as<Variable>();
            auto it = v ? sema_map.find(v->name) : sema_map.end();
            if (it != sema_map.end()) {
                const BarrierInfo &b = it->second;
                if (!b.mbar_name.empty()) {
                    // F3 full-edge producer arrive: a DEFERRED cp.async-completion arrive on the
                    // slot's mbarrier. The producer never waits its own copies (no commit/wait_group);
                    // their completion decrements the mbarrier the consumer polls. -> copies stay in
                    // flight across slots = the deep pipeline. Portable form (M3b): emit
                    // async_issue(CpAsyncGroup, token); the S2 selector lowers it to
                    // cp_async_mbarrier_arrive.
                    return Evaluate::make(Call::make(
                        Int(32), Call::async_issue,
                        {Expr((int)CompletionKind::CpAsyncGroup), mbar_slot_ref(b)},
                        Call::Intrinsic));
                }
                return emit_barrier(b, /*arrive*/ 1);
            }
        }
        return IRMutator::visit(op);
    }

public:
    LowerGPUWarpAsyncFork(const std::map<std::string, Function> &env, int warp_size)
        : env(env), fork_fuse(get_env_variable("HL_GPU_WARP_FORK_FUSE") != "0"), warp_size(warp_size),
          mbar(get_env_variable("HL_WG_MBAR") == "1") {
    }
};

// Lower the target-independent sync_requirement markers (emitted by the barrier
// analysis) to concrete barrier mechanisms, by (scope x target). Today only Block
// scope is emitted, lowered to a whole-CTA gpu_thread_barrier — byte-identical to
// emitting the barrier directly. Finer scopes (WarpGroup named barriers, Cluster
// barriers, async mbarrier) slot in here as the sync model grows; the requirement
// (who/where/what-memory) stays portable and codegen never sees a sync_requirement.
// See research/gpu_sync_model.md.
class LowerSyncRequirements : public IRMutator {
    using IRMutator::visit;

    Expr visit(const Call *op) override {
        if (op->is_intrinsic(Call::sync_requirement)) {
            internal_assert(!op->args.empty()) << "sync_requirement needs a scope.\n";
            auto scope = as_const_int(op->args[0]);
            internal_assert(scope) << "sync_requirement scope must be a constant.\n";
            switch ((SyncScope)*scope) {
            case SyncScope::Block:
                // Whole-CTA barrier; every GPU backend lowers gpu_thread_barrier.
                // Args: (scope, fence).
                internal_assert(op->args.size() == 2)
                    << "Block sync_requirement expects (scope, fence).\n";
                return Call::make(Int(32), Call::gpu_thread_barrier,
                                  {mutate(op->args[1])}, Call::Intrinsic);
            case SyncScope::WarpGroup:
                // Partial/named CTA barrier: only `count` threads (the producing +
                // consuming warp groups of an async edge) rendezvous on named barrier
                // `id`. mode 0 = wait, 1 = arrive. Args: (scope, id, count, mode). The
                // mbarrier swap for sm_90 TMA drops in here, at the same seam.
                internal_assert(op->args.size() == 4)
                    << "WarpGroup sync_requirement expects (scope, id, count, mode).\n";
                return Call::make(Int(32), Call::gpu_named_barrier,
                                  {mutate(op->args[1]), mutate(op->args[2]), mutate(op->args[3])},
                                  Call::Intrinsic);
            default:
                internal_error
                    << "lower_sync_requirements: SyncScope " << *scope
                    << " not yet handled by the mechanism selector.\n";
                return op;
            }
        }
        return IRMutator::visit(op);
    }
};

}  // namespace

Stmt inject_gpu_warp_specialization(Stmt s, const std::map<std::string, Function> &env) {
    return InjectGPUWarpSpecialization(env)(s);
}

Stmt lower_gpu_warp_async(Stmt s, const std::map<std::string, Function> &env, const Target &t) {
    return LowerGPUWarpAsyncFork(env, t.warp_size())(s);
}

namespace {
// The async-completion selector (rearch S2). Lowers Call::async_issue / async_wait to a concrete
// completion mechanism by (CompletionKind x target). Today: CpAsyncBulk -> mbarrier transaction
// (sm_90), reusing the device intrinsics mbarrier_arrive_expect_tx (T1) and mbarrier_try_wait (F3).
// The mbarrier's shared alloc + init are materialized by the EMITTING recognizer (as a shared
// store), so the transitive init->use visibility barrier is generated by InjectThreadBarriers, not
// here. CpAsyncGroup / WgmmaGroup completions fold into the existing commit/wait_group paths and are
// left for a later step (the cp.async group completion is implicit in codegen today).
class LowerAsyncCompletions : public IRMutator {
    using IRMutator::visit;

    Expr visit(const Call *op) override {
        if (op->is_intrinsic(Call::async_issue)) {
            auto kind = as_const_int(op->args[0]);
            internal_assert(kind) << "async_issue kind must be a constant.\n";
            if ((CompletionKind)*kind == CompletionKind::CpAsyncBulk) {
                // async_issue(CpAsyncBulk, token, bytes): the issuing thread arms the mbarrier with
                // the expected transaction byte count of the in-flight bulk copy.
                internal_assert(op->args.size() == 3)
                    << "async_issue(CpAsyncBulk) expects (kind, token, bytes).\n";
                return Call::make(Int(32), "mbarrier_arrive_expect_tx",
                                  {mutate(op->args[1]), mutate(op->args[2])}, Call::Intrinsic);
            }
            if ((CompletionKind)*kind == CompletionKind::CpAsyncGroup) {
                // async_issue(CpAsyncGroup, token): a DEFERRED cp.async-completion arrive on the
                // token's mbarrier (the warp-spec ring's producer side). The producer never waits its
                // own copies; their completion decrements the mbarrier the consumer polls. No byte
                // count -- the cp.async group, not a bulk transaction, drives completion.
                internal_assert(op->args.size() == 2)
                    << "async_issue(CpAsyncGroup) expects (kind, token).\n";
                return Call::make(Int(32), "cp_async_mbarrier_arrive",
                                  {mutate(op->args[1])}, Call::Intrinsic);
            }
        }
        if (op->is_intrinsic(Call::async_wait)) {
            auto kind = as_const_int(op->args[0]);
            internal_assert(kind) << "async_wait kind must be a constant.\n";
            if ((CompletionKind)*kind == CompletionKind::CpAsyncBulk ||
                (CompletionKind)*kind == CompletionKind::CpAsyncGroup) {
                // async_wait(kind, scope, token, parity): every consumer thread spins on the mbarrier
                // phase until completion (bulk bytes landed, or the cp.async group's copies completed).
                // Same mechanism (mbarrier try_wait.parity) for both kinds; only the producer-side
                // arrive differs (expect_tx vs cp.async arrive).
                internal_assert(op->args.size() == 4)
                    << "async_wait expects (kind, scope, token, parity).\n";
                return Call::make(Int(32), "mbarrier_try_wait",
                                  {mutate(op->args[2]), mutate(op->args[3])}, Call::Intrinsic);
            }
        }
        return IRMutator::visit(op);
    }
};

// Recognize a block-scope tile copy (a cooperative global->shared staging fill) and rewrite
// it to a TMA bulk-tensor load completed by an mbarrier (sm_90). See inject_tma_copies (.h)
// and fusegpu_rearch_plan.md C.3/C.4a/M4.
class InjectTmaCopies : public IRMutator {
    std::set<std::string> shared_allocs;  // names allocated in GPUShared in scope
    std::map<std::string, SwizzleLayout> shared_swizzle;  // store_in swizzle per shared alloc
public:
    // Ring mbarriers (buffer names) we wired a TMA producer onto: their mbarrier_init arrival count
    // (set for cp.async = warp width) must be patched to 1 (TMA's single expect_tx arrive).
    std::set<std::string> tma_mbars;
private:
    // Tensor-map lets to wrap around the current gpu_block (host scope -> kernel arg).
    struct MapLet {
        std::string var;    // tensor-map variable name (referenced by tma_load_2d)
        std::string src;    // global source buffer name (its .buffer is the descriptor input)
        Expr box0, box1;    // tile inner/outer extents (descriptor box)
        int swizzle;        // shared swizzle in bytes (0/32/64/128) -- must match the wgmma descriptor
    };
    std::vector<MapLet> pending_maps;
    using IRMutator::visit;

    // The shared swizzle in BYTES (0/32/64/128) the tensor map must apply so its shared layout
    // matches the consumer's store_in swizzle (and, for a wgmma operand, the descriptor swizzle).
    // SwizzleLayout::bits is 1/2/3 for 32/64/128 B (resolve_swizzle); 0 = none.
    static int swizzle_bytes(const SwizzleLayout &s) {
        return s.defined() ? (1 << (s.bits + 4)) : 0;
    }

    Stmt visit(const Allocate *op) override {
        bool shared = op->memory_type == MemoryType::GPUShared;
        if (shared) {
            shared_allocs.insert(op->name);
            shared_swizzle[op->name] = op->swizzle;
            if (get_env_variable("HL_TMA_DEBUG") == "3") {
                std::cerr << "ALLOC shared=" << op->name << " swizzle.defined=" << op->swizzle.defined()
                          << " bytes=" << swizzle_bytes(op->swizzle) << "\n";
            }
        }
        Stmt s = IRMutator::visit(op);
        if (shared) {
            shared_allocs.erase(op->name);
            shared_swizzle.erase(op->name);
        }
        return s;
    }

    Stmt visit(const For *op) override {
        if (!ends_with(op->name, gpu_block_name(0))) {
            return IRMutator::visit(op);
        }
        // At the innermost gpu_block: rewrite any TMA-eligible producers inside, then wrap the
        // block in one host tensor-map let per rewritten producer (closure -> kernel arg).
        ScopedValue<std::vector<MapLet>> save(pending_maps, {});
        Stmt body = mutate(op->body);
        Stmt block = For::make(op->name, op->min, op->max, op->for_type, op->partition_policy,
                               op->device_api, body, op->realization, op->warps_per_group);
        // The tensor-map descriptor is loop-invariant (one map per producer FUNC, keyed by
        // src/box/swizzle -- all func-level). A software-pipelined producer is duplicated into
        // prologue/steady/epilogue copies, so the same `<func>.tma_map` can be pushed several times;
        // emit one LetStmt per UNIQUE descriptor name (define once, reference many) rather than
        // shadowing copies that violate global name-uniqueness.
        std::set<std::string> emitted_maps;
        for (auto it = pending_maps.rbegin(); it != pending_maps.rend(); ++it) {
            if (!emitted_maps.insert(it->var).second) {
                continue;
            }
            // box dims are descriptor parameters (inner contiguous, then outer); swizzle in bytes
            // (0 = NONE) = the consumer's store_in swizzle, so the TMA shared layout matches.
            Expr buf = Variable::make(type_of<halide_buffer_t *>(), it->src + ".buffer");
            Expr call = Call::make(UInt(64), "halide_cuda_tensor_map",
                                   {buf, it->box0, it->box1, it->swizzle}, Call::Extern);
            block = LetStmt::make(it->var, call, block);
        }
        return block;
    }

    // Match `for (a) { for (b) { As[dst] = Src[src] } }`: a 2D tile copy. The TMA descriptor's
    // dim 0 is the CONTIGUOUS (unit-stride) axis -- which is NOT necessarily the inner loop: a
    // wgmma A operand is stored M-outer / K-contiguous, so the K loop (the contiguous one) is the
    // OUTER loop in the natural nest. We therefore key box0/coordX on whichever loop strides the
    // source with stride 1, not on loop nesting. Returns true + fills the fields on a match.
    struct TileCopy {
        std::string dst, src;        // shared dst buffer, global src buffer
        Expr box0, box1;             // contiguous (dim 0) tile extent, then outer (dim 1) extent
        Expr coordX, coordY;         // tile origin: contiguous-axis coord (c0), then outer-axis (c1)
        Expr dst_slot;               // dst element offset of the tile origin (ring slot; 0 if non-ring)
        Type elem;                   // element type of the copy
    };
    // Source stride of `var` in `index` (the change in index per unit step of var); a unit result
    // marks the contiguous axis.
    static Expr loop_source_stride(const std::string &var, const Expr &index) {
        return simplify(substitute(var, Variable::make(Int(32), var) + 1, index) - index);
    }
    // Strip leading LetStmts and descend into Block.first until we reach the 2D copy's outer For.
    // A ring-buffered / .async() producer wraps the For-nest in lets (ring-slot offsets) and a Block
    // (the trailing ring-completion marker), so the body isn't a bare For.
    //
    // DESIGN DEBT (2026-06-26): this structural peeling is brittle -- it depends on the exact IR shape
    // the ring/async lowering emits. The principled alternative is to drive TMA eligibility + the ring
    // wiring from SCHEDULE metadata (the Func is store_in(GPUShared), a direct copy, has a swizzle, and
    // the ring/async/warp-group split are all schedule facts), not from re-matching the lowered nest.
    // Revisit when the recognizer is consolidated. See perf_ladder.md R4.
    static Stmt peel_to_for_nest(Stmt s) {
        while (true) {
            if (const LetStmt *l = s.as<LetStmt>()) { s = l->body; continue; }
            if (const Block *b = s.as<Block>()) {
                // The copy For-nest is the Block half that contains a For (the other is the marker).
                if (b->first.as<For>() || b->first.as<LetStmt>() || b->first.as<Block>()) { s = b->first; continue; }
                if (b->rest.defined() && (b->rest.as<For>() || b->rest.as<LetStmt>())) { s = b->rest; continue; }
            }
            return s;
        }
    }
    static bool match_tile_copy(const Stmt &produce_body, const std::string &name, TileCopy *tc) {
        const bool dbg = get_env_variable("HL_TMA_DEBUG") == "2";
        const For *lo = peel_to_for_nest(produce_body).as<For>();
        if (!lo) {
            if (dbg) debug(0) << "TMA_NOMATCH " << name << ": peeled body not For (is "
                              << (produce_body.as<LetStmt>() ? "LetStmt" : produce_body.as<Block>() ? "Block" : "other") << ")\n";
            return false;
        }
        const For *li = peel_to_for_nest(lo->body).as<For>();
        if (!li) {
            if (dbg) debug(0) << "TMA_NOMATCH " << name << ": lo->body not For\n";
            return false;
        }
        const Store *st = peel_to_for_nest(li->body).as<Store>();
        if (!st || st->name != name) {
            if (dbg) debug(0) << "TMA_NOMATCH " << name << ": li->body not Store-to-name (store="
                              << (st ? st->name : "<none>") << ")\n";
            return false;
        }
        const Load *ld = st->value.as<Load>();  // direct copy (no cast): As(k,m) = Src(k,m)
        if (!ld) {
            return false;
        }
        // Both loop vars must actually index the copy (a real 2D tile).
        if (!expr_uses_var(ld->index, li->name) || !expr_uses_var(ld->index, lo->name)) {
            return false;
        }
        // The contiguous (unit-stride) loop becomes the descriptor's dim 0; the other is dim 1.
        const For *contig = nullptr, *outer = nullptr;
        if (is_const_one(loop_source_stride(li->name, ld->index))) {
            contig = li;
            outer = lo;
        } else if (is_const_one(loop_source_stride(lo->name, ld->index))) {
            contig = lo;
            outer = li;
        } else {
            return false;  // no unit-stride axis -> not a TMA-shaped tile copy
        }
        // Which source dim is contiguous: its min appears with coefficient 1 in the index's
        // constant (min-subtraction) term `-(sum_d src.min.d * src.stride.d)`.
        Expr base = simplify(substitute({{li->name, Expr(0)}, {lo->name, Expr(0)}}, ld->index));
        int cdim = 0;
        for (int d = 0; d < 2; d++) {
            std::string md = ld->name + ".min." + std::to_string(d);
            if (!expr_uses_var(base, md)) {
                continue;
            }
            Expr coeff = simplify(substitute(md, Expr(1), base) - substitute(md, Expr(0), base));
            if (auto c = as_const_int(coeff)) {
                if (*c == -1 || *c == 1) {
                    cdim = d;
                }
            }
        }
        int odim = 1 - cdim;
        Expr cmin = Variable::make(Int(32), ld->name + ".min." + std::to_string(cdim));
        Expr omin = Variable::make(Int(32), ld->name + ".min." + std::to_string(odim));
        tc->dst = name;
        tc->src = ld->name;
        tc->box0 = contig->extent();
        tc->box1 = outer->extent();
        tc->coordX = simplify(contig->min - cmin);  // contiguous-axis origin (c0)
        tc->coordY = simplify(outer->min - omin);   // outer-axis origin (c1)
        tc->elem = st->value.type();
        // Ring case: the destination is a per-ko shared SLOT (ring_buffer rotation). The slot base =
        // the store index at the tile origin (swizzle(0)=0), so dst = dst-buffer + that offset. For a
        // non-ring tile this simplifies to 0 (the alloc base) -- NFC for R3.
        tc->dst_slot = simplify(substitute({{contig->name, contig->min}, {outer->name, outer->min}},
                                           st->index));
        if (get_env_variable("HL_TMA_DEBUG") == "1") {
            debug(0) << "TMA_DEBUG name=" << name << " src=" << ld->name
                     << " contig_loop=" << contig->name << " cdim=" << cdim
                     << "\n  box0(contig)=" << tc->box0 << " box1(outer)=" << tc->box1
                     << "\n  coordX(c0)=" << tc->coordX << " coordY(c1)=" << tc->coordY << "\n";
        }
        return true;
    }

    // A ring/.async() producer carries its own completion marker (`async_issue(kind, full_mbar[ko])`).
    // Find that mbar Expr (undefined => non-ring producer). For the ring case, TMA must arrive on THIS
    // mbar (so the consumer's try_wait wakes), not a private one.
    static Expr find_ring_mbar(const Stmt &body) {
        class Finder : public IRVisitor {
            using IRVisitor::visit;
            void visit(const Call *op) override {
                if (op->is_intrinsic(Call::async_issue) && op->args.size() >= 2 && !found.defined()) {
                    found = op->args[1];
                }
                IRVisitor::visit(op);
            }
        public:
            Expr found;
        } f;
        body.accept(&f);
        return f.found;
    }

    Stmt visit(const ProducerConsumer *op) override {
        if (op->is_producer && get_env_variable("HL_TMA_DEBUG") == "3") {
            debug(0) << "TMA_BODY producer=" << op->name << " shared=" << shared_allocs.count(op->name)
                     << " swz=" << swizzle_bytes(shared_swizzle[op->name]) << ":\n" << op->body << "\n---\n";
        }
        if (!op->is_producer || !shared_allocs.count(op->name)) {
            return IRMutator::visit(op);
        }
        TileCopy tc;
        if (!match_tile_copy(op->body, op->name, &tc)) {
            return IRMutator::visit(op);
        }
        // A ring-buffered / .async() producer reuses the ring's full_mbar completion; the TMA must
        // arrive on it (gap 1) and write the per-ko ring SLOT (gap 2). This wiring is WIP (the swizzle
        // recovery, gap 3, is still open), so it is gated -- by default keep the correct cp.async fill.
        Expr ring_mbar = find_ring_mbar(op->body);
        if (ring_mbar.defined() && get_env_variable("HL_WG_TMA_RING") != "1") {
            return IRMutator::visit(op);
        }
        // Tile coords (element offsets from the source buffer's origin) are computed contiguity-
        // aware in match_tile_copy: coordX = dim-0 (contiguous) axis, coordY = dim-1 (outer) axis.
        Expr coordX = tc.coordX;
        Expr coordY = tc.coordY;
        Expr bytes = simplify(tc.box0 * tc.box1 * (tc.elem.bits() / 8));

        std::string mbar = op->name + ".tma_mbar";
        std::string tmap = op->name + ".tma_map";
        int swz = swizzle_bytes(shared_swizzle[op->name]);
        // The ring storage transform bakes the swizzle into the scalar store INDEX and clears the
        // Allocate.swizzle, so a ring operand loses its swizzle here (swz=0) even though the wgmma
        // descriptor still expects it -> the TMA would write unswizzled. HL_WG_TMA_SWZ overrides the
        // tensor-map swizzle to confirm/repair this until the swizzle is kept as an attribute.
        {
            std::string ov = get_env_variable("HL_WG_TMA_SWZ");
            if (!ov.empty() && swz == 0) swz = std::atoi(ov.c_str());
        }
        pending_maps.push_back({tmap, tc.src, tc.box0, tc.box1, swz});

        Expr mbar_ref = Load::make(UInt(64), mbar, 0, Buffer<>{}, Parameter{}, const_true(),
                                   ModulusRemainder{});
        Expr dst_ref = Load::make(tc.elem, tc.dst, 0, Buffer<>{}, Parameter{}, const_true(),
                                  ModulusRemainder{});
        Expr map_var = Variable::make(UInt(64), tmap);

        // RING case (gated): TMA writes the per-ko slot and arrives on the ring's OWN full_mbar (the
        // marker we found). We emit only expect_tx + the TMA load -- the ring already init's the mbar
        // and the consumer warp group already try_waits it; no private mbar / init / wait. The
        // CompletionKind upgrade (CpAsyncGroup -> CpAsyncBulk) makes lower_async_completions emit the
        // transaction-completion expect_tx instead of a cp.async-group arrive.
        if (ring_mbar.defined()) {
            // Record the ring mbar so patch_tma_mbar_counts can fix its arrival count (TMA arrives 1).
            {
                class NameOf : public IRVisitor {
                    using IRVisitor::visit;
                    void visit(const Load *l) override { if (name.empty()) name = l->name; IRVisitor::visit(l); }
                public: std::string name;
                } n;
                ring_mbar.accept(&n);
                if (!n.name.empty()) tma_mbars.insert(n.name);
            }
            Expr slot_dst = Load::make(tc.elem, tc.dst, tc.dst_slot, Buffer<>{}, Parameter{},
                                       const_true(), ModulusRemainder{});
            Stmt rissue = Evaluate::make(Call::make(Int(32), Call::async_issue,
                                                    {Expr((int)CompletionKind::CpAsyncBulk), ring_mbar, bytes},
                                                    Call::Intrinsic));
            Stmt rload = Evaluate::make(Call::make(Int(32), "tma_load_2d",
                                                   {slot_dst, map_var, coordX, coordY, ring_mbar},
                                                   Call::Intrinsic));
            return ProducerConsumer::make(op->name, true, Block::make(rissue, rload));
        }

        // Single-slot mbarrier, one expected arrive (the elected thread's expect_tx). The TMA
        // intrinsics self-elect thread 0 in codegen, so these run block-level (no IR thread guard);
        // InjectThreadBarriers GENERATES the init->use + produce->consume Block barriers (mbarrier_init
        // and tma_load_2d both register as shared stores).
        Stmt init = Evaluate::make(Call::make(Int(32), "mbarrier_init",
                                              {mbar_ref, Expr(1), Expr(1)}, Call::Intrinsic));
        Stmt issue = Evaluate::make(Call::make(Int(32), Call::async_issue,
                                               {Expr((int)CompletionKind::CpAsyncBulk), mbar_ref, bytes},
                                               Call::Intrinsic));
        Stmt load = Evaluate::make(Call::make(Int(32), "tma_load_2d",
                                              {dst_ref, map_var, coordX, coordY, mbar_ref},
                                              Call::Intrinsic));
        Stmt wait = Evaluate::make(Call::make(Int(32), Call::async_wait,
                                              {Expr((int)CompletionKind::CpAsyncBulk),
                                               Expr((int)SyncScope::Block), mbar_ref, Expr(0)},
                                              Call::Intrinsic));
        Stmt seq = Block::make({init, issue, load, wait});
        seq = Allocate::make(mbar, UInt(64), MemoryType::GPUShared, {Expr(1)}, const_true(), seq);
        return ProducerConsumer::make(op->name, true, seq);
    }
};

// After TMA is wired onto ring mbarriers, patch those mbarriers' init arrival count to 1: the ring
// emitted the count for a cp.async producer (every warp lane arrives) BEFORE TMA existed, but a TMA
// producer arrives exactly ONCE (expect_tx) -- a count of 32 would deadlock the consumer's try_wait.
// The init is one flattened call `mbarrier_init(base0,ring_n0,count0, base1,ring_n1,count1, ...)`.
class PatchTmaMbarCounts : public IRMutator {
    const std::set<std::string> &tma_mbars;
    using IRMutator::visit;
    static std::string buffer_of(const Expr &e) {
        class NameOf : public IRVisitor {
            using IRVisitor::visit;
            void visit(const Load *l) override { if (name.empty()) name = l->name; IRVisitor::visit(l); }
        public: std::string name;
        } n;
        e.accept(&n);
        return n.name;
    }
    Expr visit(const Call *op) override {
        if (op->name == "mbarrier_init") {
            std::vector<Expr> args = op->args;
            for (size_t i = 0; i + 2 < args.size(); i += 3) {
                if (tma_mbars.count(buffer_of(args[i]))) {
                    args[i + 2] = Expr(1);
                }
            }
            return Call::make(op->type, op->name, args, op->call_type);
        }
        return IRMutator::visit(op);
    }
public:
    explicit PatchTmaMbarCounts(const std::set<std::string> &m) : tma_mbars(m) {}
};
}  // namespace

Stmt inject_tma_copies(Stmt s, const Target &t) {
    if (!t.has_feature(Target::CUDACapability90) || get_env_variable("HL_WG_TMA") != "1") {
        return s;
    }
    InjectTmaCopies injector;
    s = injector(s);
    if (!injector.tma_mbars.empty()) {
        s = PatchTmaMbarCounts(injector.tma_mbars)(s);
    }
    return s;
}

Stmt lower_async_completions(Stmt s, const Target &t) {
    // The mbarrier completion mechanism is sm_90+. On other targets, async completions fall back to
    // commit/wait_group paths (handled elsewhere); leave the markers for now.
    if (!t.has_feature(Target::CUDACapability90)) {
        return s;
    }
    return LowerAsyncCompletions()(s);
}

namespace {
// Stamp each VectorReduce with the realization of its innermost enclosing GPU collective
// loop (gpu_warps -> WarpGroup, gpu_lanes -> Warp), BEFORE thread-loop fusion erases that
// scope. Pure annotation: the unified recognizer then reads the tag off the vector node at
// codegen (uniform with dp4a), so no extraction pass is needed. See gpu_recognizer_design.md.
class TagGPUVectorScope : public IRMutator {
    GPUVectorScope scope = GPUVectorScope::Register;
    using IRMutator::visit;

    Stmt visit(const For *op) override {
        ScopedValue<GPUVectorScope> s(
            scope, op->realization != GPUVectorScope::Register ? op->realization : scope);
        return IRMutator::visit(op);
    }

    Expr visit(const VectorReduce *op) override {
        Expr value = mutate(op->value);
        return VectorReduce::make(op->op, std::move(value), op->type.lanes(), scope);
    }
};
}  // namespace

Stmt fuse_gpu_thread_loops(Stmt s, const Target &t, const std::map<std::string, Function> &env) {
    // Tag vectorized collectives with their realization scope before fusion erases the
    // gpu_warps/gpu_lanes loops; codegen reads the tag to decompose into the primitive.
    s = TagGPUVectorScope()(s);
    // NormalizeIfStatements pushes the predicates between GPU blocks
    // into the innermost GPU block. FuseGPUThreadLoops would then
    // merge the predicate into the merged GPU thread.
    s = NormalizeIfStatements()(s);
    s = FuseGPUThreadLoops(t.warp_size(), env)(s);
    s = ZeroGPULoopMins()(s);
    // Lower the schedule-derived sync_requirement markers to concrete barriers
    // (scope x target). Block scope -> whole-CTA gpu_thread_barrier today.
    s = LowerSyncRequirements()(s);
    return s;
}

}  // namespace Internal
}  // namespace Halide
