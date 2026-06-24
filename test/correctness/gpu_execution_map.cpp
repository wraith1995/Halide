// Gate test for the execution-mapping model (M1; research/exec_mapping_model.md §5).
// Pure IR analysis, no GPU. Constructs small Stmts reproducing each existing GPU
// thread/warp-group structure and asserts active()/scope()/count()/join() give the
// hand-checked answer. M1 ships only when G1..G10 pass.

#include "Halide.h"

#include <cstdio>
#include <string>

using namespace Halide;
using namespace Halide::Internal;

namespace {

int failures = 0;

void check(bool ok, const std::string &msg) {
    if (!ok) {
        printf("  FAIL: %s\n", msg.c_str());
        failures++;
    }
}

void check_count(const ExecMap &m, const ActiveSet &a, int expected, const std::string &msg) {
    Expr c = simplify(m.count(a));
    auto ci = as_const_int(c);
    if (!ci || *ci != expected) {
        printf("  FAIL: %s -- count = %s, expected %d\n", msg.c_str(),
               ci ? std::to_string(*ci).c_str() : "<non-const>", expected);
        failures++;
    }
}

void check_scope(const ExecMap &m, const ActiveSet &a, ExecScope expected, const std::string &msg) {
    ExecScope s = m.scope(a);
    if (s != expected) {
        printf("  FAIL: %s -- scope = %s, expected %s\n", msg.c_str(),
               exec_scope_name(s), exec_scope_name(expected));
        failures++;
    }
}

// A walker that descends to the (first) sentinel Evaluate, tracking the ExecMap, and
// captures the active set there. Mirrors how a consuming pass (InjectThreadBarriers)
// uses the model.
struct LeafWalk : public IRVisitor {
    ExecMap &m;
    ActiveSet out;
    bool done = false;
    explicit LeafWalk(ExecMap &m)
        : m(m) {
    }
    using IRVisitor::visit;
    void visit(const For *op) override {
        m.enter_for(op);
        IRVisitor::visit(op);
        m.pop();
    }
    void visit(const IfThenElse *op) override {
        m.enter_if(op->condition);
        op->then_case.accept(this);
        m.pop();
        // deliberately skip else_case (the guard does not hold there)
    }
    void visit(const Evaluate *op) override {
        if (!done) {
            out = m.current();
            done = true;
        }
    }
};

ActiveSet leaf_active(ExecMap &m, const Stmt &kernel) {
    LeafWalk w(m);
    kernel.accept(&w);
    return w.out;
}

// Sentinel leaf (captured by LeafWalk); a side-effecting call so it is not simplified.
Stmt leaf() {
    return Evaluate::make(Call::make(Int(32), "sentinel", {}, Call::Extern));
}
Stmt other() {
    return Evaluate::make(Call::make(Int(32), "other", {}, Call::Extern));
}

Expr tvar(int i) {
    return Variable::make(Int(32), gpu_thread_name(i));
}

Stmt thread_for(int i, int lo, int hi, Stmt body, int wpg = -1) {
    return For::make(gpu_thread_name(i), lo, hi, ForType::GPUThread, Partition::Never,
                     DeviceAPI::CUDA, std::move(body), GPUVectorScope::Register, wpg);
}

Stmt gpu_block(Stmt body) {
    return For::make(gpu_block_name(0), 0, 0, ForType::GPUBlock, Partition::Never,
                     DeviceAPI::CUDA, std::move(body), GPUVectorScope::Register, -1);
}

}  // namespace

int main(int argc, char **argv) {
    // G1: block-level serial statement (no enclosing thread loop) runs on ALL lanes.
    //     The sibling thread loop supplies the block geometry (Tx=128).
    {
        Stmt k = gpu_block(Block::make(leaf(), thread_for(0, 0, 127, other())));
        ExecMap m(k);
        ActiveSet a = leaf_active(m, k);
        check_scope(m, a, ExecScope::Block, "G1 block-serial scope");
        check_count(m, a, 128, "G1 block-serial count");
    }

    // G2: inside For GPUThread tx[0,127] -- every lane runs one iteration => whole block.
    {
        Stmt k = gpu_block(thread_for(0, 0, 127, leaf()));
        ExecMap m(k);
        ActiveSet a = leaf_active(m, k);
        check_scope(m, a, ExecScope::Block, "G2 thread-loop scope");
        check_count(m, a, 128, "G2 thread-loop count");
    }

    // G3: if(tx==0) => a single lane => Thread, no barrier.
    {
        Stmt guarded = IfThenElse::make(tvar(0) == 0, leaf());
        Stmt k = gpu_block(Block::make(guarded, thread_for(0, 0, 127, other())));
        ExecMap m(k);
        ActiveSet a = leaf_active(m, k);
        check_scope(m, a, ExecScope::Thread, "G3 single-thread scope");
        check_count(m, a, 1, "G3 single-thread count");
    }

    // G4: a gpu_warps collective axis (warps_per_group=4 => 128 lanes/group), branch
    //     wg==1 => one warp group.
    {
        const std::string wg = "wg_coll" + gpu_thread_name(2);
        Expr wgv = Variable::make(Int(32), wg);
        Stmt body = IfThenElse::make(wgv == 1, leaf());
        Stmt wgfor = For::make(wg, 0, 1, ForType::GPUThread, Partition::Never, DeviceAPI::CUDA,
                               body, GPUVectorScope::Register, /*warps_per_group*/ 4);
        Stmt k = gpu_block(wgfor);
        ExecMap m(k);
        ActiveSet a = leaf_active(m, k);
        check_scope(m, a, ExecScope::WarpGroup, "G4 warp-group scope");
        check_count(m, a, 128, "G4 warp-group count");
    }

    // G5: partition_warp_groups flat-id peel: if(fv>=128 && fv<256) on a 1-D block
    //     (Tx=256) decodes onto the single thread axis => [128,255].
    {
        Expr fv = Variable::make(Int(32), "warp_flat" + gpu_thread_name(0));
        Expr cond = (fv >= 128) && (fv < 256);
        Stmt guarded = IfThenElse::make(cond, leaf());
        Stmt k = gpu_block(Block::make(guarded, thread_for(0, 0, 255, other())));
        ExecMap m(k);
        ActiveSet a = leaf_active(m, k);
        check_scope(m, a, ExecScope::WarpGroup, "G5 flat-peel scope");
        check_count(m, a, 128, "G5 flat-peel count");
    }

    // G6: warp-spec ring producer subset: For(wg,0,1,wpg=-1) + if(wg==0) + inner 128
    //     threads => the producer warp group (128 lanes).
    {
        const std::string wg = "warp_group" + gpu_thread_name(2);
        Expr wgv = Variable::make(Int(32), wg);
        Stmt inner = thread_for(0, 0, 127, leaf());
        Stmt body = IfThenElse::make(wgv == 0, inner);
        Stmt wgfor = For::make(wg, 0, 1, ForType::GPUThread, Partition::Never, DeviceAPI::CUDA,
                               body, GPUVectorScope::Register, /*wpg*/ -1);
        Stmt k = gpu_block(wgfor);
        ExecMap m(k);
        ActiveSet a = leaf_active(m, k);
        check_scope(m, a, ExecScope::WarpGroup, "G6 ring-subset scope");
        check_count(m, a, 128, "G6 ring-subset count");
    }

    // G7: join({tx==0}, whole) == whole block => Block.
    {
        Stmt k = gpu_block(thread_for(0, 0, 127, leaf()));
        ExecMap m(k);
        m.enter_if(tvar(0) == 0);
        ActiveSet tid0 = m.current();
        m.pop();
        ActiveSet whole = ActiveSet::whole_block();
        ActiveSet j = ExecMap::join(tid0, whole);
        check_scope(m, j, ExecScope::Block, "G7 join(tid0, whole) scope");
        check_count(m, j, 128, "G7 join(tid0, whole) count");
    }

    // G8: join(WG0, WG1) over a 2-group axis == both groups == whole block => Block.
    {
        const std::string wg = "wg_coll" + gpu_thread_name(2);
        Expr wgv = Variable::make(Int(32), wg);
        Stmt wgfor = For::make(wg, 0, 1, ForType::GPUThread, Partition::Never, DeviceAPI::CUDA,
                               leaf(), GPUVectorScope::Register, /*wpg*/ 4);
        Stmt k = gpu_block(wgfor);
        ExecMap m(k);
        m.enter_if(wgv == 0);
        ActiveSet g0 = m.current();
        m.pop();
        m.enter_if(wgv == 1);
        ActiveSet g1 = m.current();
        m.pop();
        ActiveSet j = ExecMap::join(g0, g1);
        check_scope(m, j, ExecScope::Block, "G8 join(WG0, WG1) scope");
        check_count(m, j, 256, "G8 join(WG0, WG1) count");
    }

    // G9: same_single_lane({tx==0}, {tx==0}) == true (=> Thread, no barrier).
    {
        Stmt k = gpu_block(thread_for(0, 0, 127, leaf()));
        ExecMap m(k);
        m.enter_if(tvar(0) == 0);
        ActiveSet a = m.current();
        m.pop();
        m.enter_if(tvar(0) == 0);
        ActiveSet b = m.current();
        m.pop();
        check(m.same_single_lane(a, b), "G9 same_single_lane(tid0, tid0)");
        // and a DIFFERENT single lane is not the same
        m.enter_if(tvar(0) == 1);
        ActiveSet c = m.current();
        m.pop();
        check(!m.same_single_lane(a, c), "G9 same_single_lane(tid0, tid1) is false");
    }

    // G10: an unrecognized guard does NOT narrow (conservative: whole block, more sync).
    {
        Expr opaque = Variable::make(Bool(), "p");
        Stmt guarded = IfThenElse::make(opaque, leaf());
        Stmt k = gpu_block(Block::make(guarded, thread_for(0, 0, 127, other())));
        ExecMap m(k);
        check(!m.recognizes_guard(opaque), "G10 recognizes_guard(opaque) is false");
        ActiveSet a = leaf_active(m, k);
        check_scope(m, a, ExecScope::Block, "G10 unrecognized-guard scope (conservative)");
        check_count(m, a, 128, "G10 unrecognized-guard count");
    }

    if (failures) {
        printf("gpu_execution_map: %d check(s) FAILED\n", failures);
        return 1;
    }
    printf("Success!\n");
    return 0;
}
