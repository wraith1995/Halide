#ifndef HALIDE_EXPR_H
#define HALIDE_EXPR_H

/** \file
 * Base classes for Halide expressions (\ref Halide::Expr) and statements (\ref Halide::Internal::Stmt)
 */

#include <string>
#include <vector>

#include "IntrusivePtr.h"
#include "Type.h"

namespace Halide {

struct bfloat16_t;
struct float16_t;

namespace Internal {

class IRMutator;
class IRVisitor;

// Exprs, in order of strength. Code in IRMatch.h and the
// simplifier relies on this order for canonicalization of
// expressions, so you may need to update those modules if you
// change this list.
#define HALIDE_FOR_EACH_IR_EXPR(X) \
    X(IntImm)                      \
    X(UIntImm)                     \
    X(FloatImm)                    \
    X(StringImm)                   \
    X(Broadcast)                   \
    X(Cast)                        \
    X(Reinterpret)                 \
    X(Variable)                    \
    X(Add)                         \
    X(Sub)                         \
    X(Mod)                         \
    X(Mul)                         \
    X(Div)                         \
    X(Min)                         \
    X(Max)                         \
    X(EQ)                          \
    X(NE)                          \
    X(LT)                          \
    X(LE)                          \
    X(GT)                          \
    X(GE)                          \
    X(And)                         \
    X(Or)                          \
    X(Not)                         \
    X(Select)                      \
    X(Load)                        \
    X(Ramp)                        \
    X(Call)                        \
    X(Let)                         \
    X(Shuffle)                     \
    X(VectorReduce)

/* Stmts */
#define HALIDE_FOR_EACH_IR_STMT(X) \
    X(LetStmt)                     \
    X(AssertStmt)                  \
    X(ProducerConsumer)            \
    X(For)                         \
    X(Acquire)                     \
    X(Store)                       \
    X(Provide)                     \
    X(Allocate)                    \
    X(Free)                        \
    X(Realize)                     \
    X(Block)                       \
    X(Fork)                        \
    X(IfThenElse)                  \
    X(Evaluate)                    \
    X(Prefetch)                    \
    X(Atomic)                      \
    X(HoistedStorage)

#define HALIDE_FOR_EACH_IR_NODE(X) \
    HALIDE_FOR_EACH_IR_EXPR(X)     \
    HALIDE_FOR_EACH_IR_STMT(X)

/** All our IR node types get unique IDs for the purposes of RTTI */
enum class IRNodeType : uint8_t {
#define DECL_ENUM(X) X,
    HALIDE_FOR_EACH_IR_NODE(DECL_ENUM)
#undef DECL_ENUM
};

const char *IRNodeType_string(IRNodeType type);

constexpr IRNodeType StrongestExprNodeType = IRNodeType::VectorReduce;

/** The abstract base classes for a node in the Halide IR. */
struct IRNode {

    /** We use the visitor pattern to traverse IR nodes throughout the
     * compiler, so we have a virtual accept method which accepts
     * visitors.
     */
    virtual void accept(IRVisitor *v) const = 0;
    IRNode(IRNodeType t)
        : node_type(t) {
    }
    virtual ~IRNode() = default;

    /** These classes are all managed with intrusive reference
     * counting, so we also track a reference count. It's mutable
     * so that we can do reference counting even through const
     * references to IR nodes.
     */
    mutable RefCount ref_count;

    /** Each IR node subclass has a unique identifier. We can compare
     * these values to do runtime type identification. We don't
     * compile with rtti because that injects run-time type
     * identification stuff everywhere (and often breaks when linking
     * external libraries compiled without it), and we only want it
     * for IR nodes. One might want to put this value in the vtable,
     * but that adds another level of indirection, and for Exprs we
     * have 32 free bits in between the ref count and the Type
     * anyway, so this doesn't increase the memory footprint of an IR node.
     */
    IRNodeType node_type;
};

template<>
inline RefCount &ref_count<IRNode>(const IRNode *t) noexcept {
    return t->ref_count;
}

template<>
inline void destroy<IRNode>(const IRNode *t) {
    delete t;
}

/** IR nodes are split into expressions and statements. These are
   similar to expressions and statements in C - expressions
   represent some value and have some type (e.g. x + 3), and
   statements are side-effecting pieces of code that do not
   represent a value (e.g. assert(x > 3)) */

/** A base class for statement nodes. They have no properties or
   methods beyond base IR nodes for now. */
struct BaseStmtNode : public IRNode {
    BaseStmtNode(IRNodeType t)
        : IRNode(t) {
    }
    virtual Stmt mutate_stmt(IRMutator *v) const = 0;
};

/** A base class for expression nodes. They all contain their types
 * (e.g. Int(32), Float(32)) */
struct BaseExprNode : public IRNode {
    BaseExprNode(IRNodeType t)
        : IRNode(t) {
    }
    virtual Expr mutate_expr(IRMutator *v) const = 0;
    Type type;
};

/** We use the "curiously recurring template pattern" to avoid
   duplicated code in the IR Nodes. These classes live between the
   abstract base classes and the actual IR Nodes in the
   inheritance hierarchy. It provides an implementation of the
   accept function necessary for the visitor pattern to work, and
   a concrete instantiation of a unique IRNodeType per class. */
template<typename T>
struct ExprNode : public BaseExprNode {
    void accept(IRVisitor *v) const override;
    Expr mutate_expr(IRMutator *v) const override;
    ExprNode()
        : BaseExprNode(T::_node_type) {
    }
    ~ExprNode() override = default;
};

template<typename T>
struct StmtNode : public BaseStmtNode {
    void accept(IRVisitor *v) const override;
    Stmt mutate_stmt(IRMutator *v) const override;
    StmtNode()
        : BaseStmtNode(T::_node_type) {
    }
    ~StmtNode() override = default;
};

/** IR nodes are passed around opaque handles to them. This is a
   base class for those handles. It manages the reference count,
   and dispatches visitors. */
struct IRHandle : public IntrusivePtr<const IRNode> {
    HALIDE_ALWAYS_INLINE
    IRHandle() = default;

    HALIDE_ALWAYS_INLINE
    IRHandle(const IRNode *p)
        : IntrusivePtr<const IRNode>(p) {
    }

    /** Dispatch to the correct visitor method for this node. E.g. if
     * this node is actually an Add node, then this will call
     * IRVisitor::visit(const Add *) */
    void accept(IRVisitor *v) const {
        ptr->accept(v);
    }

    /** Downcast this ir node to its actual type (e.g. Add, or
     * Select). This returns nullptr if the node is not of the requested
     * type. Example usage:
     *
     * if (const Add *add = node->as<Add>()) {
     *   // This is an add node
     * }
     */
    template<typename T>
    const T *as() const {
        if (ptr && ptr->node_type == T::_node_type) {
            return (const T *)ptr;
        }
        return nullptr;
    }

    IRNodeType node_type() const {
        return ptr->node_type;
    }
};

/** Integer constants */
struct IntImm : public ExprNode<IntImm> {
    int64_t value;

    static const IntImm *make(Type t, int64_t value);

    static const IRNodeType _node_type = IRNodeType::IntImm;
};

/** Unsigned integer constants */
struct UIntImm : public ExprNode<UIntImm> {
    uint64_t value;

    static const UIntImm *make(Type t, uint64_t value);

    static const IRNodeType _node_type = IRNodeType::UIntImm;
};

/** Floating point constants */
struct FloatImm : public ExprNode<FloatImm> {
    double value;

    static const FloatImm *make(Type t, double value);

    static const IRNodeType _node_type = IRNodeType::FloatImm;
};

/** String constants */
struct StringImm : public ExprNode<StringImm> {
    std::string value;

    static const StringImm *make(const std::string &val);

    static const IRNodeType _node_type = IRNodeType::StringImm;
};

}  // namespace Internal

/** A fragment of Halide syntax. It's implemented as reference-counted
 * handle to a concrete expression node, but it's immutable, so you
 * can treat it as a value type. */
struct Expr : public Internal::IRHandle {
    /** Make an undefined expression */
    HALIDE_ALWAYS_INLINE
    Expr() = default;

    /** Make an expression from a concrete expression node pointer (e.g. Add) */
    HALIDE_ALWAYS_INLINE
    Expr(const Internal::BaseExprNode *n)
        : IRHandle(n) {
    }

    /** Make an expression representing numeric constants of various types. */
    // @{
    explicit Expr(int8_t x)
        : IRHandle(Internal::IntImm::make(Int(8), x)) {
    }
    explicit Expr(int16_t x)
        : IRHandle(Internal::IntImm::make(Int(16), x)) {
    }
    Expr(int32_t x)
        : IRHandle(Internal::IntImm::make(Int(32), x)) {
    }
    explicit Expr(int64_t x)
        : IRHandle(Internal::IntImm::make(Int(64), x)) {
    }
    explicit Expr(uint8_t x)
        : IRHandle(Internal::UIntImm::make(UInt(8), x)) {
    }
    explicit Expr(uint16_t x)
        : IRHandle(Internal::UIntImm::make(UInt(16), x)) {
    }
    explicit Expr(uint32_t x)
        : IRHandle(Internal::UIntImm::make(UInt(32), x)) {
    }
    explicit Expr(uint64_t x)
        : IRHandle(Internal::UIntImm::make(UInt(64), x)) {
    }
    Expr(float16_t x)
        : IRHandle(Internal::FloatImm::make(Float(16), (double)x)) {
    }
    Expr(bfloat16_t x)
        : IRHandle(Internal::FloatImm::make(BFloat(16), (double)x)) {
    }
#if HALIDE_CPP_COMPILER_HAS_FLOAT16
    explicit Expr(_Float16 x)
        : IRHandle(Internal::FloatImm::make(Float(16), (double)x)) {
    }
#endif
    Expr(float x)
        : IRHandle(Internal::FloatImm::make(Float(32), x)) {
    }
    explicit Expr(double x)
        : IRHandle(Internal::FloatImm::make(Float(64), x)) {
    }
    // @}

    /** Make an expression representing a const string (i.e. a StringImm) */
    Expr(const std::string &s)
        : IRHandle(Internal::StringImm::make(s)) {
    }

    /** Override get() to return a BaseExprNode * instead of an IRNode * */
    HALIDE_ALWAYS_INLINE
    const Internal::BaseExprNode *get() const {
        return (const Internal::BaseExprNode *)ptr;
    }

    /** Get the type of this expression node */
    HALIDE_ALWAYS_INLINE
    Type type() const {
        return get()->type;
    }
};

/** This lets you use an Expr as a key in a map of the form
 * map<Expr, Foo, ExprCompare> */
struct ExprCompare {
    bool operator()(const Expr &a, const Expr &b) const {
        return a.get() < b.get();
    }
};

/** A single-dimensional span. Includes all numbers between min and
 * (min + extent - 1). */
struct Range {
    Expr min, extent;

    Range() = default;
    Range(const Expr &min_in, const Expr &extent_in);
};

/** A multi-dimensional box. The outer product of the elements */
typedef std::vector<Range> Region;

/** An enum describing different address spaces to be used with Func::store_in. */
enum class MemoryType {
    /** Let Halide select a storage type automatically */
    Auto,

    /** Heap/global memory. Allocated using halide_malloc, or
     * halide_device_malloc */
    Heap,

    /** Stack memory. Allocated using alloca. Requires a constant
     * size. Corresponds to per-thread local memory on the GPU. If all
     * accesses are at constant coordinates, may be promoted into the
     * register file at the discretion of the register allocator. */
    Stack,

    /** Register memory. The allocation should be promoted into the
     * register file. All stores must be at constant coordinates. May
     * be spilled to the stack at the discretion of the register
     * allocator. */
    Register,

    /** Allocation is stored in GPU shared memory. Also known as
     * "local" in OpenCL, and "threadgroup" in metal. Can be shared
     * across GPU threads within the same block. */
    GPUShared,

    /** Allocation is stored in GPU texture memory and accessed through
     * hardware sampler */
    GPUTexture,

    /** Allocate Locked Cache Memory to act as local memory */
    LockedCache,
    /** Vector Tightly Coupled Memory. HVX (Hexagon) local memory available on
     * v65+. This memory has higher performance and lower power. Ideal for
     * intermediate buffers. Necessary for vgather-vscatter instructions
     * on Hexagon */
    VTCM,

    /** AMX Tile register for X86. Any data that would be used in an AMX matrix
     * multiplication must first be loaded into an AMX tile register. */
    AMXTile,
};

/** A bank-conflict-avoidance swizzle composed onto an allocation's affine
 * (stride) layout. Given the logical (affine) element index i, the physical
 * element index is:
 *
 *     phys = i ^ (((i >> shift) & ((1 << bits) - 1)) << base)
 *
 * a self-inverse XOR permutation of aligned blocks (the CUTLASS Swizzle<B,M,S>
 * family). All fields are in element units. `bits == 0` is the identity (no
 * swizzle). The swizzle is applied only at the codegen address seam; the IR
 * index expression stays affine, so it remains a single shared invariant that
 * every accessor of the allocation observes. See \ref Func::swizzle_storage. */
struct SwizzleLayout {
    int bits = 0;   ///< Number of address bits permuted (0 = identity).
    int base = 0;   ///< Low bit where the XOR field is injected; granule = elem << base.
    int shift = 0;  ///< Low bit of the field read to form the XOR.

    bool defined() const {
        return bits > 0;
    }
    bool operator==(const SwizzleLayout &o) const {
        return bits == o.bits && base == o.base && shift == o.shift;
    }
    bool operator!=(const SwizzleLayout &o) const {
        return !(*this == o);
    }
    // Provided so the generic IREquality comparator can order Allocate nodes.
    bool operator<(const SwizzleLayout &o) const {
        if (bits != o.bits) {
            return bits < o.bits;
        }
        if (base != o.base) {
            return base < o.base;
        }
        return shift < o.shift;
    }
};

/** Named shared-memory swizzle modes used with \ref Func::swizzle_storage to
 * avoid GPU shared-memory bank conflicts. Resolved to a concrete \ref
 * SwizzleLayout (in element units) using the Func's element size. The granule
 * is a 16-byte (128-bit) bank line; the suffix is the number of distinct bank
 * lines permuted. Use the raw SwizzleLayout overload for exact (B,M,S) control. */
enum class Swizzle {
    None,
    XOR_32B,
    XOR_64B,
    XOR_128B,
};

namespace Internal {

/** An enum describing a type of loop traversal. Used in schedules,
 * and in the For loop IR node. Serial is a conventional ordered for
 * loop. Iterations occur in increasing order, and each iteration must
 * appear to have finished before the next begins. Parallel, GPUBlock,
 * and GPUThread are parallel and unordered: iterations may occur in
 * any order, and multiple iterations may occur
 * simultaneously. Vectorized and GPULane are parallel and
 * synchronous: they act as if all iterations occur at the same time
 * in lockstep. */
enum class ForType {
    Serial,
    Parallel,
    Vectorized,
    Unrolled,
    Extern,
    GPUBlock,
    GPUThread,
    GPULane,
};

/** Where a vectorized op is physically realized on a GPU. Orthogonal to
 * ForType (which describes loop structure): a loop may be Vectorized and,
 * separately, carry a realization that says whether the vector lives in one
 * thread's registers, across a warp's lanes, or across a warp group's
 * threads. Register is today's behavior and the default. One matcher reads a
 * vectorized op and selects the collective instruction by
 * (op-kind x realization x target); the realization is never an instruction
 * directive. See research/gpu_collective_vectorization.md. */
enum class GPUVectorScope {
    Register,   // in one thread's registers (today: ld/st.v4, dp4a, cp.async, fma)
    Warp,       // across a warp's lanes (shfl, redux, ldmatrix, mma.sync)
    WarpGroup,  // across a warp group's threads (wgmma, tcgen05)
};

/** The set of threads that must rendezvous at a synchronization point. A
 * schedule-derived, target-independent property (the same thread -> warp ->
 * warp-group -> block -> cluster ladder as GPUVectorScope / the gpu_warps and
 * gpu_clusters tiers). A SyncRequirement carries a SyncScope; the mechanism
 * (whole-CTA barrier / named barrier / mbarrier / cluster barrier / ...) is
 * matched from (scope x target x situation), never directed. A coarser scope's
 * mechanism may satisfy a finer requirement when every thread of the coarser
 * scope reaches it. See research/gpu_sync_model.md. */
enum class SyncScope {
    Warp,       // lanes of one warp (often implicit / __syncwarp)
    WarpGroup,  // warps of one gpu_warps group (partial/named barrier)
    Block,      // the whole CTA (today's __syncthreads)
    Cluster,    // CTAs of a cluster, sm_90+ (cluster barrier / DSMEM)
};

/** The kind of asynchronous completion an async_issue/async_wait requirement tracks. Like
 * SyncScope for barriers, this is target-independent; the selector lowers (CompletionKind x
 * SyncScope x target) to a concrete completion mechanism (cp.async commit/wait_group, an mbarrier
 * transaction on sm_90, or wgmma.commit/wait_group). See research/fusegpu_rearch_plan.md. */
enum class CompletionKind {
    CpAsyncGroup,  // cp.async (per-thread) -> commit_group / wait_group
    CpAsyncBulk,   // bulk/tensor copy (sm_90 cp.async.bulk / TMA) -> mbarrier transaction (expect_tx)
    WgmmaGroup,    // wgmma.mma_async -> wgmma.commit_group / wait_group
};

/** Check if for_type executes for loop iterations in parallel and unordered. */
bool is_unordered_parallel(ForType for_type);

/** Returns true if for_type executes for loop iterations in parallel. */
bool is_parallel(ForType for_type);

/** Returns true if for_type is GPUBlock, GPUThread, or GPULane. */
bool is_gpu(ForType for_type);

/** A reference-counted handle to a statement node. */
struct Stmt : public IRHandle {
    Stmt() = default;
    Stmt(const BaseStmtNode *n)
        : IRHandle(n) {
    }

    /** Override get() to return a BaseStmtNode * instead of an IRNode * */
    HALIDE_ALWAYS_INLINE
    const BaseStmtNode *get() const {
        return (const Internal::BaseStmtNode *)ptr;
    }

    /** This lets you use a Stmt as a key in a map of the form
     * map<Stmt, Foo, Stmt::Compare> */
    struct Compare {
        bool operator()(const Stmt &a, const Stmt &b) const {
            return a.ptr < b.ptr;
        }
    };
};

}  // namespace Internal
}  // namespace Halide

#endif
