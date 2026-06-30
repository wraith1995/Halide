#include "CodeGen_PTX_Dev.h"
#include "CSE.h"
#include "CanonicalizeGPUVars.h"
#include "CodeGen_GPU_Dev.h"
#include "CodeGen_Internal.h"
#include "CodeGen_LLVM.h"
#include "ConciseCasts.h"
#include "Debug.h"
#include "ExprUsesVar.h"
#include "IREquality.h"
#include "IRMatch.h"
#include "IRMutator.h"
#include "IROperator.h"
#include "IRPrinter.h"
#include "LLVM_Headers.h"
#include "LLVM_Runtime_Linker.h"
#include "Simplify.h"
#include "Solve.h"
#include "Target.h"
#include "Util.h"

#include <fstream>

namespace Halide {
namespace Internal {

using std::string;
using std::vector;

using namespace Halide::ConciseCasts;

using namespace llvm;

#ifdef WITH_NVPTX

namespace {

/** A code generator that emits GPU code from a given Halide stmt. */
class CodeGen_PTX_Dev : public CodeGen_LLVM, public CodeGen_GPU_Dev {
public:
    /** Create a PTX device code generator. */
    CodeGen_PTX_Dev(const Target &host);
    ~CodeGen_PTX_Dev() override;

    void add_kernel(Stmt stmt,
                    const std::string &name,
                    const std::vector<DeviceArgument> &args) override;

    static void test();

    std::vector<char> compile_to_src() override;
    std::string get_current_kernel_name() override;

    void dump() override;

    std::string print_gpu_name(const std::string &name) override;

    std::string api_unique_name() override {
        return "cuda";
    }

protected:
    using CodeGen_LLVM::visit;

    /** (Re)initialize the PTX module. This is separate from compile, since
     * a PTX device module will often have many kernels compiled into it for
     * a single pipeline. */
    /* override */ void init_module() override;

    /** We hold onto the basic block at the start of the device
     * function in order to inject allocas */
    llvm::BasicBlock *entry_block;

    /** Nodes for which we need to override default behavior for the GPU runtime */
    // @{
    void visit(const Call *) override;
    void visit(const For *) override;
    void visit(const Allocate *) override;
    void visit(const Free *) override;
    void visit(const AssertStmt *) override;
    void visit(const Load *) override;
    void visit(const Store *) override;
    void visit(const Atomic *) override;
    void codegen_vector_reduce(const VectorReduce *op, const Expr &init) override;
    // @}

    /** Emit one Hopper wgmma.mma_async (m64n{N}k16, fp16->f32, N in {16,32,...,256}) for a
     * WarpGroup tile reduce: accumulate the product of the shared tiles described by
     * desc_a/desc_b into the N/2 per-thread f32 accumulator registers (acc, read-modify-write).
     * scale_d selects accumulate (true, k>0) vs overwrite (false, k==0). Emitted as inline PTX
     * asm (no LLVM intrinsic exists); the caller wraps the k-loop in wgmma.fence / commit_group /
     * wait_group. Returns the updated {N/2 x f32} accumulator. N is the wgmma N dimension --
     * large N (n64/n128/n256) is ~95% of peak vs ~38% for n16 (Luo et al. 2402.13499). The
     * fragment lane<->element map (frag_row_m/frag_col_n) extends to N/2 regs by construction.
     * See research/gpu_recognizer_design.md §5a/§5f. */
    llvm::Value *emit_wgmma(int n, llvm::Value *acc, llvm::Value *desc_a,
                            llvm::Value *desc_b, bool scale_d);

    // P1: set when a cp.async copy (vectorized global->shared) was emitted since the last
    // gpu_thread_barrier, so the barrier commits + waits for it (synchronous cp.async).
    bool emitted_cp_async = false;

    // M0 wgmma: the {f32 x 8} accumulator from the one wgmma.mma_async collective
    // emitted per kernel (the recognizer's per-element calls all extract from it).
    // Reset per add_kernel. See research/gpu_recognizer_design.md S5b.
    // Keyed by REGISTER BANK = the D fragment's base register (the accum_reg D_in load index,
    // = m_it*(N/2)). One m64nN wgmma per bank: a BM=128 consumer tile is 2 stacked m64n128 wgmmas
    // (m_it=0 rows 0-63 in bank 0, m_it=1 rows 64-127 in bank N/2) -- Hopper has no m128 wgmma, so
    // the two supertiles are TWO instructions into two register banks, not one m64n256. The frag8/
    // scalar (non-accumulator) path always uses bank 0 (a single m64nN <= m64n256). Reset per kernel.
    std::map<int, llvm::Value *> cached_wgmma_acc;
    // The basic block `cached_wgmma_acc` was emitted into. The cache is only valid within
    // that block: the recognizer always co-locates the wgmma emit and its fragment extracts
    // in one straight-line block, so when codegen has moved to a DIFFERENT block (e.g. a
    // second consumer warp group's partition branch in M3) the cached Values would not
    // dominate the new uses -- invalidate so that scope re-emits its own wgmma. Reset per kernel.
    llvm::BasicBlock *cached_wgmma_block = nullptr;

    /** Build a 64-bit Hopper wgmma shared matrix descriptor for the operand tile
     * at `tile_origin` (element index) within shared allocation `buffer`. The
     * start-address field is the tile's shared-window byte offset (the addrspace(3)
     * pointer is the offset directly; dynamic shared starts at 0). lbo_bytes /
     * sbo_bytes are the leading- / stride-dimension byte offsets of the core-matrix
     * layout (M0 first guesses, pinned against the f64 oracle on H100). swizzle=0
     * (no swizzle) for M0. See research/gpu_recognizer_design.md S5b. */
    llvm::Value *build_wgmma_descriptor(const std::string &buffer, Type elem_type,
                                        const Expr &tile_origin, int lbo_bytes, int sbo_bytes,
                                        int swizzle_bytes = 0);
    // The wgmma descriptor swizzle mode for a shared operand: read from the operand's
    // recorded store_in SwizzleLayout (so it matches the TMA tensor-map swizzle). 0 if none.
    int operand_swizzle_bytes(const std::string &buffer) const;

    /** Apply a shared-memory bank-conflict swizzle (recorded per allocation in
     * visit(Allocate)) to the element index, at the address seam. Identity for
     * non-swizzled buffers. The IR index stays affine; only the emitted address
     * is permuted, so producer and consumer of the same allocation agree. */
    llvm::Value *codegen_swizzled_index(const std::string &buffer, Type type, llvm::Value *index) override;

    /** Swizzle + element-size-in-bytes for each swizzled shared allocation, keyed
     * by name. The byte size lets the hook rescale element-unit swizzle params to
     * the units of a wider access (e.g. the 4-wide u128 store path). */
    std::map<std::string, std::pair<SwizzleLayout, int>> shared_swizzles;

    std::string mcpu_target() const override;
    std::string mcpu_tune() const override;
    std::string mattrs() const override;
    bool use_soft_float_abi() const override;
    int native_vector_bits() const override;
    bool promote_indices() const override {
        return false;
    }

    Type upgrade_type_for_arithmetic(const Type &t) const override {
        return t;
    }
    Type upgrade_type_for_storage(const Type &t) const override;

    /** Map from simt variable names (e.g. foo.block_id_x) to the llvm ptx
     * intrinsic functions to call to get them. */
    std::string simt_intrinsic(const std::string &name);

    bool supports_atomic_add(const Type &t) const override;
};

CodeGen_PTX_Dev::CodeGen_PTX_Dev(const Target &host)
    : CodeGen_LLVM(host) {
    context = new llvm::LLVMContext();
}

CodeGen_PTX_Dev::~CodeGen_PTX_Dev() {
    // This is required as destroying the context before the module
    // results in a crash. Really, responsibility for destruction
    // should be entirely in the parent class.
    // TODO: Figure out how to better manage the context -- e.g. allow using
    // same one as the host.
    module.reset();
    delete context;
}

Type CodeGen_PTX_Dev::upgrade_type_for_storage(const Type &t) const {
    if (t.element_of() == Float(16)) {
        return t;
    }
    return CodeGen_LLVM::upgrade_type_for_storage(t);
}

void CodeGen_PTX_Dev::add_kernel(Stmt stmt,
                                 const std::string &name,
                                 const std::vector<DeviceArgument> &args) {
    internal_assert(module != nullptr);

    debug(2) << "In CodeGen_PTX_Dev::add_kernel\n";

    cached_wgmma_acc.clear();
    cached_wgmma_block = nullptr;

    // Now deduce the types of the arguments to our function
    vector<llvm::Type *> arg_types(args.size());
    for (size_t i = 0; i < args.size(); i++) {
        if (args[i].is_buffer) {
            arg_types[i] = ptr_t;
        } else {
            arg_types[i] = llvm_type_of(args[i].type);
        }
    }

    // Make our function
    FunctionType *func_t = FunctionType::get(void_t, arg_types, false);
    function = llvm::Function::Create(func_t, llvm::Function::ExternalLinkage, name, module.get());
    set_function_attributes_from_halide_target_options(*function);

    // Mark the buffer args as no alias
    for (size_t i = 0; i < args.size(); i++) {
        if (args[i].is_buffer) {
            function->addParamAttr(i, Attribute::NoAlias);
        }
    }

    function->setCallingConv(llvm::CallingConv::PTX_Kernel);

    // Make the initial basic block
    entry_block = BasicBlock::Create(*context, "entry", function);
    builder->SetInsertPoint(entry_block);

    // Put the arguments in the symbol table
    vector<string> arg_sym_names;
    {
        size_t i = 0;
        for (auto &fn_arg : function->args()) {

            string arg_sym_name = args[i].name;
            sym_push(arg_sym_name, &fn_arg);
            fn_arg.setName(arg_sym_name);
            arg_sym_names.push_back(arg_sym_name);

            i++;
        }
    }

    // We won't end the entry block yet, because we'll want to add
    // some allocas to it later if there are local allocations. Start
    // a new block to put all the code.
    BasicBlock *body_block = BasicBlock::Create(*context, "body", function);
    builder->SetInsertPoint(body_block);

    debug(1) << "Generating llvm bitcode for kernel...\n";
    // Ok, we have a module, function, context, and a builder
    // pointing at a brand new basic block. We're good to go.
    stmt.accept(this);

    // Now we need to end the function
    builder->CreateRetVoid();

    // Make the entry block point to the body block
    builder->SetInsertPoint(entry_block);
    builder->CreateBr(body_block);

    // Add the nvvm annotation that it is a kernel function.
    llvm::Metadata *md_args[] = {
        llvm::ValueAsMetadata::get(function),
        MDString::get(*context, "kernel"),
        llvm::ValueAsMetadata::get(ConstantInt::get(i32_t, 1))};

    MDNode *md_node = MDNode::get(*context, md_args);

    module->getOrInsertNamedMetadata("nvvm.annotations")->addOperand(md_node);

    // Emit `.maxntid` (launch bounds) from the kernel's GPU-thread loop extents. Without it ptxas
    // assumes the kernel might launch with up to 1024 threads/block and budgets only ~64 registers/
    // thread, spilling the wgmma D accumulator to local memory (the matmul_3 PTX diff: 32KB local
    // depot + st.local in the steady loop, where the reference's __launch_bounds__ keeps `d[]` in
    // registers). Telling ptxas the real (smaller) block size lets it spend up to 65536/N registers/
    // thread instead. Only emitted when all thread extents are compile-time constants.
    {
        class ThreadExtents : public IRVisitor {
            using IRVisitor::visit;
            void visit(const For *op) override {
                for (int i = 0; i < 3; i++) {
                    if (ends_with(op->name, gpu_thread_name(i))) {
                        if (auto e = as_const_int(simplify(op->extent()))) {
                            // A dimension can appear on several fused thread loops; the block extent
                            // is the max over them (they share the same %tid.i).
                            extent[i] = std::max(extent[i], (int64_t)*e);
                        } else {
                            known[i] = false;
                        }
                    }
                }
                IRVisitor::visit(op);
            }

        public:
            int64_t extent[3] = {1, 1, 1};
            bool known[3] = {true, true, true};
        } te;
        stmt.accept(&te);
        if (te.known[0] && te.known[1] && te.known[2]) {
            // LLVM 21's NVPTX reads launch bounds from the `nvvm.maxntid` function attribute (the
            // old nvvm.annotations "maxntidx" path is no longer honored). Value is the comma-joined
            // x,y,z block dims; ptxas then budgets up to 65536/(x*y*z) registers/thread.
            std::string v = std::to_string(te.extent[0]) + "," + std::to_string(te.extent[1]) +
                            "," + std::to_string(te.extent[2]);
            function->addFnAttr("nvvm.maxntid", v);
            debug(2) << "PTX kernel " << name << " nvvm.maxntid = " << v << "\n";
        }
    }

    // Now verify the function is ok
    verifyFunction(*function);

    // Finally, verify the module is ok
    verifyModule(*module);

    debug(2) << "Done generating llvm bitcode for PTX\n";

    // Clear the symbol table
    for (const auto &arg_sym_name : arg_sym_names) {
        sym_pop(arg_sym_name);
    }
}

void CodeGen_PTX_Dev::init_module() {
    shared_swizzles.clear();
    // This class uses multiple inheritance. It's a GPU device code generator,
    // and also an llvm-based one. Both of these track strict_float presence,
    // but OffloadGPULoops only sets the GPU device code generator flag, so here
    // we set the CodeGen_LLVM flag to match.
    CodeGen_LLVM::any_strict_float = CodeGen_GPU_Dev::any_strict_float;

    init_context();

    module = get_initial_module_for_ptx_device(target, context);

    struct Intrinsic {
        const char *name;
        Type ret_type;
        const char *intrin_name;
        vector<Type> arg_types;
    };

    Intrinsic ptx_intrins[] = {
        {"dp4a", Int(32), "dp4a_s32_s32", {Int(8, 4), Int(8, 4), Int(32)}},
        {"dp4a", Int(32), "dp4a_s32_u32", {Int(8, 4), UInt(8, 4), Int(32)}},
        {"dp4a", Int(32), "dp4a_u32_s32", {UInt(8, 4), Int(8, 4), Int(32)}},
        {"dp4a", UInt(32), "dp4a_u32_u32", {UInt(8, 4), UInt(8, 4), UInt(32)}},
        {"dp2a", Int(32), "dp2a_s32_s32", {Int(16, 4), Int(8, 4), Int(32)}},
        {"dp2a", Int(32), "dp2a_s32_u32", {Int(16, 4), UInt(8, 4), Int(32)}},
        {"dp2a", Int(32), "dp2a_u32_s32", {UInt(16, 4), Int(8, 4), Int(32)}},
        {"dp2a", UInt(32), "dp2a_u32_u32", {UInt(16, 4), UInt(8, 4), UInt(32)}},
        {"round", Float(32), "llvm.rint.f32", {Float(32)}},
        {"round", Float(64), "llvm.rint.f64", {Float(64)}},
    };

    for (auto &&i : ptx_intrins) {
        auto *fn = declare_intrin_overload(i.name, i.ret_type, i.intrin_name, std::move(i.arg_types));
        function_does_not_access_memory(fn);
        fn->addFnAttr(llvm::Attribute::NoUnwind);
    }

    if (CodeGen_GPU_Dev::any_strict_float) {
        set_strict_fp_math();
        in_strict_float = target.has_feature(Target::StrictFloat);
    } else {
        set_fast_fp_math();
    }
}

void CodeGen_PTX_Dev::visit(const Call *op) {
    if (op->is_intrinsic() && op->name == "wgmma_m64n16k16_f32_accum_reg") {
        // Mainloop 1b (Option A, register accumulator). Accumulate ONE wgmma tile into a
        // LOOP-CARRIED D fragment held in `prod`'s per-thread registers (the recognizer emits
        // 8 SCALAR stores prod[base+i] = accum_reg(i, ...), mirroring the M1 scalar+cache path
        // so there is no vector store to register memory and no per-lane scalarization of the
        // collective). Args: (reg, D_in_load, n_chunks, LoadA, strideA, LoadB, strideB) -> f32.
        //   - D_in_load is a SCALAR Load marking prod + its per-thread base index; codegen seeds
        //     the accumulator from prod[base+0..7] (the running sum carried across ko by
        //     compute_at), so every chunk uses scaleD=1 (accumulate).
        //   - The wgmma is emitted ONCE per ko body (cached); the R=N/2 scalar stores extract reg i.
        // Args: (reg, N, D_in_load, n_chunks, LoadA, strideA, LoadB, strideB) -> f32.
        internal_assert(op->args.size() == 8u) << "wgmma accum_reg arg count mismatch\n";
        auto reg = as_const_int(op->args[0]);
        auto n_dim = as_const_int(op->args[1]);
        const Load *dl = op->args[2].as<Load>();
        auto n_chunks = as_const_int(op->args[3]);
        const Load *la = op->args[4].as<Load>();
        auto stride_a = as_const_int(op->args[5]);
        const Load *lb = op->args[6].as<Load>();
        auto stride_b = as_const_int(op->args[7]);
        internal_assert(reg && n_dim && dl && n_chunks && la && stride_a && lb && stride_b)
            << "wgmma accum_reg args malformed\n";
        const int N = (int)*n_dim;
        const int R = N / 2;
        // The register BANK = the D fragment base (D_in load index, = m_it*(N/2)). Each bank is its
        // own m64nN wgmma: a BM=128 tile is 2 stacked m64n128 wgmmas (bank 0 rows 0-63, bank R rows
        // 64-127). reg is the fragment register WITHIN the bank (0..R-1). For one m_it (BM<=64) bank=0
        // and this is exactly the old single-wgmma behavior.
        auto bank_opt = as_const_int(simplify(dl->index));
        internal_assert(bank_opt) << "wgmma accum_reg D_in index must be a constant register bank\n";
        const int bank = (int)*bank_opt;

        // Drop caches built in a different basic block (they would not dominate here).
        if (!cached_wgmma_acc.empty() && cached_wgmma_block != builder->GetInsertBlock()) {
            cached_wgmma_acc.clear();
        }
        if (cached_wgmma_acc.find(bank) == cached_wgmma_acc.end()) {
            // Seed the {f32 x R} accumulator from prod[bank+0..bank+R-1] (R scalar loads).
            llvm::Type *f32 = llvm::Type::getFloatTy(*context);
            llvm::StructType *acc_ty = llvm::StructType::get(*context, std::vector<llvm::Type *>(R, f32));
            llvm::Value *acc = llvm::UndefValue::get(acc_ty);
            for (int j = 0; j < R; j++) {
                Expr slot = simplify(dl->index + j);
                Expr load_j = Load::make(dl->type, dl->name, slot, Buffer<>(), dl->param,
                                         const_true(), ModulusRemainder());
                acc = builder->CreateInsertValue(acc, codegen(load_j), j);
            }
            auto emit_wgmma_asm = [&](const char *s) {
                llvm::FunctionType *ft = llvm::FunctionType::get(void_t, false);
                llvm::InlineAsm *ia = llvm::InlineAsm::get(ft, s, "", /*hasSideEffects*/ true);
                builder->CreateCall(ia);
            };
            int sbo = 256 * (int)*n_chunks;
            emit_wgmma_asm("wgmma.fence.sync.aligned;");
            for (int c = 0; c < (int)*n_chunks; c++) {
                Expr off_a = simplify(la->index + Expr((int)(*stride_a) * c));
                Expr off_b = simplify(lb->index + Expr((int)(*stride_b) * c));
                llvm::Value *desc_a = build_wgmma_descriptor(la->name, la->type.element_of(),
                                                             off_a, /*lbo*/ 128, sbo,
                                                             operand_swizzle_bytes(la->name));
                llvm::Value *desc_b = build_wgmma_descriptor(lb->name, lb->type.element_of(),
                                                             off_b, /*lbo*/ 128, sbo,
                                                             operand_swizzle_bytes(lb->name));
                // scaleD=1 ALWAYS: prod already holds the running sum across prior ko iterations.
                acc = emit_wgmma(N, acc, desc_a, desc_b, /*scale_d*/ true);
            }
            emit_wgmma_asm("wgmma.commit_group.sync.aligned;");
            emit_wgmma_asm("wgmma.wait_group.sync.aligned 0;");
            cached_wgmma_acc[bank] = acc;
            cached_wgmma_block = builder->GetInsertBlock();
        }
        value = builder->CreateExtractValue(cached_wgmma_acc[bank], (unsigned)*reg);
        return;
    }

    if (op->is_intrinsic() && (op->name == "wgmma_m64n16k16_f32" ||
                               op->name == "wgmma_m64n16k16_f32_frag8")) {
        // Hopper tile reduce, emitted by lower_warp_group_tiles. Two forms share the same
        // emit; they differ only in how the cached {f32 x 8} D fragment is returned:
        //   scalar "wgmma_m64n16k16_f32"      (reg, n_chunks, LoadA, strideA, LoadB, strideB)
        //                                      -> extract one fragment register (unroll path).
        //   vector "wgmma_m64n16k16_f32_frag8"(n_chunks, LoadA, strideA, LoadB, strideB)
        //                                      -> the whole fragment as a <8 x f32> vector
        //                                         (vector-native path; PROTOTYPE for 1.x).
        // K = n_chunks*16; each chunk is one wgmma.mma_async accumulating into the same D
        // (scaleD carry). The collective is emitted ONCE per kernel. See §5b/§5c.
        // Args now carry the wgmma N dimension: scalar (reg, N, n_chunks, LoadA,sA, LoadB,sB),
        // vector frag8 (N, n_chunks, LoadA,sA, LoadB,sB). R = N/2 per-thread f32 registers.
        bool vec = (op->name == "wgmma_m64n16k16_f32_frag8");
        int b = vec ? 0 : 1;  // arg base: scalar form has reg at [0]
        internal_assert(op->args.size() == (vec ? 6u : 7u))
            << "wgmma_m64n16k16_f32 arg count mismatch\n";
        auto reg = vec ? std::optional<int64_t>(0) : as_const_int(op->args[0]);
        auto n_dim = as_const_int(op->args[b + 0]);
        auto n_chunks = as_const_int(op->args[b + 1]);
        const Load *la = op->args[b + 2].as<Load>();
        auto stride_a = as_const_int(op->args[b + 3]);
        const Load *lb = op->args[b + 4].as<Load>();
        auto stride_b = as_const_int(op->args[b + 5]);
        internal_assert(reg && n_dim && n_chunks && la && stride_a && lb && stride_b)
            << "wgmma_m64n16k16_f32 args malformed\n";
        const int N = (int)*n_dim;
        const int R = N / 2;  // per-thread accumulator registers
        if (getenv("HL_DEBUG_WGMMA")) {
            debug(0) << "[wgtile] codegen wgmma " << (vec ? "frag8(vec)" : "reg")
                     << " N=" << N << " n_chunks=" << *n_chunks << " strideA=" << *stride_a
                     << " strideB=" << *stride_b << "\n";
        }

        // Non-accumulator path: a single m64nN wgmma (N <= 256 => one register bank). Bank 0.
        const int bank = 0;
        // Drop a cache that was built in a different basic block (it would not dominate here).
        if (!cached_wgmma_acc.empty() && cached_wgmma_block != builder->GetInsertBlock()) {
            cached_wgmma_acc.clear();
        }
        if (cached_wgmma_acc.find(bank) == cached_wgmma_acc.end()) {
            // Zeroed {f32 x R} D fragment. Chunk 0 overwrites (scaleD=0); the zero init
            // is belt-and-suspenders. First-guess core-matrix offsets for a 64x16 (A) /
            // 16xN (B) f16 K-major tile (Colfax/CUTLASS no-swizzle): LBO 128 B, SBO 256 B.
            llvm::Type *f32 = llvm::Type::getFloatTy(*context);
            llvm::StructType *acc_ty = llvm::StructType::get(*context, std::vector<llvm::Type *>(R, f32));
            llvm::Value *acc = llvm::UndefValue::get(acc_ty);
            for (int j = 0; j < R; j++) {
                acc = builder->CreateInsertValue(acc, llvm::ConstantFP::get(f32, 0.0), j);
            }

            auto emit_wgmma_asm = [&](const char *s) {
                llvm::FunctionType *ft = llvm::FunctionType::get(void_t, false);
                llvm::InlineAsm *ia = llvm::InlineAsm::get(ft, s, "", /*hasSideEffects*/ true);
                builder->CreateCall(ia);
            };
            // fence (make prior register/shared writes visible to the async MMA), then the
            // K/16 chunks issued back-to-back into one commit group, accumulating into D
            // (chunk 0 scaleD=0 overwrite, rest scaleD=1 carry), then commit + wait once.
            // NOTE: this wait_group 0 is the *unpipelined* placement (num_stages=1 in Triton
            // terms). The waits are emitted separably so a future loop-pipelining pass can
            // hoist them across a staging loop without touching this mma path.
            // LBO (K-core-matrix stride = ki*mi = 64 elems = 128 B) is constant. SBO
            // (the M-/N-core-matrix stride) is the operand's mo/no storage stride =
            // ki*mi*ko_total = 8*K elems = 16*K bytes, because the [mo][ko][mi][ki]
            // layout interleaves ALL of K between consecutive M-core-matrices. With
            // K = 16*n_chunks that is 256*n_chunks bytes (256 for M0's K=16). TODO:
            // read LBO/SBO off the producer's shared storage strides (auto-layout).
            int sbo = 256 * (int)*n_chunks;
            emit_wgmma_asm("wgmma.fence.sync.aligned;");
            for (int c = 0; c < (int)*n_chunks; c++) {
                Expr off_a = simplify(la->index + Expr((int)(*stride_a) * c));
                Expr off_b = simplify(lb->index + Expr((int)(*stride_b) * c));
                llvm::Value *desc_a = build_wgmma_descriptor(la->name, la->type.element_of(),
                                                             off_a, /*lbo*/ 128, sbo,
                                                             operand_swizzle_bytes(la->name));
                llvm::Value *desc_b = build_wgmma_descriptor(lb->name, lb->type.element_of(),
                                                             off_b, /*lbo*/ 128, sbo,
                                                             operand_swizzle_bytes(lb->name));
                acc = emit_wgmma(N, acc, desc_a, desc_b, /*scale_d*/ c > 0);
            }
            emit_wgmma_asm("wgmma.commit_group.sync.aligned;");
            emit_wgmma_asm("wgmma.wait_group.sync.aligned 0;");
            cached_wgmma_acc[bank] = acc;
            cached_wgmma_block = builder->GetInsertBlock();
        }
        if (vec) {
            // Vector-native: return the whole fragment as a <R x f32> (R=N/2). The recognizer
            // stores it with an R-lane non-affine scatter index (one vector op, no unroll).
            llvm::Type *f32 = llvm::Type::getFloatTy(*context);
            llvm::Value *v = llvm::UndefValue::get(llvm::FixedVectorType::get(f32, R));
            for (int j = 0; j < R; j++) {
                v = builder->CreateInsertElement(v, builder->CreateExtractValue(cached_wgmma_acc[bank], j),
                                                 (uint64_t)j);
            }
            value = v;
        } else {
            value = builder->CreateExtractValue(cached_wgmma_acc[bank], (unsigned)*reg);
        }
        return;
    }

    // F3 (mbarrier completion): the deep cp.async ring pipeline's full edge. The mbarrier
    // address is carried as a scalar Load(UInt(64), <edge>.full_mbar, slot) so that
    // ExtractSharedAndHeapAllocations folds the allocation's shared-window offset into the
    // slot index for free (same path as a normal shared Load); codegen never loads the value,
    // it derives the addrspace(3) pointer and uses its integer value as the .shared byte offset
    // (dynamic shared starts at 0 -- identical to the wgmma descriptor's tile origin). Emitted
    // as inline PTX asm because LLVM 21 NVPTX lacks the parity try_wait intrinsic. See §5e.
    auto mbar_shared_addr = [&](const Expr &ref) -> llvm::Value * {
        const Load *m = ref.as<Load>();
        internal_assert(m) << "mbarrier intrinsic address must be a Load carrier.\n";
        llvm::Value *ptr = codegen_buffer_pointer(m->name, m->type.element_of(), m->index);
        return builder->CreatePtrToInt(ptr, i32_t);  // base 0 => int value IS the shared offset
    };
    if (op->is_intrinsic() && op->name == "mbarrier_init") {
        // Arm all N ring slots of a full mbarrier with `count` expected producer arrivals
        // (= producer warp-group thread count). Done ONCE per block by thread 0 (init is a
        // single-thread op; multiple inits of one mbarrier race), then a CTA barrier so the
        // armed barriers are visible before any producer arrive / consumer wait. This marker
        // is placed at block level (outside the thread loops) so it runs on all threads; the
        // tid==0 guard + barrier here make it once-per-block. Args: (mbar_base_ref, N, count).
        // Arm EVERY slot of EVERY producer AND barrier in ONE self-contained opaque inline asm:
        // it reads %tid itself, predicates each `@p mbarrier.init`, then an UNCONDITIONAL
        // `bar.sync 0`. Init+barrier as a single side-effecting asm is the only robust form -- the
        // compiler can neither split it nor conditionalize the barrier. Every prior form (CondBr
        // diamond; predicated init + separate barrier; even per-producer opaque asms) let the
        // compiler/scheduler place a barrier into an `if(tid==0)` region so other lanes skipped it
        // -> deadlock. A SINGLE combined init lands in the uniform entry region. Args: flattened
        // triples (base_ref Load, ring_n, count) per producer.
        internal_assert(op->args.size() % 3u == 0u && !op->args.empty())
            << "mbarrier_init expects flattened (base_ref, N, count) triples.\n";
        std::vector<llvm::Value *> addrs;  // per slot
        std::vector<llvm::Value *> counts;
        for (size_t t = 0; t < op->args.size(); t += 3) {
            const Load *base = op->args[t].as<Load>();
            auto n = as_const_int(op->args[t + 1]);
            internal_assert(base && n) << "mbarrier_init triple malformed.\n";
            llvm::Value *cnt = codegen(op->args[t + 2]);
            for (int i = 0; i < (int)*n; i++) {
                Expr slot = simplify(base->index + i);  // u64 elements; each mbarrier is 8 B
                llvm::Value *ptr = codegen_buffer_pointer(base->name, base->type.element_of(), slot);
                addrs.push_back(builder->CreatePtrToInt(ptr, i32_t));
                counts.push_back(cnt);
            }
        }
        const int nslots = (int)addrs.size();
        std::string asm_str =
            "{ .reg .pred mbar_e; .reg .u32 mbar_t0, mbar_t1;\n"
            "  mov.u32 mbar_t0, %tid.x;\n"
            "  mov.u32 mbar_t1, %tid.y;\n"
            "  or.b32 mbar_t0, mbar_t0, mbar_t1;\n"
            "  mov.u32 mbar_t1, %tid.z;\n"
            "  or.b32 mbar_t0, mbar_t0, mbar_t1;\n"
            "  setp.eq.u32 mbar_e, mbar_t0, 0;\n";
        std::vector<llvm::Value *> args;
        std::string constraints;
        for (int s = 0; s < nslots; s++) {
            asm_str += "  @mbar_e mbarrier.init.shared.b64 [$" + std::to_string(2 * s) + "], $" +
                       std::to_string(2 * s + 1) + ";\n";
            args.push_back(addrs[s]);
            args.push_back(counts[s]);
            constraints += (s ? ",r,r" : "r,r");
        }
        // NOTE: NO bar.sync inside this asm. Halide guards block-level code (this init) to thread 0,
        // so a barrier here would be thread-0-only -> deadlock. Visibility relies on the kernel's
        // existing all-threads CTA barrier between this block-level init and the first mbarrier use.
        asm_str += " }";
        std::vector<llvm::Type *> argtys(2 * nslots, i32_t);
        llvm::FunctionType *ft = llvm::FunctionType::get(void_t, argtys, false);
        llvm::InlineAsm *ia = llvm::InlineAsm::get(ft, asm_str, constraints, /*hasSideEffects*/ true);
        builder->CreateCall(ia, args);
        // NOTE: the CTA barrier that makes the armed mbarriers visible to all threads is emitted
        // SEPARATELY by the lowering (a Block sync_requirement after this init), NOT here. Emitting
        // it inside this handler let the compiler hoist an `if(tid==0)` around the predicated inits
        // and pull the barrier into it, so the other lanes skipped it -> deadlock. A standalone
        // convergent gpu_thread_barrier statement stays uniform.
        value = ConstantInt::get(i32_t, 0);
        return;
    }
    if (op->is_intrinsic() && op->name == "cp_async_mbarrier_arrive") {
        // cp.async.mbarrier.arrive.shared.b64 [addr] -- a DEFERRED arrival: when this thread's
        // prior cp.async copies complete, the mbarrier's pending count is decremented. The
        // producer never waits; the consumer's try_wait observes completion. Args: (mbar_ref).
        internal_assert(op->args.size() == 1u) << "cp_async_mbarrier_arrive expects (mbar_ref).\n";
        llvm::Value *addr = mbar_shared_addr(op->args[0]);
        llvm::FunctionType *ft = llvm::FunctionType::get(void_t, {i32_t}, false);
        // Diagnostic: HL_WG_MBAR_PLAIN swaps the deferred cp.async-completion arrive for a PLAIN
        // mbarrier.arrive (fires immediately, ignores cp.async). Isolates "does the count/parity
        // handshake complete the phase?" (plain works) from "does cp.async.mbarrier.arrive's
        // deferred completion fire?" (only cp.async fails). Data is wrong with PLAIN (no copy
        // wait) -- it only answers hang-vs-not.
        // .noinc is REQUIRED: the default cp.async.mbarrier.arrive INCREMENTS the pending count by 1
        // immediately then decrements on cp.async completion (net zero) -> EXPECTED is never reached
        // -> deadlock. .noinc skips the increment, so each thread's arrive nets -1 and the count
        // (init = producer thread count) reaches 0 when all copies complete. (PLAIN mbarrier.arrive
        // decrements directly -- that's why it worked; this is the deep-pipeline equivalent.)
        const char *asm_str = get_env_variable("HL_WG_MBAR_PLAIN") == "1"
                                  ? "{ .reg .b64 mbar_s; mbarrier.arrive.shared.b64 mbar_s, [$0]; }"
                                  : "cp.async.mbarrier.arrive.noinc.shared.b64 [$0];";
        llvm::InlineAsm *ia = llvm::InlineAsm::get(ft, asm_str, "r", /*hasSideEffects*/ true);
        builder->CreateCall(ia, {addr});
        // This consumes the in-flight cp.async (their completion now arrives on the mbarrier),
        // so the downstream named/CTA barrier must NOT also commit+wait them (would be redundant
        // and would re-serialize the producer). Clear the pending-copy flag.
        emitted_cp_async = false;
        value = ConstantInt::get(i32_t, 0);
        return;
    }
    if (op->is_intrinsic() && op->name == "mbarrier_arrive") {
        // Plain count arrive on the empty (WAR) ring edge: this consumer thread, having drained its
        // wgmma read of the slot (the wgmma.wait_group precedes this), signals the slot free. Every
        // block thread executes it and increments the mbarrier's pending count by 1; when the count
        // (= block thread total, set by mbarrier_init) is reached the phase flips and the producer
        // Q iters ahead unblocks. No election guard (all consumers arrive), no expect_tx/cp.async
        // coupling -- a pure count handshake. The .b64 state result is discarded via a scratch reg.
        internal_assert(op->args.size() == 1u) << "mbarrier_arrive expects (mbar_ref).\n";
        llvm::Value *addr = mbar_shared_addr(op->args[0]);
        llvm::FunctionType *ft = llvm::FunctionType::get(void_t, {i32_t}, false);
        llvm::InlineAsm *ia = llvm::InlineAsm::get(
            ft, "{ .reg .b64 mbar_s; mbarrier.arrive.shared.b64 mbar_s, [$0]; }", "r",
            /*hasSideEffects*/ true);
        builder->CreateCall(ia, {addr});
        value = ConstantInt::get(i32_t, 0);
        return;
    }
    if (op->is_intrinsic() && op->name == "mbarrier_arrive_expect_tx") {
        // TMA (F4): the issuing thread arms the mbarrier with the EXPECTED transaction byte count of
        // an in-flight bulk-tensor copy. cp.async.bulk.tensor decrements this tx count as the bytes
        // land; the consumer's try_wait.parity then observes completion (reusing the F3 mbarrier
        // wait). Issued ONCE by the elected thread (the recognizer guards it). Args: (mbar_ref, bytes).
        internal_assert(op->args.size() == 3u)
            << "mbarrier_arrive_expect_tx expects (mbar_ref, bytes, elected_lane).\n";
        llvm::Value *addr = mbar_shared_addr(op->args[0]);
        llvm::Value *bytes = codegen(op->args[1]);
        llvm::Value *elected = codegen(op->args[2]);
        llvm::FunctionType *ft = llvm::FunctionType::get(void_t, {i32_t, i32_t, i32_t}, false);
        // Single-thread arm on the MODEL's elected lane ($2 = ExecMap::elected_lane): the expect_tx
        // fires ONCE, and a warp-spec sub-region producer elects its OWN first lane (not global tid
        // 0). Guard = (tid.x == elected) && tid.y == 0 && tid.z == 0.
        const char *asm_str =
            "{ .reg .pred tma_e, tma_p; .reg .u32 tma_t0, tma_t1; .reg .b64 tma_st;\n"
            "  mov.u32 tma_t0, %tid.y; mov.u32 tma_t1, %tid.z; or.b32 tma_t0, tma_t0, tma_t1;\n"
            "  setp.eq.u32 tma_p, tma_t0, 0;\n"
            "  mov.u32 tma_t0, %tid.x;\n"
            "  setp.eq.and.u32 tma_e, tma_t0, $2, tma_p;\n"
            "  @tma_e mbarrier.arrive.expect_tx.shared::cta.b64 tma_st, [$0], $1; }";
        llvm::InlineAsm *ia = llvm::InlineAsm::get(ft, asm_str, "r,r,r", /*hasSideEffects*/ true);
        builder->CreateCall(ia, {addr, bytes, elected});
        value = ConstantInt::get(i32_t, 0);
        return;
    }
    if (op->is_intrinsic() && op->name == "tma_load_2d") {
        // TMA (F4): bulk tile load global->shared through a CUtensorMap descriptor, with mbarrier
        // transaction completion. ONE inline asm replaces the whole cooperative per-thread cp.async
        // fill -- the hardware generates the addresses + applies the tensor-map swizzle. Issued by
        // one elected thread (the recognizer guards it). On sm_90 each CTA is a size-1 cluster, so
        // the .shared::cluster destination addressing works with no explicit cluster launch.
        // Args: (dst_smem_ref, tensor_map_ptr_u64, coord_x, coord_y, mbar_ref). dst/mbar are Load
        // carriers (shared byte offset = the addrspace(3) ptr's int value, dynamic shared base 0).
        internal_assert(op->args.size() == 6u)
            << "tma_load_2d expects (dst, map, x, y, mbar, elected_lane).\n";
        llvm::Value *dst = mbar_shared_addr(op->args[0]);
        llvm::Value *map = codegen(op->args[1]);
        llvm::Value *x = codegen(op->args[2]);
        llvm::Value *y = codegen(op->args[3]);
        llvm::Value *mbar = mbar_shared_addr(op->args[4]);
        llvm::Value *elected = codegen(op->args[5]);
        llvm::FunctionType *ft =
            llvm::FunctionType::get(void_t, {i32_t, i64_t, i32_t, i32_t, i32_t, i32_t}, false);
        // Single-thread issue on the MODEL's elected lane ($5 = ExecMap::elected_lane): the bulk copy
        // must issue ONCE, and a warp-spec sub-region producer (e.g. Bs on tid in [32,64)) elects its
        // OWN first lane, not global tid 0. Guard = (tid.x == elected) && tid.y == 0 && tid.z == 0.
        const char *asm_str =
            "{ .reg .pred tma_e, tma_p; .reg .u32 tma_t0, tma_t1;\n"
            "  mov.u32 tma_t0, %tid.y; mov.u32 tma_t1, %tid.z; or.b32 tma_t0, tma_t0, tma_t1;\n"
            "  setp.eq.u32 tma_p, tma_t0, 0;\n"
            "  mov.u32 tma_t0, %tid.x;\n"
            "  setp.eq.and.u32 tma_e, tma_t0, $5, tma_p;\n"
            "  @tma_e cp.async.bulk.tensor.2d.shared::cluster.global.tile"
            ".mbarrier::complete_tx::bytes [$0], [$1, {$2, $3}], [$4]; }";
        llvm::InlineAsm *ia =
            llvm::InlineAsm::get(ft, asm_str, "r,l,r,r,r,r", /*hasSideEffects*/ true);
        builder->CreateCall(ia, {dst, map, x, y, mbar, elected});
        value = ConstantInt::get(i32_t, 0);
        return;
    }
    if (op->is_intrinsic() && op->name == "mbarrier_try_wait") {
        // Spin on mbarrier.try_wait.parity until the awaited phase (parity = (ko/N)&1) completes,
        // i.e. all producer cp.async into this slot are visible. Args: (mbar_ref, parity) for the
        // all-threads full-edge wait, or (mbar_ref, parity, elected_lane) for the empty-edge
        // producer_acquire -- there ONLY the elected (TMA-issuing) lane needs to wait before it
        // overwrites the slot, so we gate the spin to that one lane and let every other thread fall
        // straight through (avoids a 256-thread mbarrier poll storm; the others don't touch the slot
        // and their own consume is gated by the full edge). The try_wait is emitted as asm returning
        // 0/1; the spin loop is built in LLVM IR.
        internal_assert(op->args.size() == 2u || op->args.size() == 3u)
            << "mbarrier_try_wait expects (mbar_ref, parity[, elected_lane]).\n";
        llvm::Value *addr = mbar_shared_addr(op->args[0]);
        llvm::Value *parity = codegen(op->args[1]);
        llvm::Function *fn = builder->GetInsertBlock()->getParent();
        llvm::BasicBlock *loop_bb = llvm::BasicBlock::Create(*context, "mbar_wait", fn);
        llvm::BasicBlock *after_bb = llvm::BasicBlock::Create(*context, "mbar_ready", fn);
        if (op->args.size() == 3u) {
            // Gate: only (tid.x == elected && tid.y == 0 && tid.z == 0) enters the spin; mirrors the
            // expect_tx election so the acquire-waiter is exactly the TMA issuer. Others skip to after.
            llvm::Value *elected = codegen(op->args[2]);
            llvm::FunctionType *eft = llvm::FunctionType::get(i32_t, {i32_t}, false);
            llvm::InlineAsm *eia = llvm::InlineAsm::get(
                eft,
                "{ .reg .pred mbe, mbp; .reg .u32 mbt0, mbt1;\n"
                "  mov.u32 mbt0, %tid.y; mov.u32 mbt1, %tid.z; or.b32 mbt0, mbt0, mbt1;\n"
                "  setp.eq.u32 mbp, mbt0, 0;\n"
                "  mov.u32 mbt0, %tid.x;\n"
                "  setp.eq.and.u32 mbe, mbt0, $1, mbp;\n"
                "  selp.u32 $0, 1, 0, mbe; }",
                "=r,r", /*hasSideEffects*/ true);
            llvm::Value *am_elected = builder->CreateCall(eia, {elected});
            llvm::Value *is_elected = builder->CreateICmpNE(am_elected, ConstantInt::get(i32_t, 0));
            builder->CreateCondBr(is_elected, loop_bb, after_bb);
        } else {
            builder->CreateBr(loop_bb);
        }
        builder->SetInsertPoint(loop_bb);
        llvm::FunctionType *ft = llvm::FunctionType::get(i32_t, {i32_t, i32_t}, false);
        // LLVM IR-level inline asm: operands are $0/$1/$2 and `%` is literal (NOT a GCC-style
        // escape), so use a plain block-scoped predicate name `mbar_p` -- `%p` would emit invalid
        // `%%p` and also collide with the kernel's existing %p<N> predicate bank.
        llvm::InlineAsm *ia = llvm::InlineAsm::get(
            ft,
            "{ .reg .pred mbar_p;\n"
            "  mbarrier.try_wait.parity.shared.b64 mbar_p, [$1], $2;\n"
            "  selp.u32 $0, 1, 0, mbar_p; }",
            "=r,r,r", /*hasSideEffects*/ true);
        llvm::Value *done = builder->CreateCall(ia, {addr, parity});
        llvm::Value *ready = builder->CreateICmpNE(done, ConstantInt::get(i32_t, 0));
        builder->CreateCondBr(ready, after_bb, loop_bb);
        builder->SetInsertPoint(after_bb);
        // Once the mbarrier phase completes, the TMA/cp.async tile is in shared memory -- but it was
        // written through the ASYNC proxy. A wgmma operand read (or any generic-proxy access) is NOT
        // guaranteed to observe that write without an async-proxy fence. Emit it here, right after the
        // wait, so visibility no longer relies on incidental bar.sync ordering (Hopper TMA->wgmma; cf.
        // Triton's `fence.proxy.async.shared::cta` and CUTLASS fence_view_async_shared).
        {
            llvm::FunctionType *fft = llvm::FunctionType::get(llvm::Type::getVoidTy(*context), {}, false);
            llvm::InlineAsm *fence = llvm::InlineAsm::get(fft, "fence.proxy.async.shared::cta;",
                                                          "", /*hasSideEffects*/ true);
            builder->CreateCall(fence, {});
        }
        value = ConstantInt::get(i32_t, 0);
        return;
    }

    if (op->is_intrinsic() && op->name == "setmaxnreg") {
        // Hopper (sm_90) per-warp-group register reallocation. setmaxnreg.{dec,inc}.sync.aligned.u32
        // is a WARP-GROUP-COLLECTIVE instruction: all 128 threads of the warp group must execute it
        // converged, so it is emitted unguarded at the entry of the warp group's branch (the branch's
        // group-range guard already restricts it to that group's lanes -- it must NOT be wrapped in a
        // single-thread predicate). dec deallocates this group's registers down to N (a producer/DMA
        // group frees registers); inc allocates up to N (a wgmma consumer group). N must be a
        // compile-time immediate multiple of 8 in [24, 256]. Args: (regs, increase).
        internal_assert(op->args.size() == 2u) << "setmaxnreg expects (regs, increase).\n";
        auto regs = as_const_int(op->args[0]);
        auto increase = as_const_int(op->args[1]);
        internal_assert(regs && increase)
            << "setmaxnreg expects compile-time-constant (regs, increase).\n";
        std::string asm_str =
            "setmaxnreg." + std::string(*increase != 0 ? "inc" : "dec") + ".sync.aligned.u32 $0;";
        llvm::FunctionType *ft = llvm::FunctionType::get(void_t, {i32_t}, false);
        // "n" constraint = a compile-time integer immediate (setmaxnreg requires the count be
        // immediate). hasSideEffects so it is never reordered/elided.
        llvm::InlineAsm *ia = llvm::InlineAsm::get(ft, asm_str, "n", /*hasSideEffects*/ true);
        builder->CreateCall(ia, {ConstantInt::get(i32_t, (int)*regs)});
        value = ConstantInt::get(i32_t, 0);
        return;
    }

    if (op->is_intrinsic(Call::gpu_thread_barrier)) {
        // Even though we always insert a __syncthreads equivalent
        // (which has both a device and shared memory fence)
        // check to make sure the intrinsic has the right number of
        // arguments
        internal_assert(op->args.size() == 1) << "gpu_thread_barrier() intrinsic must specify memory fence type.\n";

        auto fence_type_ptr = as_const_int(op->args[0]);
        internal_assert(fence_type_ptr) << "gpu_thread_barrier() parameter is not a constant integer.\n";

        // P1: if cp.async copies are in flight, commit the group and wait for all of them
        // before the block sync, so the shared data is ready (synchronous cp.async). Each
        // thread waits its own group; the barrier then makes the shared writes visible.
        if (emitted_cp_async) {
            builder->CreateCall(llvm::Intrinsic::getOrInsertDeclaration(
                module.get(), llvm::Intrinsic::nvvm_cp_async_commit_group));
            builder->CreateCall(llvm::Intrinsic::getOrInsertDeclaration(
                                    module.get(), llvm::Intrinsic::nvvm_cp_async_wait_group),
                                builder->getInt32(0));
            emitted_cp_async = false;
        }

        llvm::Function *barrier;
        if ((barrier = module->getFunction("llvm.nvvm.barrier.cta.sync.aligned.all")) && barrier->getIntrinsicID() != 0) {
            // LLVM 20.1.6 and above: https://github.com/llvm/llvm-project/pull/140615
            builder->CreateCall(barrier, builder->getInt32(0));
        } else if ((barrier = module->getFunction("llvm.nvvm.barrier0")) && barrier->getIntrinsicID() != 0) {
            // LLVM 21.1.5 and below: Testing for llvm.nvvm.barrier0 can be removed once we drop support for LLVM 20
            builder->CreateCall(barrier);
        } else {
            internal_error << "Could not find PTX barrier intrinsic llvm.nvvm.barrier0 nor llvm.nvvm.barrier.cta.sync.aligned.all\n";
        }
        value = ConstantInt::get(i32_t, 0);
        return;
    }

    if (op->is_intrinsic(Call::gpu_named_barrier)) {
        // gpu_named_barrier(barrier_id, thread_count, mode):
        //   mode 0 -> bar.sync   barrier_id, thread_count  (wait)
        //   mode 1 -> bar.arrive barrier_id, thread_count  (non-blocking arrive)
        // These are *partial* (non-aligned) named CTA barriers: only thread_count
        // threads of the block participate, which is required for warp
        // specialization where producer and consumer warps run different code and
        // a whole-CTA __syncthreads would deadlock.
        internal_assert(op->args.size() == 3)
            << "gpu_named_barrier() intrinsic expects (barrier_id, thread_count, mode).\n";
        auto mode = as_const_int(op->args[2]);
        internal_assert(mode && (*mode == 0 || *mode == 1))
            << "gpu_named_barrier() mode must be the constant 0 (wait) or 1 (arrive).\n";

        // P1b: a warp-spec producer that issued cp.async copies into a shared ring slot
        // must commit + wait for them before its full-release barrier, so the consumer
        // sees the data. The overlap comes from the producer running ahead (ring_buffer):
        // it issues slot k+1's cp.async while the consumer computes slot k.
        if (emitted_cp_async) {
            builder->CreateCall(llvm::Intrinsic::getOrInsertDeclaration(
                module.get(), llvm::Intrinsic::nvvm_cp_async_commit_group));
            builder->CreateCall(llvm::Intrinsic::getOrInsertDeclaration(
                                    module.get(), llvm::Intrinsic::nvvm_cp_async_wait_group),
                                builder->getInt32(0));
            emitted_cp_async = false;
        }

        Value *barrier_id = codegen(op->args[0]);
        Value *thread_count = codegen(op->args[1]);
        llvm::Intrinsic::ID id = (*mode == 1)
                                     ? llvm::Intrinsic::nvvm_barrier_cta_arrive_count
                                     : llvm::Intrinsic::nvvm_barrier_cta_sync_count;
        llvm::Function *barrier = llvm::Intrinsic::getOrInsertDeclaration(module.get(), id);
        internal_assert(barrier) << "Could not find PTX named-barrier intrinsic.\n";
        builder->CreateCall(barrier, {barrier_id, thread_count});
        value = ConstantInt::get(i32_t, 0);
        return;
    }

    // TODO: It would be better if CodeGen_LLVM could handle overloaded intrin calls by default.
    value = call_overloaded_intrin(op->type, op->name, op->args);
    if (!value) {
        CodeGen_LLVM::visit(op);
    }
}

string CodeGen_PTX_Dev::simt_intrinsic(const string &name) {
    if (ends_with(name, gpu_thread_name(0))) {
        return "llvm.nvvm.read.ptx.sreg.tid.x";
    } else if (ends_with(name, gpu_thread_name(1))) {
        return "llvm.nvvm.read.ptx.sreg.tid.y";
    } else if (ends_with(name, gpu_thread_name(2))) {
        return "llvm.nvvm.read.ptx.sreg.tid.z";
    } else if (ends_with(name, gpu_block_name(0))) {
        return "llvm.nvvm.read.ptx.sreg.ctaid.x";
    } else if (ends_with(name, gpu_block_name(1))) {
        return "llvm.nvvm.read.ptx.sreg.ctaid.y";
    } else if (ends_with(name, gpu_block_name(2))) {
        return "llvm.nvvm.read.ptx.sreg.ctaid.z";
    }
    internal_error << "simt_intrinsic called on bad variable name\n";
    return "";
}

void CodeGen_PTX_Dev::visit(const For *loop) {
    if (is_gpu(loop->for_type)) {
        Expr simt_idx = Call::make(Int(32), simt_intrinsic(loop->name), std::vector<Expr>(), Call::Extern);
        internal_assert(is_const_zero(loop->min));
        sym_push(loop->name, codegen(simt_idx));
        codegen(loop->body);
        sym_pop(loop->name);
    } else {
        CodeGen_LLVM::visit(loop);
    }
}

void CodeGen_PTX_Dev::visit(const Allocate *alloc) {
    user_assert(!alloc->new_expr.defined()) << "Allocate node inside PTX kernel has custom new expression.\n"
                                            << "(Memoization is not supported inside GPU kernels at present.)\n";
    if (alloc->memory_type == MemoryType::GPUShared) {
        // PTX uses zero in address space 3 as the base address for shared memory
        Value *shared_base = Constant::getNullValue(PointerType::get(*context, 3));
        sym_push(alloc->name, shared_base);
        if (alloc->swizzle.defined()) {
            shared_swizzles[alloc->name] = {alloc->swizzle, alloc->type.bytes()};
        }
    } else {
        debug(2) << "Allocate " << alloc->name << " on device\n";

        string allocation_name = alloc->name;
        debug(3) << "Pushing allocation called " << allocation_name << " onto the symbol table\n";

        // Jump back to the entry and generate an alloca. Note that by
        // jumping back we're rendering any expression we carry back
        // meaningless, so we had better only be dealing with
        // constants here.
        int32_t size = alloc->constant_allocation_size();
        internal_assert(size > 0)
            << "Allocation " << alloc->name << " has a dynamic size. "
            << "This should have been moved to the heap by the "
            << "fuse_gpu_thread_loops lowering pass.\n";

        BasicBlock *here = builder->GetInsertBlock();

        builder->SetInsertPoint(entry_block);
        Value *ptr = builder->CreateAlloca(llvm_type_of(alloc->type), ConstantInt::get(i32_t, size));
        builder->SetInsertPoint(here);
        sym_push(allocation_name, ptr);
    }
    codegen(alloc->body);
}

llvm::Value *CodeGen_PTX_Dev::codegen_swizzled_index(const std::string &buffer, Type type, llvm::Value *index) {
    auto it = shared_swizzles.find(buffer);
    if (it == shared_swizzles.end()) {
        return index;
    }
    const SwizzleLayout &s = it->second.first;
    const int alloc_bytes = it->second.second;
    const int access_bytes = type.bytes();

    // The swizzle params are in units of the allocation's element. If this access
    // is wider (e.g. the 4-wide u128 store/load path), its index is in coarser
    // units, so shift the params down by log2(access/alloc); the granule >= access
    // invariant guarantees they stay non-negative. A narrower access shifts up.
    int base = s.base;
    int shift = s.shift;
    if (access_bytes > alloc_bytes) {
        int ratio = access_bytes / alloc_bytes, lg = 0;
        while (ratio > 1) {
            ratio >>= 1;
            lg++;
        }
        internal_assert(base >= lg && shift >= lg)
            << "Swizzled access (" << access_bytes << "B) is wider than the swizzle granule "
            << "for " << buffer << "; granule must be >= the access width.\n";
        base -= lg;
        shift -= lg;
    } else if (alloc_bytes > access_bytes) {
        int ratio = alloc_bytes / access_bytes, lg = 0;
        while (ratio > 1) {
            ratio >>= 1;
            lg++;
        }
        base += lg;
        shift += lg;
    }

    // phys = i ^ (((i >> shift) & ((1 << bits) - 1)) << base), elementwise.
    llvm::Type *ty = index->getType();
    llvm::Type *scalar_ty = ty->getScalarType();
    auto konst = [&](uint64_t v) -> llvm::Value * {
        llvm::Constant *c = llvm::ConstantInt::get(scalar_ty, v);
        if (ty->isVectorTy()) {
            auto lanes = llvm::cast<llvm::FixedVectorType>(ty)->getNumElements();
            c = llvm::ConstantVector::getSplat(llvm::ElementCount::getFixed(lanes), c);
        }
        return c;
    };
    llvm::Value *field = builder->CreateLShr(index, konst(shift));
    field = builder->CreateAnd(field, konst((uint64_t(1) << s.bits) - 1));
    field = builder->CreateShl(field, konst(base));
    return builder->CreateXor(index, field);
}

void CodeGen_PTX_Dev::visit(const Free *f) {
    sym_pop(f->name);
}

void CodeGen_PTX_Dev::visit(const AssertStmt *op) {
    // Discard the error message for now.
    Expr trap = Call::make(Int(32), "halide_ptx_trap", {}, Call::Extern);
    codegen(IfThenElse::make(!op->condition, Evaluate::make(trap)));
}

void CodeGen_PTX_Dev::visit(const Load *op) {

    // Do aligned 4-wide 32-bit loads as a single i128 load.
    const Ramp *r = op->index.as<Ramp>();
    // TODO: lanes >= 4, not lanes == 4
    if (is_const_one(op->predicate) && r && is_const_one(r->stride) && r->lanes == 4 && op->type.bits() == 32) {
        ModulusRemainder align = op->alignment;
        if (align.modulus % 4 == 0 && align.remainder % 4 == 0) {
            Expr index = simplify(r->base / 4);
            Expr equiv = Load::make(UInt(128), op->name, index,
                                    op->image, op->param, const_true(), align / 4);
            equiv = reinterpret(op->type, equiv);
            codegen(equiv);
            return;
        }
    }

    CodeGen_LLVM::visit(op);
}

void CodeGen_PTX_Dev::visit(const Store *op) {
    // P1: a vectorized (4-wide, 16 B) contiguous copy from global to shared lowers to
    // `cp.async.cg.shared.global.16`, which copies global->shared directly (bypassing
    // registers) — the codegen of a vectorized shared<-global copy, gated on sm_80+.
    // Detect `Store(shared, Load(global))` with matching unit-stride 4-wide ramps. The
    // following gpu_thread_barrier commits + waits (synchronous; ring_buffer adds overlap).
    if (!emit_atomic_stores && target.get_cuda_capability_lower_bound() >= 80 &&
        get_env_variable("HL_NO_CP_ASYNC").empty() &&
        is_const_one(op->predicate)) {
        const Ramp *r = op->index.as<Ramp>();
        const Load *ld = op->value.as<Load>();
        const Ramp *lr = ld ? ld->index.as<Ramp>() : nullptr;
        // cp.async.cg moves a 16-byte chunk; accept any element width whose contiguous
        // vector is 16 B -- 4-wide 32-bit, or 8-wide 16-bit (the f16 core-matrix ki run
        // produced by a vectorized operand staging copy, P3/1.2).
        int payload_bits = r ? op->value.type().bits() * r->lanes : 0;
        if (r && ld && lr && is_const_one(ld->predicate) &&
            is_const_one(r->stride) && is_const_one(lr->stride) &&
            payload_bits == 128 && lr->lanes == r->lanes) {
            Value *dst = codegen_buffer_pointer(op->name, op->value.type().element_of(), r->base);
            Value *src = codegen_buffer_pointer(ld->name, ld->type.element_of(), lr->base);
            // cp.async copies global (as1) -> shared (as3). dst must be shared; src must be
            // global (cast a generic as0 pointer to as1; skip a shared->shared copy).
            unsigned src_as = src->getType()->getPointerAddressSpace();
            if (dst->getType()->getPointerAddressSpace() == 3 && src_as != 3) {
                if (src_as != 1) {
                    src = builder->CreateAddrSpaceCast(src, llvm::PointerType::get(*context, 1));
                }
                llvm::Function *cp = llvm::Intrinsic::getOrInsertDeclaration(
                    module.get(), llvm::Intrinsic::nvvm_cp_async_cg_shared_global_16);
                builder->CreateCall(cp, {dst, src});
                emitted_cp_async = true;
                return;
            }
        }
    }

    // Issue atomic store if we are inside an Atomic node.
    if (emit_atomic_stores) {
        user_assert(is_const_one(op->predicate)) << "Atomic update does not support predicated store.\n";
        user_assert(op->value.type().bits() >= 32) << "CUDA: 8-bit or 16-bit atomics are not supported.\n";
    }

    // Do aligned 4-wide 32-bit stores as a single i128 store.
    const Ramp *r = op->index.as<Ramp>();
    // TODO: lanes >= 4, not lanes == 4
    if (is_const_one(op->predicate) && r && is_const_one(r->stride) && r->lanes == 4 && op->value.type().bits() == 32) {
        ModulusRemainder align = op->alignment;
        if (align.modulus % 4 == 0 && align.remainder % 4 == 0) {
            Expr index = simplify(r->base / 4);
            Expr value = reinterpret(UInt(128), op->value);
            Stmt equiv = Store::make(op->name, value, index, op->param, const_true(), align / 4);
            codegen(equiv);
            return;
        }
    }

    CodeGen_LLVM::visit(op);
}

void CodeGen_PTX_Dev::visit(const Atomic *op) {
    // CUDA requires all the threads in a warp to perform the same operations,
    // which means our mutex will lead to deadlock.
    user_assert(op->mutex_name.empty())
        << "The atomic update requires a mutex lock, which is not supported in CUDA.\n";

    // Issue atomic stores.
    ScopedValue<bool> old_emit_atomic_stores(emit_atomic_stores, true);
    CodeGen_LLVM::visit(op);
}

// The NVPTX backend generates really terrible code if loads aren't 32-bit. This
// mutator replaces 8- or 16-bit loads aligned to 32-bits with 32-bit loads of fewer
// lanes instead.
class RewriteLoadsAs32Bit : public IRMutator {
    using IRMutator::visit;

    Expr visit(const Load *op) override {
        if (op->type.is_scalar() || op->type.bits() * op->type.lanes() < 32) {
            return IRMutator::visit(op);
        }

        Expr index = mutate(op->index);
        int sub_lanes = 32 / op->type.bits();
        const Ramp *idx = index.as<Ramp>();
        if (idx &&
            is_const_one(op->predicate) &&
            is_const_one(idx->stride) &&
            op->alignment.modulus % sub_lanes == 0 &&
            op->alignment.remainder % sub_lanes == 0) {
            Expr new_idx = simplify(idx->base / sub_lanes);
            int load_lanes = op->type.lanes() / sub_lanes;
            if (op->type.lanes() > sub_lanes) {
                new_idx = Ramp::make(new_idx, 1, load_lanes);
            }
            Expr new_load = Load::make(Int(32, load_lanes), op->name, new_idx, op->image, op->param, const_true(load_lanes), op->alignment / sub_lanes);
            return reinterpret(op->type, new_load);
        } else if (index.same_as(op->index)) {
            return op;
        } else {
            return Load::make(op->type, op->name, std::move(index), op->image, op->param, op->predicate, op->alignment);
        }
    }
};

llvm::Value *CodeGen_PTX_Dev::build_wgmma_descriptor(const std::string &buffer, Type elem_type,
                                                    const Expr &tile_origin, int lbo_bytes, int sbo_bytes,
                                                    int swizzle_bytes) {
    // Pointer to the operand tile origin. The shared base is null in addrspace(3)
    // (see visit(Allocate)), so the addrspace(3) pointer's integer value IS the
    // shared-window byte offset of the tile -- exactly the descriptor's start
    // address. (Dynamic shared starts at offset 0; no cvta needed for M0. If
    // static shared is ever reserved before the dynamic region this needs a
    // cvta.generic.to.shared instead.)
    llvm::Value *base_ptr = codegen_buffer_pointer(buffer, elem_type, tile_origin);
    llvm::Value *addr = builder->CreatePtrToInt(base_ptr, i64_t);

    // Hopper wgmma shared matrix descriptor (PTX ISA 8.0, "Matrix Descriptor"):
    //   bits [0:14)  start address  (byte addr >> 4)
    //   bits [16:30) leading-dim byte offset (LBO >> 4)
    //   bits [32:46) stride-dim byte offset  (SBO >> 4)
    //   bits [49:52) matrix base offset (0 for M0, no swizzle)
    //   bits [62:64) swizzle mode (0 = none)
    auto enc14 = [&](llvm::Value *v) {
        llvm::Value *sh = builder->CreateLShr(v, llvm::ConstantInt::get(i64_t, 4));
        return builder->CreateAnd(sh, llvm::ConstantInt::get(i64_t, 0x3FFF));
    };
    llvm::Value *desc = enc14(addr);
    auto or_field = [&](uint64_t value14, int shift) {
        uint64_t f = ((value14 >> 4) & 0x3FFFull) << shift;
        if (f) {
            desc = builder->CreateOr(desc, llvm::ConstantInt::get(i64_t, f));
        }
    };
    // Swizzle mode (bits [62:64)): 0 none / 1 128B / 2 64B / 3 32B (CUTLASS LayoutType). The
    // operand's shared layout is the store_in/TMA swizzle, so the descriptor reads it with the
    // matching mode (M5b). For a swizzled descriptor the LBO/SBO are the canonical swizzle-atom
    // values, NOT the dense core-matrix ones the caller passes -- the hardware derives the core
    // matrices from (start, swizzle). 128B K-major f16: LBO=16, SBO=1024 (fast.cu matmul_3
    // make_smem_desc; the address is the per-chunk LOGICAL tile offset, swizzle applied by HW).
    int swz_mode = swizzle_bytes == 128 ? 1 : swizzle_bytes == 64 ? 2 :
                   swizzle_bytes == 32  ? 3 : 0;
    if (swz_mode) {
        lbo_bytes = 16;
        sbo_bytes = swizzle_bytes == 128 ? 1024 : swizzle_bytes == 64 ? 512 : 256;
    }
    or_field((uint64_t)lbo_bytes, 16);
    or_field((uint64_t)sbo_bytes, 32);
    if (swz_mode) {
        desc = builder->CreateOr(desc, llvm::ConstantInt::get(i64_t, (uint64_t)swz_mode << 62));
    }
    return desc;
}

int CodeGen_PTX_Dev::operand_swizzle_bytes(const std::string &buffer) const {
    auto it = shared_swizzles.find(buffer);
    if (it == shared_swizzles.end()) {
        return 0;
    }
    const SwizzleLayout &s = it->second.first;
    return s.defined() ? (1 << (s.bits + 4)) : 0;  // bits 1/2/3 -> 32/64/128 B
}

llvm::Value *CodeGen_PTX_Dev::emit_wgmma(int n, llvm::Value *acc, llvm::Value *desc_a,
                                         llvm::Value *desc_b, bool scale_d) {
    // The per-thread accumulator fragment is {f32 x R}, R = N/2 (m64nNk16 f32). The descriptors
    // are operands $R and $R+1. Build the operand list dynamically so N scales (n16 R=8 ... n256
    // R=128). wgmma.mma_async.sync.aligned.m64nNk16.f32.f16.f16 {d0..d_{R-1}}, descA, descB,
    // scaleD, 1, 1, 0, 0; -- R read-modify-write f32 accumulators (outputs tied to inputs).
    internal_assert(n >= 16 && n % 8 == 0 && n <= 256) << "wgmma N out of range: " << n << "\n";
    const int R = n / 2;
    llvm::Type *f32 = llvm::Type::getFloatTy(*context);
    llvm::Type *i64 = llvm::Type::getInt64Ty(*context);
    llvm::StructType *acc_ty = llvm::StructType::get(*context, std::vector<llvm::Type *>(R, f32));

    std::string regs, tied, outs;
    for (int i = 0; i < R; i++) {
        regs += (i ? ",$" : "$") + std::to_string(i);
        tied += std::to_string(i) + ",";
        outs += "=f,";
    }
    // descA/descB operand numbers: the R tied inputs consume operand slots R..2R-1, so the two
    // descriptor inputs are $(2R) and $(2R+1) -- NOT $R/$R+1 (which alias the tied accumulators).
    const std::string scale = scale_d ? "1" : "0";
    const std::string asm_str =
        "wgmma.mma_async.sync.aligned.m64n" + std::to_string(n) + "k16.f32.f16.f16 {" +
        regs + "}, $" + std::to_string(2 * R) + ", $" + std::to_string(2 * R + 1) + ", " +
        scale + ", 1, 1, 0, 0;";
    const std::string constraints = outs + tied + "l,l";  // R outputs, R tied inputs, descA, descB

    std::vector<llvm::Type *> arg_tys(R, f32);
    arg_tys.push_back(i64);
    arg_tys.push_back(i64);
    llvm::FunctionType *fn_ty = llvm::FunctionType::get(acc_ty, arg_tys, false);
    // wgmma has side effects (mutates the async accumulator state) -> must not be DCE'd/reordered.
    llvm::InlineAsm *ia = llvm::InlineAsm::get(fn_ty, asm_str, constraints, /*hasSideEffects*/ true);

    std::vector<llvm::Value *> args;
    args.reserve(R + 2);
    for (int i = 0; i < R; i++) {
        args.push_back(builder->CreateExtractValue(acc, i));
    }
    args.push_back(desc_a);
    args.push_back(desc_b);
    return builder->CreateCall(ia, args);
}

void CodeGen_PTX_Dev::codegen_vector_reduce(const VectorReduce *op, const Expr &init) {
    // Unified collective recognizer, reduce family: decompose on the vector's realization
    // (carried ON the node, stamped pre-fusion from the gpu_warps/gpu_lanes scope, so it
    // survives thread fusion to here). See research/gpu_recognizer_design.md.
    //   - WarpGroup -> wgmma (the new row; added at M-mma). No existing kernel produces a
    //     WarpGroup reduce, so falling back here is new-but-inert today.
    //   - Register and Warp fall through to the dp4a/dp2a table + fma fallback below — the
    //     current behavior for both (the Warp arm gains mma.sync/redux later, branching here
    //     before the table). Keeping Warp on the table keeps gpu_lanes reduces byte-identical.
    if (op->realization == GPUVectorScope::WarpGroup) {
        CodeGen_LLVM::codegen_vector_reduce(op, init);
        return;
    }

    // Pattern match 8/16-bit dot products
    struct Pattern {
        VectorReduce::Operator op;
        int factor;
        Expr pattern;
        const char *name;
        int flags;
        enum {
            SwapOps = 1 << 0,  // This happens before narrowing op 1 below.
            NarrowOp1 = 1 << 1,
        };
    };
    static Expr wild_i8x = Variable::make(Int(8, 0), "*");
    static Expr wild_u8x = Variable::make(UInt(8, 0), "*");
    static Expr wild_i16x = Variable::make(Int(16, 0), "*");
    static Expr wild_u16x = Variable::make(UInt(16, 0), "*");
    // TODO: Support rewriting to arbitrary calls in IRMatch and use that instead
    // of expr_match here. That would probably allow avoiding the redundant swapping
    // operands logic.
    static const Pattern patterns[] = {
        {VectorReduce::Add, 4, i32(widening_mul(wild_i8x, wild_i8x)), "dp4a"},
        {VectorReduce::Add, 4, i32(widening_mul(wild_i8x, wild_u8x)), "dp4a"},
        {VectorReduce::Add, 4, i32(widening_mul(wild_u8x, wild_i8x)), "dp4a"},
        {VectorReduce::Add, 4, u32(widening_mul(wild_u8x, wild_u8x)), "dp4a"},
        {VectorReduce::Add, 4, widening_mul(wild_i16x, wild_i16x), "dp2a", Pattern::NarrowOp1},
        {VectorReduce::Add, 4, widening_mul(wild_i16x, wild_u16x), "dp2a", Pattern::NarrowOp1},
        {VectorReduce::Add, 4, widening_mul(wild_u16x, wild_i16x), "dp2a", Pattern::NarrowOp1},
        {VectorReduce::Add, 4, widening_mul(wild_u16x, wild_u16x), "dp2a", Pattern::NarrowOp1},
        {VectorReduce::Add, 4, widening_mul(wild_i16x, wild_i16x), "dp2a", Pattern::SwapOps | Pattern::NarrowOp1},
        {VectorReduce::Add, 4, widening_mul(wild_u16x, wild_i16x), "dp2a", Pattern::SwapOps | Pattern::NarrowOp1},
        {VectorReduce::Add, 4, widening_mul(wild_i16x, wild_u16x), "dp2a", Pattern::SwapOps | Pattern::NarrowOp1},
        {VectorReduce::Add, 4, widening_mul(wild_u16x, wild_u16x), "dp2a", Pattern::SwapOps | Pattern::NarrowOp1},
    };

    const int input_lanes = op->value.type().lanes();
    const int factor = input_lanes / op->type.lanes();

    std::vector<Expr> matches;
    for (const Pattern &p : patterns) {
        if (p.op != op->op || factor % p.factor != 0) {
            continue;
        }
        if (!expr_match(p.pattern, op->value, matches)) {
            continue;
        }
        Expr a = matches[0];
        Expr b = matches[1];
        if (p.flags & Pattern::SwapOps) {
            std::swap(a, b);
        }
        if (p.flags & Pattern::NarrowOp1) {
            // This pattern needs the second operand to be narrowed further.
            Expr b_narrow = lossless_cast(b.type().narrow(), b);
            if (!b_narrow.defined()) {
                b_narrow = lossless_cast(b.type().narrow().with_code(halide_type_uint), b);
                if (!b_narrow.defined()) {
                    continue;
                }
            }
            b = b_narrow;
        }
        Expr i = init;
        if (!i.defined()) {
            i = cast(op->value.type(), 0);
        }

        vector<Expr> result;
        for (int l = 0; l < op->type.lanes(); l++) {
            // To compute a single lane of the output, we'll
            // extract the appropriate slice of the args, which
            // have been reinterpreted as 32-bit vectors, then
            // call either dp4a or dp2a the appropriate number of
            // times, and finally sum the result.
            Expr i_slice = Shuffle::make_extract_element(i, l);
            for (int i = 0; i < factor; i += p.factor) {
                Expr a_slice = Shuffle::make_slice(a, i + l * factor, 1, p.factor);
                Expr b_slice = Shuffle::make_slice(b, i + l * factor, 1, p.factor);
                i_slice = Call::make(i_slice.type(), p.name, {a_slice, b_slice, i_slice}, Call::PureExtern);
            }
            i_slice = RewriteLoadsAs32Bit()(i_slice);
            i_slice = simplify(i_slice);
            i_slice = common_subexpression_elimination(i_slice);
            result.push_back(i_slice);
        }
        // Concatenate the per-lane results to get the full vector result
        Expr equiv = Shuffle::make_concat(result);
        equiv.accept(this);
        return;
    }
    CodeGen_LLVM::codegen_vector_reduce(op, init);
}

string CodeGen_PTX_Dev::mcpu_target() const {
    if (target.has_feature(Target::CUDACapability90)) {
        // The architecture-specific "a" variant is required for the Hopper
        // tensor-core / wgmma instructions (sm_90a), matching CUTLASS/nvcc.
        return "sm_90a";
    } else if (target.has_feature(Target::CUDACapability86)) {
        return "sm_86";
    } else if (target.has_feature(Target::CUDACapability80)) {
        return "sm_80";
    } else if (target.has_feature(Target::CUDACapability75)) {
        return "sm_75";
    } else if (target.has_feature(Target::CUDACapability70)) {
        return "sm_70";
    } else if (target.has_feature(Target::CUDACapability61)) {
        return "sm_61";
    } else if (target.has_feature(Target::CUDACapability50)) {
        return "sm_50";
    } else if (target.has_feature(Target::CUDACapability35)) {
        return "sm_35";
    } else if (target.has_feature(Target::CUDACapability32)) {
        return "sm_32";
    } else if (target.has_feature(Target::CUDACapability30)) {
        return "sm_30";
    } else {
        return "sm_20";
    }
}

string CodeGen_PTX_Dev::mcpu_tune() const {
    return mcpu_target();
}

string CodeGen_PTX_Dev::mattrs() const {
    if (target.has_feature(Target::CUDACapability90)) {
        // PTX ISA 8.0 is the floor for the Hopper wgmma.mma_async family.
        return "+ptx80";
    } else if (target.has_feature(Target::CUDACapability86)) {
        return "+ptx71";
    } else if (target.has_feature(Target::CUDACapability80)) {
        return "+ptx70";
    } else if (target.has_feature(Target::CUDACapability75)) {
        return "+ptx63";
    } else if (target.has_feature(Target::CUDACapability70)) {
        return "+ptx60";
    } else if (target.has_feature(Target::CUDACapability61)) {
        return "+ptx50";
    } else if (target.features_any_of({Target::CUDACapability32,
                                       Target::CUDACapability50})) {
        // sm_32 needs ptx isa 4.0 even though it seems to break the ordering
        return "+ptx40";
    } else if (target.features_any_of({Target::CUDACapability35,
                                       Target::CUDACapability30})) {
        return "+ptx32";
    }
    // Let LLVM pick
    return "";
}

bool CodeGen_PTX_Dev::use_soft_float_abi() const {
    return false;
}

vector<char> CodeGen_PTX_Dev::compile_to_src() {
    debug(2) << "In CodeGen_PTX_Dev::compile_to_src";

    // DISABLED - hooked in here to force PrintBeforeAll option - seems to be the only way?
    /*char* argv[] = { "llc", "-print-before-all" };*/
    /*int argc = sizeof(argv)/sizeof(char*);*/
    /*cl::ParseCommandLineOptions(argc, argv, "Halide PTX internal compiler\n");*/

    // Allocate target machine (similar to code in CodeGen_Internal.cpp make_target_machine)
    std::string err_str;
    const llvm::Target *llvm_target = TargetRegistry::lookupTarget(
        module->getTargetTriple(),
        err_str);
    auto triple = llvm::Triple(module->getTargetTriple());
    internal_assert(llvm_target) << "Could not create LLVM target for " << triple.str() << "\n";

    TargetOptions options;
    options.AllowFPOpFusion = CodeGen_GPU_Dev::any_strict_float ? llvm::FPOpFusion::Strict : llvm::FPOpFusion::Fast;
#if LLVM_VERSION < 230
    options.NoInfsFPMath = !CodeGen_GPU_Dev::any_strict_float;
    options.NoNaNsFPMath = !CodeGen_GPU_Dev::any_strict_float;
#endif
    options.HonorSignDependentRoundingFPMathOption = !CodeGen_GPU_Dev::any_strict_float;
    options.NoZerosInBSS = false;
    options.GuaranteedTailCallOpt = false;

    std::unique_ptr<TargetMachine>
        target_machine(llvm_target->createTargetMachine(
            triple,
            mcpu_target(), mattrs(), options,
            llvm::Reloc::PIC_,
            llvm::CodeModel::Small,
            CodeGenOptLevel::Aggressive));

    internal_assert(target_machine.get()) << "Could not allocate target machine!";

    module->setDataLayout(target_machine->createDataLayout());

    // Set up passes
    llvm::SmallString<8> outstr;
    raw_svector_ostream ostream(outstr);
    ostream.SetUnbuffered();

    // NVidia's libdevice library uses a __nvvm_reflect to choose
    // how to handle denormalized numbers. (The pass replaces calls
    // to __nvvm_reflect with a constant via a map lookup. The inliner
    // pass then resolves these situations to fast code, often a single
    // instruction per decision point.)
    //
    // The default is (more) IEEE like handling. FTZ mode flushes them
    // to zero. (This may only apply to single-precision.)
    //
    // The libdevice documentation covers other options for math accuracy
    // such as replacing division with multiply by the reciprocal and
    // use of fused-multiply-add, but they do not seem to be controlled
    // by this __nvvvm_reflect mechanism and may be flags to earlier compiler
    // passes.
    const int kFTZDenorms = 1;

    // Insert a module flag for the FTZ handling.
    module->addModuleFlag(llvm::Module::Override, "nvvm-reflect-ftz",
                          kFTZDenorms);

    if (kFTZDenorms) {
        for (llvm::Function &fn : *module) {
            fn.addFnAttr("nvptx-f32ftz", "true");
        }
    }

    const bool do_loop_opt = get_target().has_feature(Target::EnableLLVMLoopOpt);

    // Define and run optimization pipeline with new pass manager
    PipelineTuningOptions pto;
    pto.LoopInterleaving = do_loop_opt;
    pto.LoopVectorization = do_loop_opt;
    pto.SLPVectorization = true;  // Note: SLP vectorization has no analogue in the Halide scheduling model
    pto.LoopUnrolling = do_loop_opt;
    pto.ForgetAllSCEVInLoopUnroll = true;

    llvm::PassBuilder pb(target_machine.get(), pto);

    // These analysis managers have to be declared in this order.
    llvm::LoopAnalysisManager lam;
    llvm::FunctionAnalysisManager fam;
    llvm::CGSCCAnalysisManager cgam;
    llvm::ModuleAnalysisManager mam;

    // Register all the basic analyses with the managers.
    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    pb.crossRegisterProxies(lam, fam, cgam, mam);
    ModulePassManager mpm;

    using OptimizationLevel = llvm::OptimizationLevel;
    OptimizationLevel level = OptimizationLevel::O3;

    target_machine->registerPassBuilderCallbacks(pb);

    mpm = pb.buildPerModuleDefaultPipeline(level);
    mpm.run(*module, mam);

    if (llvm::verifyModule(*module, &errs())) {
        report_fatal_error("Transformation resulted in an invalid module\n");
    }

    // Optimization pipeline completed; run codegen pipeline

    // NOTE: use of the "legacy" PassManager here is still required; it is deprecated
    // for optimization, but is still the only complete API for codegen as of work-in-progress
    // LLVM14. At the time of this comment (Dec 2021), there is no firm plan as to when codegen will
    // be fully available in the new PassManager, so don't worry about this 'legacy'
    // tag until there's any indication that the old APIs start breaking.
    //
    // See:
    // https://lists.llvm.org/pipermail/llvm-dev/2021-April/150100.html
    // https://releases.llvm.org/13.0.0/docs/ReleaseNotes.html#changes-to-the-llvm-ir
    // https://groups.google.com/g/llvm-dev/c/HoS07gXx0p8
    legacy::PassManager module_pass_manager;
    module_pass_manager.add(createTargetTransformInfoWrapperPass(target_machine->getTargetIRAnalysis()));

    // Override default to generate verbose assembly.
    target_machine->Options.MCOptions.AsmVerbose = true;

    // Output string stream

    // Ask the target to add backend passes as necessary.
    bool fail = target_machine->addPassesToEmitFile(module_pass_manager, ostream, nullptr,
                                                    CodeGenFileType::AssemblyFile, true);
    internal_assert(!fail) << "Failed to set up passes to emit PTX source\n";
    module_pass_manager.run(*module);

    // Codegen pipeline completed.
    debug(2) << [&] {
        dump();
        return "Done with CodeGen_PTX_Dev::compile_to_src";
    }();

    debug(1) << "PTX kernel:\n"
             << outstr.c_str() << "\n";

    vector<char> buffer(outstr.begin(), outstr.end());

    // Dump the SASS too if the cuda SDK is in the path
    debug(2) << "Compiling PTX to SASS. Will fail if CUDA SDK is not installed (and in the path).\n";
    debug(2) << [&] {
        TemporaryFile ptx(get_current_kernel_name(), ".ptx");
        TemporaryFile sass(get_current_kernel_name(), ".sass");

        std::ofstream f(ptx.pathname());
        f.write(buffer.data(), buffer.size());
        f.close();

        if (run_process({"ptxas", "--gpu-name", mcpu_target(), ptx.pathname(), "-o", sass.pathname()}) == 0) {
            (void)run_process({"nvdisasm", sass.pathname()});  // Don't care if it fails
        }

        // Note: It works to embed the contents of the .sass file in
        // the buffer instead of the ptx source, and this could help
        // with app startup times. Expose via the target?
        /*
        {
            std::ifstream f(sass.pathname());
            buffer.clear();
            f.seekg(0, std::ios_base::end);
            std::streampos sz = f.tellg();
            buffer.resize(sz);
            f.seekg(0, std::ios_base::beg);
            f.read(buffer.data(), sz);
        }
        */
        return "";
    }();

    // Null-terminate the ptx source
    buffer.push_back(0);
    return buffer;
}

int CodeGen_PTX_Dev::native_vector_bits() const {
    // PTX doesn't really do vectorization. The widest type is a double.
    return 64;
}

string CodeGen_PTX_Dev::get_current_kernel_name() {
    return get_llvm_function_name(function);
}

void CodeGen_PTX_Dev::dump() {
    module->print(dbgs(), nullptr, false, true);
}

std::string CodeGen_PTX_Dev::print_gpu_name(const std::string &name) {
    return name;
}

bool CodeGen_PTX_Dev::supports_atomic_add(const Type &t) const {
    if (t.bits() < 32) {
        // TODO: Half atomics are supported by compute capability 7.x or higher.
        return false;
    }
    if (t.is_int_or_uint()) {
        return true;
    }
    if (t.is_float() && t.bits() == 32) {
        return true;
    }
    if (t.is_float() && t.bits() == 64) {
        // double atomics are supported since CC6.1
        return target.get_cuda_capability_lower_bound() >= 61;
    }
    return false;
}

}  // namespace

std::unique_ptr<CodeGen_GPU_Dev> new_CodeGen_PTX_Dev(const Target &target) {
    return std::make_unique<CodeGen_PTX_Dev>(target);
}

#else  // WITH_PTX

std::unique_ptr<CodeGen_GPU_Dev> new_CodeGen_PTX_Dev(const Target &target) {
    user_error << "PTX not enabled for this build of Halide.\n";
    return nullptr;
}

#endif  // WITH_PTX

}  // namespace Internal
}  // namespace Halide
