# Mamba-2's SSD recurrence on the tensor cores. Same story as cuda_attention:
# no capability is named so it is detected from the host.
HL_TARGET ?= host-cuda

export HL_LLVM_ARGS ?= -nvptx-no-f32x2

include ../support/Makefile.inc

# One head's worth of state and channels, the chunk the recurrence is blocked
# into, how long a sequence is, and how many (batch * head) sequences there
# are. The last is where all the parallelism comes from: the chunks of one
# sequence have to be walked in order.
SEQ ?= 4096
STATE ?= 64
CHANNELS ?= 64
CHUNK ?= 64
HEADS ?= 128
GROUPS ?= 1

CUDA_SDK ?= /usr/local/cuda
CXXFLAGS += -I $(CUDA_SDK)/include
LDFLAGS += -L $(CUDA_SDK)/lib64 -Wl,-rpath,$(CUDA_SDK)/lib64

# inductive, rdom_undef, or rdom - see the generator.
SCAN ?= inductive
WMMA ?= false


SHAPE = seq=$(SEQ) state=$(STATE) channels=$(CHANNELS) chunk=$(CHUNK) heads=$(HEADS) groups=$(GROUPS) wmma=$(WMMA)
SHAPE_FWD = $(SHAPE) scan=$(SCAN)
CXXFLAGS += -DSEQ=$(SEQ) -DSTATE=$(STATE) -DCHANNELS=$(CHANNELS) -DCHUNK=$(CHUNK) -DHEADS=$(HEADS) -DGROUPS=$(GROUPS)

all: $(BIN)/$(HL_TARGET)/runner

# generate_filter_main catches and reports what a generator raises, but only
# when libHalide was built with WITH_EXCEPTIONS - the catch is compiled out
# otherwise, while the throw is not. The terminate handler prints the message
# either way, which is what the tests link it for.
TERMINATE_HANDLER = ../../test/common/terminate_handler.cpp

$(GENERATOR_BIN)/mamba2.generator: mamba2_generator.cpp $(TERMINATE_HANDLER) $(GENERATOR_DEPS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LIBHALIDE_LDFLAGS)

$(BIN)/%/mamba2.a: $(GENERATOR_BIN)/mamba2.generator
	@mkdir -p $(@D)
	$^ -g mamba2 -f mamba2 -e $(GENERATOR_OUTPUTS),conceptual_stmt_html -o $(@D) \
	    target=$* $(SHAPE_FWD)

# The ablation forms of the inductive state (scan=fusion and scan=slide in
# the generators), built into the runners beside the SCAN form when WMMA=true,
# whose schedule they take apart. Fusion recomputes the state from the first
# chunk at every chunk, quadratic in SEQ/CHUNK, so it is only built and timed
# with ABLATE_FUSION=1.
ABLATE_FUSION ?= 0
ABLATION = $(if $(filter true,$(WMMA)),1,0)
ABLATION_FORMS = $(if $(filter 1,$(ABLATION)),sld $(if $(filter 1,$(ABLATE_FUSION)),fus))
CXXFLAGS += -DABLATION=$(ABLATION) -DABLATE_FUSION=$(if $(filter 1,$(ABLATE_FUSION)),1,0)

$(BIN)/%/mamba2_sld.a: $(GENERATOR_BIN)/mamba2.generator
	@mkdir -p $(@D)
	$^ -g mamba2 -f mamba2_sld -e $(GENERATOR_OUTPUTS) -o $(@D) \
	    target=$* $(SHAPE) scan=slide

$(BIN)/%/mamba2_fus.a: $(GENERATOR_BIN)/mamba2.generator
	@mkdir -p $(@D)
	$^ -g mamba2 -f mamba2_fus -e $(GENERATOR_OUTPUTS) -o $(@D) \
	    target=$* $(SHAPE) scan=fusion

$(BIN)/%/runner: runner.cpp $(BIN)/%/mamba2.a $(foreach f,$(ABLATION_FORMS),$(BIN)/%/mamba2_$(f).a)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -I$(BIN)/$* -Wall $^ -o $@ $(LDFLAGS) -L $(HALIDE_DISTRIB_PATH)/lib -lHalide -Wl,-rpath,$(HALIDE_DISTRIB_PATH)/lib -lcuda -lcudart -ldl -lpthread -lz

test: $(BIN)/$(HL_TARGET)/runner
	$^

$(GENERATOR_BIN)/mamba2_bwd.generator: mamba2_bwd_generator.cpp $(TERMINATE_HANDLER) $(GENERATOR_DEPS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(filter-out %.h,$^) -o $@ $(LIBHALIDE_LDFLAGS)

$(BIN)/%/mamba2_bwd.a: $(GENERATOR_BIN)/mamba2_bwd.generator
	@mkdir -p $(@D)
	$^ -g mamba2_bwd -f mamba2_bwd -e $(GENERATOR_OUTPUTS),conceptual_stmt_html -o $(@D) \
	    target=$* $(SHAPE) scan=$(SCAN)

$(BIN)/%/mamba2_bwd_sld.a: $(GENERATOR_BIN)/mamba2_bwd.generator
	@mkdir -p $(@D)
	$^ -g mamba2_bwd -f mamba2_bwd_sld -e $(GENERATOR_OUTPUTS) -o $(@D) \
	    target=$* $(SHAPE) scan=slide

$(BIN)/%/mamba2_bwd_fus.a: $(GENERATOR_BIN)/mamba2_bwd.generator
	@mkdir -p $(@D)
	$^ -g mamba2_bwd -f mamba2_bwd_fus -e $(GENERATOR_OUTPUTS) -o $(@D) \
	    target=$* $(SHAPE) scan=fusion

$(BIN)/%/runner_bwd: runner_bwd.cpp $(BIN)/%/mamba2_bwd.a $(foreach f,$(ABLATION_FORMS),$(BIN)/%/mamba2_bwd_$(f).a)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -I$(BIN)/$* -Wall $^ -o $@ $(LDFLAGS) -L $(HALIDE_DISTRIB_PATH)/lib -lHalide -Wl,-rpath,$(HALIDE_DISTRIB_PATH)/lib -lcuda -lcudart -ldl -lpthread -lz

test_bwd: $(BIN)/$(HL_TARGET)/runner_bwd
	$^

clean:
	rm -rf $(BIN)

.SECONDARY:
