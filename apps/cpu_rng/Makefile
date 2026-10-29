HL_TARGET ?= host

include ../support/Makefile.inc

# Streams and steps per stream: the output is sized well past the last
# level cache, and the materialized state trajectory is four times it.
LANES ?= 32
STEPS ?= 4194304
PAR ?= false

$(GENERATOR_BIN)/rng.generator: rng_generator.cpp $(GENERATOR_DEPS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(filter %.cpp,$^) -o $@ $(LIBHALIDE_LDFLAGS)

$(BIN)/%/rng_ind.a: $(GENERATOR_BIN)/rng.generator
	@mkdir -p $(@D)
	$< -g rng -f rng_ind -e $(GENERATOR_OUTPUTS) -o $(@D) \
	    target=$*-no_runtime lanes=$(LANES) scan=inductive par=$(PAR)

$(BIN)/%/rng_unf.a: $(GENERATOR_BIN)/rng.generator
	@mkdir -p $(@D)
	$< -g rng -f rng_unf -e $(GENERATOR_OUTPUTS) -o $(@D) \
	    target=$*-no_runtime lanes=$(LANES) scan=unfolded par=$(PAR)

$(BIN)/%/rng_rdom.a: $(GENERATOR_BIN)/rng.generator
	@mkdir -p $(@D)
	$< -g rng -f rng_rdom -e $(GENERATOR_OUTPUTS) -o $(@D) \
	    target=$*-no_runtime lanes=$(LANES) scan=rdom par=$(PAR)

# The ablation forms (scan=fusion, slide, slide_unroll, slide_fold in the generator).
# Fusion recomputes the walk from the seeds every step, quadratic in STEPS,
# so it is only built and timed with ABLATE_FUSION=1, at a small STEPS.
ABLATE_FUSION ?= 0
ABLATION = rng_sld rng_slu rng_slf $(if $(filter 1,$(ABLATE_FUSION)),rng_fus)

$(BIN)/%/rng_fus.a: $(GENERATOR_BIN)/rng.generator
	@mkdir -p $(@D)
	$< -g rng -f rng_fus -e $(GENERATOR_OUTPUTS) -o $(@D) \
	    target=$*-no_runtime lanes=$(LANES) scan=fusion par=$(PAR)

$(BIN)/%/rng_sld.a: $(GENERATOR_BIN)/rng.generator
	@mkdir -p $(@D)
	$< -g rng -f rng_sld -e $(GENERATOR_OUTPUTS) -o $(@D) \
	    target=$*-no_runtime lanes=$(LANES) scan=slide par=$(PAR)

$(BIN)/%/rng_slu.a: $(GENERATOR_BIN)/rng.generator
	@mkdir -p $(@D)
	$< -g rng -f rng_slu -e $(GENERATOR_OUTPUTS) -o $(@D) \
	    target=$*-no_runtime lanes=$(LANES) scan=slide_unroll par=$(PAR)

$(BIN)/%/rng_slf.a: $(GENERATOR_BIN)/rng.generator
	@mkdir -p $(@D)
	$< -g rng -f rng_slf -e $(GENERATOR_OUTPUTS) -o $(@D) \
	    target=$*-no_runtime lanes=$(LANES) scan=slide_fold par=$(PAR)

$(BIN)/%/runtime.a: $(GENERATOR_BIN)/rng.generator
	@mkdir -p $(@D)
	$< -r runtime -o $(@D) target=$*

$(BIN)/%/runner: runner.cpp $(BIN)/%/rng_ind.a $(BIN)/%/rng_unf.a $(BIN)/%/rng_rdom.a $(foreach a,$(ABLATION),$(BIN)/%/$(a).a) $(BIN)/%/runtime.a
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -I$(BIN)/$* -DLANES=$(LANES) -DSTEPS=$(STEPS) -DPARALLEL=$(if $(filter true,$(PAR)),1,0) \
	    -DABLATE_FUSION=$(if $(filter 1,$(ABLATE_FUSION)),1,0) \
	    -Wall -O3 -march=native -ffast-math $^ -o $@ $(LDFLAGS) -lpthread -ldl

test: $(BIN)/$(HL_TARGET)/runner
	$<

# Byte-exact comparison against Julia's stdlib bulk fill. Julia always
# forks eight SIMD substreams, so build with LANES=8:
#   make clean && make LANES=8 test_julia
JULIA ?= $(HOME)/.juliaup/bin/julia
test_julia: $(BIN)/$(HL_TARGET)/runner
	$(JULIA) julia_ref.jl $(STEPS) $(BIN)/julia_seeds.bin $(BIN)/julia_ref.bin
	JULIA_SEEDS=$(BIN)/julia_seeds.bin JULIA_REF=$(BIN)/julia_ref.bin $<

clean:
	rm -rf $(BIN)
