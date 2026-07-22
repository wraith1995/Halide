include ../support/Makefile.inc

.PHONY: build clean test

build: $(BIN)/$(HL_TARGET)/test

$(GENERATOR_BIN)/chebyshev_inductive.generator: chebyshev_inductive_generator.cpp $(GENERATOR_DEPS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(filter %.cpp,$^) -o $@ $(LIBHALIDE_LDFLAGS)

$(BIN)/%/chebyshev_inductive.a: $(GENERATOR_BIN)/chebyshev_inductive.generator
	@mkdir -p $(@D)
	$^ -g chebyshev_inductive -f chebyshev_inductive -e $(GENERATOR_OUTPUTS) -o $(@D) target=$*-no_runtime inductive=true

$(BIN)/%/chebyshev_inductive_noninductive.a: $(GENERATOR_BIN)/chebyshev_inductive.generator
	@mkdir -p $(@D)
	$^ -g chebyshev_inductive -f chebyshev_inductive_noninductive -e $(GENERATOR_OUTPUTS) -o $(@D) target=$*-no_runtime inductive=false

$(BIN)/%/runtime.a: $(GENERATOR_BIN)/chebyshev_inductive.generator
	@mkdir -p $(@D)
	$^ -r runtime -o $(@D) target=$*

$(BIN)/%/test: test.cpp $(BIN)/%/chebyshev_inductive.a $(BIN)/%/chebyshev_inductive_noninductive.a $(BIN)/%/runtime.a
	@mkdir -p $(@D)
	$(CXX-$*) $(CXXFLAGS-$*) -Wall -O2 -I$(BIN)/$* $^ -o $@ $(LDFLAGS-$*)

clean:
	rm -rf $(BIN)

test: $(BIN)/$(HL_TARGET)/test
	$<
