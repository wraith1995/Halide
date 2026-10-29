# Steven Raphael's standardized inductive-Func benchmarks (from the
# paper-benchmarks-rebase branch), built as JIT binaries against the
# distrib. Each prints one hb::print_row line per variant; the suite
# driver (../inductive_benchmarks.py) parses them.
#
# The external baselines need headers this box may lack: Boost.odeint
# (header-only) for the ODE app and oneTBB for prefixsum. They are
# built when present (apt: libboost-dev libtbb-dev). SUNDIALS_DIR, an install
# of SUNDIALS built with SUNDIALS_PRECISION=SINGLE, adds its rows to the ODE
# app.
HL_TARGET ?= host

include ../support/Makefile.inc

# fast-math for the host-side references and baselines, matching the
# floating-point semantics Halide's own code generation assumes.
CXXFLAGS_JIT = -O3 -march=native -ffast-math -std=c++17 -I$(HALIDE_DISTRIB_PATH)/include -I$(HALIDE_DISTRIB_PATH)/tools
LDFLAGS_JIT = -L$(HALIDE_DISTRIB_PATH)/lib -lHalide -lpthread -ldl -Wl,-rpath,$(HALIDE_DISTRIB_PATH)/lib

BINS = $(BIN)/viterbi_log $(BIN)/chebyshev_test $(BIN)/prefixsum_bench $(BIN)/prefixsum_bench_rdom $(BIN)/prefixsum_bench_fold
# BOOST_DIR, a directory holding boost/, is searched instead of the system
# headers when set.
BOOST_INCLUDE = $(if $(BOOST_DIR),$(BOOST_DIR),/usr/include)
ifneq (,$(wildcard $(BOOST_INCLUDE)/boost/numeric/odeint.hpp))
BINS += $(BIN)/ode_observer_sparse_fused_test
endif
# TBB_DIR, an install prefix of oneTBB (e.g. a venv with pip's tbb-devel),
# is searched instead of the system headers when set.
ifneq (,$(TBB_DIR))
TBB_FLAGS = -I$(TBB_DIR)/include -L$(TBB_DIR)/lib -Wl,-rpath,$(TBB_DIR)/lib
TBB_INCLUDE = $(TBB_DIR)/include
else
TBB_INCLUDE = /usr/include
endif
ifneq (,$(wildcard $(TBB_INCLUDE)/oneapi/tbb.h $(TBB_INCLUDE)/tbb/tbb.h))
BINS += $(BIN)/prefixsum_bench_tbb
endif

all: $(BINS)

$(BIN)/prefixsum_bench_tbb: prefixsum_bench_tbb.cpp ../support/bench_harness.h
	@mkdir -p $(@D)
	$(CXX) -O3 -march=native -ffast-math -std=c++17 $< $(TBB_FLAGS) -ltbb -lpthread -o $@

ifneq (,$(SUNDIALS_DIR))
ODE_FLAGS = -DWITH_SUNDIALS -I$(SUNDIALS_DIR)/include -L$(SUNDIALS_DIR)/lib \
	-lsundials_arkode -lsundials_nvecserial -lsundials_core -Wl,-rpath,$(SUNDIALS_DIR)/lib
endif

$(BIN)/ode_observer_sparse_fused_test: ode_observer_sparse_fused_test.cpp ../support/bench_harness.h ../support/jit_support.h
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_JIT) $(if $(BOOST_DIR),-I$(BOOST_DIR)) $< $(LDFLAGS_JIT) $(ODE_FLAGS) -o $@

$(BIN)/%: %.cpp ../support/bench_harness.h ../support/jit_support.h
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS_JIT) $< $(LDFLAGS_JIT) -o $@

clean:
	rm -rf $(BIN)
