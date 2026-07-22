cmake_minimum_required(VERSION 3.28)
project(chebyshev_inductive)

enable_testing()

# Set up language settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

# Find Halide
find_package(Halide REQUIRED)

# Generator
add_halide_generator(
    chebyshev_inductive.generator
    SOURCES chebyshev_inductive_generator.cpp
    LINK_LIBRARIES Halide::Tools
)

# Filters: an inductive version and a non-inductive one.
add_halide_library(
    chebyshev_inductive
    FROM chebyshev_inductive.generator
    PARAMS inductive=true
)
add_halide_library(
    chebyshev_inductive_noninductive
    FROM chebyshev_inductive.generator
    GENERATOR chebyshev_inductive
    PARAMS inductive=false
)

# Main executable
add_executable(chebyshev_inductive_test test.cpp)
target_link_libraries(
    chebyshev_inductive_test
    PRIVATE
    Halide::Tools
    chebyshev_inductive
    chebyshev_inductive_noninductive
)

# Test that the app actually works!
add_test(NAME chebyshev_inductive_test COMMAND chebyshev_inductive_test)
set_tests_properties(
    chebyshev_inductive_test
    PROPERTIES
    LABELS chebyshev_inductive
    PASS_REGULAR_EXPRESSION "Success!"
    SKIP_REGULAR_EXPRESSION "\\[SKIP\\]"
)
