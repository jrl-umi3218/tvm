##########################
# -- Google benchmark -- #
##########################

# Disable tests
option(BENCHMARK_ENABLE_TESTING "Enable testing of the benchmark library." OFF)
add_definitions(-Wno-error)
add_subdirectory(benchmark)
