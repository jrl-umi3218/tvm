##########################
# -- Google benchmark -- #
##########################

# Disable tests
set(BENCHMARK_ENABLE_TESTING OFF)
# Disable install
set(BENCHMARK_ENABLE_INSTALL OFF)
add_definitions(-Wno-error)
add_subdirectory(benchmark)
