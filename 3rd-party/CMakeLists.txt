##########################
# -- Google benchmark -- #
##########################

# Disable tests
option(BENCHMARK_ENABLE_TESTING "Enable testing of the benchmark library." OFF)
if(NOT WIN32)
  add_definitions(-Wno-error)
endif()
add_subdirectory(benchmark EXCLUDE_FROM_ALL)
