cmake_minimum_required(VERSION 3.1.3)

set(PROJECT_NAME TVM)
set(PROJECT_DESCRIPTION "...")
set(PROJECT_URL "https://github.com/jrl-umi3218/tvm")
set(PROJECT_DEBUG_POSTFIX "_d")
set(PROJECT_VERSION 0.8.0)
set(PROJECT_USE_CMAKE_EXPORT TRUE)

include(cmake/base.cmake)
include(cmake/msvc-specific.cmake)

# Disable -Werror on Unix for now.
set(CXX_DISABLE_WERROR True)
set(DOXYGEN_USE_MATHJAX "YES")

project(TVM CXX)

if(NOT WIN32)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-sign-conversion -std=c++1y -pedantic")
else()
  add_definitions("-D_ENABLE_EXTENDED_ALIGNED_STORAGE")
endif()


# If we were given coverage options, we will only pass them to the core
# library.
set(GENERATE_COVERAGE FALSE CACHE BOOL "Generate coverage data")
if("${CMAKE_CXX_FLAGS}" MATCHES "--coverage")
  set(GENERATE_COVERAGE TRUE CACHE BOOL "Generate coverage data" FORCE)
  string(REPLACE "--coverage" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  string(REPLACE "--coverage" "" CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
  string(REPLACE "--coverage" "" CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS}")
endif()

#############################
# - External dependencies - #
#############################

#############
# - Eigen - #
#############
add_project_dependency(Eigen3 REQUIRED)

###################
# - eigen-lssol - #
###################
add_project_dependency(eigen-lssol REQUIRED)

###################
# - eigen-qld - #
###################
add_project_dependency(eigen-qld REQUIRED)

###################
# - eigen-qld - #
###################
add_project_dependency(eigen-quadprog REQUIRED)

#############
# - RBDyn - #
#############
add_project_dependency(RBDyn REQUIRED)

################
# - sch-core - #
################
add_project_dependency(sch-core REQUIRED)

#####################
# - mc_rbdyn_urdf - #
#####################
add_project_dependency(mc_rbdyn_urdf REQUIRED)

#########################

# For MSVC, set local environment variable to enable finding the built dll
# of the main library when launching ctest with RUN_TESTS
if(MSVC)
  set(CMAKE_MSVCIDE_RUN_PATH "\$(SolutionDir)/src/\$(Configuration)")
endif(MSVC)

add_subdirectory(3rd-party)
add_subdirectory(src)
if(${BUILD_TESTING})
  add_subdirectory(tests)
endif()
