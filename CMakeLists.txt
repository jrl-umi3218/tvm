cmake_minimum_required(VERSION 3.1.3)

set(PROJECT_NAME TVM)
set(PROJECT_DESCRIPTION "...")
set(PROJECT_URL "https://github.com/jrl-umi3218/tvm")
set(PROJECT_DEBUG_POSTFIX "_d")
set(PROJECT_VERSION 0.8.0)
set(PROJECT_USE_CMAKE_EXPORT TRUE)

set(CMAKE_CXX_STANDARD 17)

include(cmake/base.cmake)
include(cmake/msvc-specific.cmake)

# Disable -Werror on Unix for now.
set(CXX_DISABLE_WERROR True)
set(DOXYGEN_USE_MATHJAX "YES")
set(DOXYGEN_DISTRIBUTE_GROUP_DOC YES)

project(TVM CXX)

if(NOT WIN32)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-sign-conversion -pedantic")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_ENABLE_EXTENDED_ALIGNED_STORAGE -D_SILENCE_CXX17_NEGATORS_DEPRECATION_WARNING")
endif()

set(USING_SOLVER FALSE)
set(SOLVER_FLAGS "")
set(SOLVER_LIBS "")

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
option(TVM_WITH_LSSOL "Build with LSSOL support" OFF)
if(TVM_WITH_LSSOL)
  add_project_dependency(eigen-lssol REQUIRED)
  list(APPEND SOLVER_FLAGS "-DTVM_USE_LSSOL")
  list(APPEND SOLVER_LIBS eigen-lssol::eigen-lssol)
  set(USING_SOLVER TRUE)
endif()

###################
# - eigen-qld - #
###################
option(TVM_WITH_QLD "Build with QLD support" ON)
if(TVM_WITH_QLD)
  add_project_dependency(eigen-qld REQUIRED)
  list(APPEND SOLVER_FLAGS "-DTVM_USE_QLD")
  list(APPEND SOLVER_LIBS eigen-qld::eigen-qld)
  set(USING_SOLVER TRUE)
endif()

###################
# - eigen-qld - #
###################
option(TVM_WITH_QUADPROG "Build with Quadprog support" OFF)
if(TVM_WITH_QUADPROG)
  add_project_dependency(eigen-quadprog REQUIRED)
  list(APPEND SOLVER_FLAGS "-DTVM_USE_QUADPROG")
  list(APPEND SOLVER_LIBS eigen-quadprog::eigen-quadprog)
  set(USING_SOLVER TRUE)
endif()


##########################
# - Robotic components - #
##########################
option(TVM_WITH_ROBOT "Build robots' related components" ON)
if(TVM_WITH_ROBOT)
  #############
  # - RBDyn - #
  #############
  add_project_dependency(RBDyn 1.2.1 REQUIRED)
  if(NOT TARGET RBDyn::Parsers)
    message(FATAL_ERROR "RBDyn must be built with parsers support")
  endif()

  ################
  # - sch-core - #
  ################
  add_project_dependency(sch-core REQUIRED)
endif()

#########################

if(NOT USING_SOLVER)
  message( FATAL_ERROR "You need to select at least one solver" )
endif()

# For MSVC, set local environment variable to enable finding the built dll
# of the main library when launching ctest with RUN_TESTS
if(MSVC)
  set(CMAKE_MSVCIDE_RUN_PATH "\$(SolutionDir)/src/\$(Configuration);${CMAKE_INSTALL_PREFIX}/bin")
endif(MSVC)

# Grouping projects in solution folders
set_property(GLOBAL PROPERTY USE_FOLDERS ON) 

add_subdirectory(3rd-party)
add_subdirectory(src)
if(${BUILD_TESTING})
  add_subdirectory(tests)
	add_subdirectory(examples)
endif()
