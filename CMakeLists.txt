cmake_minimum_required(VERSION 2.8.12)

INCLUDE(cmake/base.cmake)
INCLUDE(cmake/boost.cmake)
INCLUDE(cmake/eigen.cmake)
INCLUDE(cmake/msvc-specific.cmake)

SET(PROJECT_NAME TVM)
SET(PROJECT_DESCRIPTION "...")
SET(PROJECT_URL "https://gite.lirmm.fr/adrien_escande_at_gmail_com/VariableManagement")
SET(PROJECT_DEBUG_POSTFIX "_d")

# Disable -Werror on Unix for now.
SET(CXX_DISABLE_WERROR True)

SETUP_PROJECT()

option(DISABLE_TESTS "Disable unit tests" OFF)

if(NOT WIN32)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-sign-conversion -std=c++0x -pedantic")
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
SEARCH_FOR_EIGEN()

###################
# - eigen-lssol - #
###################
add_required_dependency(eigen-lssol eigen-lssol_d)

#############
# - RBDyn - #
#############
add_required_dependency(RBDyn RBDyn_d)

################
# - sch-core - #
################
add_required_dependency(sch-core sch-core_d)

#####################
# - mc_rbdyn_urdf - #
#####################
add_required_dependency(mc_rbdyn_urdf mc_rbdyn_urdf_d)

#########################

# For MSVC, set local environment variable to enable finding the built dll
# of the main library when launching ctest with RUN_TESTS
IF(MSVC)
  SET(CMAKE_MSVCIDE_RUN_PATH "\$(SolutionDir)/src/\$(Configuration)")
ENDIF(MSVC)

add_subdirectory(3rd-party)
add_subdirectory(src)
if(NOT ${DISABLE_TESTS})
  add_subdirectory(tests)
endif()

# Add dependency towards the library in the pkg-config file.
PKG_CONFIG_APPEND_LIBS(TVM)

SETUP_PROJECT_FINALIZE()
