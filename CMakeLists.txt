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

if(NOT WIN32)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-sign-conversion -std=c++0x -pedantic")
endif()

#########################
# External dependencies
SEARCH_FOR_EIGEN()

#########################

# For MSVC, set local environment variable to enable finding the built dll
# of the main library when launching ctest with RUN_TESTS
IF(MSVC)
  SET(CMAKE_MSVCIDE_RUN_PATH "\$(SolutionDir)/src/\$(Configuration)")
ENDIF(MSVC)

add_subdirectory(3rd-party)
add_subdirectory(src)
add_subdirectory(tests)

# Add dependency towards the library in the pkg-config file.
PKG_CONFIG_APPEND_LIBS(TVM)

SETUP_PROJECT_FINALIZE()
