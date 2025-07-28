#!/bin/bash

readonly CMAKE_BUILD_TYPE=$1
readonly this_dir=`cd $(dirname $0); pwd`
readonly tvm_dir=`cd $this_dir/../../../; pwd`
readonly project_dir=$HOME/test-tvm

mkdir -p $project_dir/examples
mkdir -p $project_dir/helpers
mkdir -p $project_dir/build

cp $tvm_dir/tests/SolverTestFunctions.* $project_dir/helpers
cp -r $tvm_dir/examples/*.cpp $project_dir/examples
cp -r $tvm_dir/tests/doctest $project_dir/helpers

cat > $project_dir/CMakeLists.txt << EOF
cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)

project(tvm_consumer LANGUAGES CXX)
enable_testing()

find_package(TVM REQUIRED)

file(GLOB tvm_helpers_src helpers/*.cpp)
add_library(tvm_helpers OBJECT \${tvm_helpers_src})
target_compile_definitions(tvm_helpers PUBLIC $<TARGET_PROPERTY:TVM::TVM,INTERFACE_COMPILE_DEFINITIONS>)
target_include_directories(tvm_helpers PUBLIC $<TARGET_PROPERTY:TVM::TVM,INTERFACE_INCLUDE_DIRECTORIES>)

file(GLOB tvm_examples examples/*.cpp)

foreach(example \${tvm_examples})
  get_filename_component(name \${example} NAME_WE)
  add_executable(\${name} \${example} \$<TARGET_OBJECTS:tvm_helpers>)
  target_link_libraries(\${name} PUBLIC TVM::TVM)
  target_include_directories(\${name} PUBLIC \${PROJECT_SOURCE_DIR}/helpers)
  add_test(\${name} \${name})
endforeach()
EOF

cd $project_dir/build
cmake ../ -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} || exit 1
cmake --build . --config ${CMAKE_BUILD_TYPE} || exit 1
ctest -V -C ${CMAKE_BUILD_TYPE} || exit 1
