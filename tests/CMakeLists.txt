find_package(benchmark)
if(WIN32)
  #temporary hard code
  include_directories("F:/Duplicate_work/Work/Code/Dependencies/install/msvc14/include")
  link_directories("F:/Duplicate_work/Work/Code/Dependencies/install/msvc14/lib")
endif()

set(TESTSOURCES
    AssignmentTest.cpp
    CompiledAssignmentTest.cpp
    ConstraintTest.cpp
    main.cpp
    Mockup.cpp
    OutputSelectorTest.cpp
    SolvingRequirementsTest.cpp)

set(TESTHEADERS
    Mockup.h
)

add_executable(main ${TESTSOURCES} ${TESTHEADERS})
target_link_libraries(main PUBLIC TVM)
GENERATE_MSVC_DOT_USER_FILE(main)

add_executable(TestDisabled TestDisabled.cpp)
target_link_libraries(TestDisabled PUBLIC TVM)
add_test(TestDisabled TestDisabled)
GENERATE_MSVC_DOT_USER_FILE(TestDisabled)

add_custom_target(tvm_benchmarks)

macro(addBenchmark name)
  add_executable(${name} ${name}.cpp ${ARGN})
  add_custom_command(TARGET tvm_benchmarks
    COMMAND ${name}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running ${name} benchmark"
    )
  target_link_libraries(${name} PUBLIC TVM benchmark)
  if(NOT WIN32)
    target_link_libraries(${name} PUBLIC pthread)
  else()
    target_link_libraries(${name} PUBLIC shlwapi)
  endif()
  # Adding a project configuration file (for MSVC only)
  GENERATE_MSVC_DOT_USER_FILE(${name})
endmacro()

addBenchmark(TestData)
