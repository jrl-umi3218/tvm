if(WIN32)
  #temporary hard code
  include_directories("F:/Duplicate_work/Work/Code/Dependencies/install/msvc14/include")
  link_directories("F:/Duplicate_work/Work/Code/Dependencies/install/msvc14/lib")
endif()

if(WIN32)
  #This is one of the way to avoid link errors related to static variables in program_options
  add_definitions(/bigobj)
endif(WIN32)

macro(addUnitTest name)
  add_executable(${name} ${name}.cpp ${ARGN})
  target_link_libraries(${name} TVM)
  add_test(${name}Unit ${name})
  # Adding a project configuration file (for MSVC only)
  GENERATE_MSVC_DOT_USER_FILE(${name})
endmacro(addUnitTest)

addUnitTest(AssignmentTest)
addUnitTest(CompiledAssignmentTest)
addUnitTest(ConstraintTest)
addUnitTest(MatrixPropertiesTest)
addUnitTest(OutputSelectorTest)
addUnitTest(SolvingRequirementsTest)
addUnitTest(UtilsTest)
addUnitTest(VariableTest)

set(TESTSOURCES
    main.cpp
    Mockup.cpp
    SolverTest.cpp
)

set(TESTHEADERS
    Mockup.h
)

add_executable(main ${TESTSOURCES} ${TESTHEADERS})
target_link_libraries(main TVM)
GENERATE_MSVC_DOT_USER_FILE(main)
pkg_config_use_dependency(main EigenQP)

add_executable(TestDisabled TestDisabled.cpp)
target_link_libraries(TestDisabled TVM)
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
  target_link_libraries(${name} TVM benchmark)
  if(NOT WIN32)
    target_link_libraries(${name} pthread)
  else()
    target_link_libraries(${name} shlwapi)
  endif()
  # Adding a project configuration file (for MSVC only)
  GENERATE_MSVC_DOT_USER_FILE(${name})
endmacro()

addBenchmark(TestData)
