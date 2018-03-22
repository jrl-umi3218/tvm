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
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${name}.cpp)
    add_executable(${name} ${name}.cpp ${ARGN})
  elseif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${name}.in.cpp)
    configure_file(${name}.in.cpp
      ${CMAKE_CURRENT_BINARY_DIR}/${name}.cpp)
    add_executable(${name} ${CMAKE_CURRENT_BINARY_DIR}/${name}.cpp ${ARGN})
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  else()
    message(SEND_ERROR "Cannot add a test ${name}, neither ${name}.cpp nor ${name}.in.cpp exists in ${CMAKE_CURRENT_SOURCE_DIR}")
  endif()
  target_link_libraries(${name} TVM)
  add_test(${name}Unit ${name})
  # Adding a project configuration file (for MSVC only)
  GENERATE_MSVC_DOT_USER_FILE(${name})
endmacro(addUnitTest)

addUnitTest(AssignmentTest)
addUnitTest(CompiledAssignmentTest)
addUnitTest(ConstraintTest)
addUnitTest(LinearizedTaskConstraintTest SolverTestFunctions.cpp)
addUnitTest(MatrixPropertiesTest)
addUnitTest(OutputSelectorTest)
addUnitTest(SolvingRequirementsTest)
addUnitTest(TaskDynamicsTest SolverTestFunctions.cpp)
addUnitTest(UtilsTest SolverTestFunctions.cpp)
addUnitTest(VariableTest)

add_required_dependency(hrp2_drc_description)
execute_process(COMMAND "${PKG_CONFIG_EXECUTABLE}"
  "--variable=datadir" "hrp2_drc_description"
  OUTPUT_VARIABLE "HRP2_DRC_DESCRIPTION_PATH"
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
message("Found HRP2_DRC_DESCRIPTION_PATH: ${HRP2_DRC_DESCRIPTION_PATH}")
add_required_dependency(mc_env_description)
execute_process(COMMAND "${PKG_CONFIG_EXECUTABLE}"
  "--variable=datadir" "mc_env_description"
  OUTPUT_VARIABLE "MC_ENV_DESCRIPTION_PATH"
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
message("Found MC_ENV_DESCRIPTION_PATH: ${MC_ENV_DESCRIPTION_PATH}")
add_optional_dependency(roscpp)
if(${ROSCPP_FOUND})
  add_required_dependency(tf2_ros)
  addUnitTest(RobotProblem RobotPublisher_ros.cpp)
  pkg_config_use_dependency(RobotProblem roscpp)
  pkg_config_use_dependency(RobotProblem tf2_ros)
else()
  addUnitTest(RobotProblem RobotPublisher_noros.cpp)
endif()
target_compile_options(RobotProblem PRIVATE -DHRP2_DRC_DESCRIPTION_PATH=\"${HRP2_DRC_DESCRIPTION_PATH}\" -DMC_ENV_DESCRIPTION_PATH=\"${MC_ENV_DESCRIPTION_PATH}\")

set(TESTSOURCES
    main.cpp
    Mockup.cpp
    SolverTest.cpp
    SolverTestFunctions.cpp
)

set(TESTHEADERS
    Mockup.h
    SolverTestFunctions.h
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

add_optional_dependency(Tasks Tasks_d)
if(TASKS_FOUND STREQUAL "")
  set(TASKS_FOUND 0)
endif()
if(TASKS_DEBUG_FOUND STREQUAL "")
  set(TASKS_DEBUG_FOUND 0)
endif()
if(${TASKS_FOUND} OR ${TASKS_DEBUG_FOUND})
  add_required_dependency(mc_rbdyn_urdf mc_rbdyn_urdf_d)
  addBenchmark(CompareTasks)
  target_link_libraries(CompareTasks boost_system)
  target_compile_options(CompareTasks PRIVATE -DHRP2_DRC_DESCRIPTION_PATH=\"${HRP2_DRC_DESCRIPTION_PATH}\" -DMC_ENV_DESCRIPTION_PATH=\"${MC_ENV_DESCRIPTION_PATH}\")
  pkg_config_use_dependency(CompareTasks Tasks)
endif()
