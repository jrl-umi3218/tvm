if(WIN32)
  # This is one of the way to avoid link errors related to static variables in
  # program_options
  add_definitions(/bigobj)
endif(WIN32)

macro(addUnitTest name)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${name}.cpp)
    add_executable(${name} ${name}.cpp ${ARGN})
  elseif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${name}.in.cpp)
    configure_file(${name}.in.cpp ${CMAKE_CURRENT_BINARY_DIR}/${name}.cpp)
    add_executable(${name} ${CMAKE_CURRENT_BINARY_DIR}/${name}.cpp ${ARGN})
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  else()
    message(
      SEND_ERROR
        "Cannot add a test ${name}, neither ${name}.cpp nor ${name}.in.cpp exists in ${CMAKE_CURRENT_SOURCE_DIR}"
    )
  endif()
  target_link_libraries(${name} PUBLIC TVM)
  add_test(${name}Unit ${name})
  # Adding a project configuration file (for MSVC only)
  generate_msvc_dot_user_file(${name} "${CMAKE_INSTALL_PREFIX}/bin")
  set_target_properties(${name} PROPERTIES FOLDER "Tests")
  target_compile_definitions(${name} PUBLIC ${TEST_FLAGS})
endmacro(addUnitTest)

addunittest(AffineExprTest)
addunittest(AssignmentTest)
addunittest(CompiledAssignmentTest)
addunittest(ConstraintTest)
addunittest(DependencyGraph)
addunittest(DiagnosticTest SolverTestFunctions.cpp)
addunittest(EventsTest)
addunittest(FunctionTest)
addunittest(HierarchicalLeastSquareSolverTest SolverTestFunctions.cpp)
addunittest(LeastSquareSolverTest SolverTestFunctions.cpp)
addunittest(LinearizedTaskConstraintTest SolverTestFunctions.cpp)
addunittest(MatrixPropertiesTest)
addunittest(MetaTest)
addunittest(OutputSelectorTest)
addunittest(PairElementTokenTest)
addunittest(RangeCountingTest)
addunittest(RangeTest)
addunittest(SolverTest SolverTestFunctions.cpp)
addunittest(SolvingRequirementsTest)
addunittest(SubstitutionTest)
addunittest(TaskDynamicsTest SolverTestFunctions.cpp)
addunittest(UtilsTest SolverTestFunctions.cpp)
addunittest(VariableVectorTest)
addunittest(VariableTest)

if(TVM_WITH_ROBOT)
  find_package(jvrc_description REQUIRED)
  if("${jvrc_description_INSTALL_PREFIX}" STREQUAL "")
    if("${jvrc_description_SOURCE_PREFIX}" STREQUAL "")
      message(
        FATAL_ERROR
          "Your jvrc_description package does not define where to find the data"
      )
    endif()
    set(JVRC_DESCRIPTION_PATH "${jvrc_description_SOURCE_PREFIX}")
  else()
    set(JVRC_DESCRIPTION_PATH "${jvrc_description_INSTALL_PREFIX}")
  endif()
  find_package(mc_env_description REQUIRED)
  if("${mc_env_description_INSTALL_PREFIX}" STREQUAL "")
    if("${mc_env_description_SOURCE_PREFIX}" STREQUAL "")
      message(
        FATAL_ERROR
          "Your mc_env_description package does not define where to find the data"
      )
    endif()
    set(MC_ENV_DESCRIPTION_PATH "${mc_env_description_SOURCE_PREFIX}")
  else()
    set(MC_ENV_DESCRIPTION_PATH "${mc_env_description_INSTALL_PREFIX}")
  endif()
  find_package(roscpp QUIET)
  if(${roscpp_FOUND})
    add_required_dependency(roscpp)
    add_required_dependency(tf2_ros)
    addunittest(RobotProblem RobotPublisher_ros.cpp)
    pkg_config_use_dependency(RobotProblem roscpp)
    pkg_config_use_dependency(RobotProblem tf2_ros)
  else()
    addunittest(RobotProblem RobotPublisher_noros.cpp)
  endif()
  target_compile_options(
    RobotProblem
    PRIVATE -DJVRC_DESCRIPTION_PATH=\"${JVRC_DESCRIPTION_PATH}\"
            -DMC_ENV_DESCRIPTION_PATH=\"${MC_ENV_DESCRIPTION_PATH}\")
  addunittest(WrenchDistributionTest)
endif()

set(TESTSOURCES main.cpp Mockup.cpp SolverBasicTest.cpp SolverTestFunctions.cpp)

set(TESTHEADERS Mockup.h SolverTestFunctions.h)

add_executable(main ${TESTSOURCES} ${TESTHEADERS})
target_link_libraries(main TVM)
generate_msvc_dot_user_file(main "${CMAKE_INSTALL_PREFIX}/bin")
set_target_properties(main PROPERTIES FOLDER "Tests")

add_executable(TestDisabled TestDisabled.cpp)
target_link_libraries(TestDisabled TVM)
add_test(TestDisabled TestDisabled)
generate_msvc_dot_user_file(TestDisabled "${CMAKE_INSTALL_PREFIX}/bin")
set_target_properties(TestDisabled PROPERTIES FOLDER "Tests")

add_custom_target(tvm_benchmarks)
set_target_properties(tvm_benchmarks PROPERTIES FOLDER "Benchmarks")

macro(addBenchmark name)
  add_executable(${name} ${name}.cpp ${ARGN})
  add_custom_command(
    TARGET tvm_benchmarks
    COMMAND ${name}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running ${name} benchmark")
  target_link_libraries(${name} PUBLIC TVM benchmark)
  if(NOT WIN32)
    target_link_libraries(${name} PUBLIC pthread)
  else()
    target_link_libraries(${name} PUBLIC shlwapi)
  endif()
  # Adding a project configuration file (for MSVC only)
  generate_msvc_dot_user_file(${name} "${CMAKE_INSTALL_PREFIX}/bin")
  set_target_properties(${name} PROPERTIES FOLDER "Benchmarks")
endmacro()

addbenchmark(TestData)

if(TVM_WITH_ROBOT)
  find_package(Tasks QUIET)
  if(${Tasks_FOUND})
    find_package(Boost REQUIRED COMPONENTS system timer)
    addbenchmark(CompareTasks)
    if(NOT WIN32)
      target_link_libraries(CompareTasks PUBLIC Boost::system)
    endif()
    target_compile_options(
      CompareTasks
      PRIVATE -DJVRC_DESCRIPTION_PATH=\"${JVRC_DESCRIPTION_PATH}\"
              -DMC_ENV_DESCRIPTION_PATH=\"${MC_ENV_DESCRIPTION_PATH}\")
    target_link_libraries(CompareTasks PUBLIC Tasks::Tasks)
  endif()
endif()
