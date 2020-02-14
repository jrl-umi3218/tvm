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
  GENERATE_MSVC_DOT_USER_FILE(${name} "${CMAKE_INSTALL_PREFIX}/bin")
endmacro(addUnitTest)

addUnitTest(AssignmentTest)
addUnitTest(CompiledAssignmentTest)
addUnitTest(ConstraintTest)
addUnitTest(LeastSquareSolverTest SolverTestFunctions.cpp)
addUnitTest(LinearizedTaskConstraintTest SolverTestFunctions.cpp)
addUnitTest(MatrixPropertiesTest)
addUnitTest(OutputSelectorTest)
addUnitTest(SolverTest SolverTestFunctions.cpp)
addUnitTest(SolvingRequirementsTest)
addUnitTest(SubstitutionTest)
addUnitTest(TaskDynamicsTest SolverTestFunctions.cpp)
addUnitTest(UtilsTest SolverTestFunctions.cpp)
addUnitTest(VariableTest)
addUnitTest(WrenchDistributionTest)
#pkg_config_use_dependency(WrenchDistributionTest eigen-lssol)

find_package(hrp2_drc_description REQUIRED)
if("${hrp2_drc_description_INSTALL_PREFIX}" STREQUAL "")
  if("${hrp2_drc_description_SOURCE_PREFIX}" STREQUAL "")  
    message(FATAL_ERROR "Your hrp2_drc_description package does not define where to find the data")
  endif()                                     
  set(HRP2_DRC_DESCRIPTION_PATH "${hrp2_drc_description_SOURCE_PREFIX}")   
else()
  set(HRP2_DRC_DESCRIPTION_PATH "${hrp2_drc_description_INSTALL_PREFIX}")
endif()
find_package(mc_env_description REQUIRED)
if("${mc_env_description_INSTALL_PREFIX}" STREQUAL "")
  if("${mc_env_description_SOURCE_PREFIX}" STREQUAL "")  
    message(FATAL_ERROR "Your mc_env_description package does not define where to find the data")
  endif()                                     
  set(MC_ENV_DESCRIPTION_PATH "${mc_env_description_SOURCE_PREFIX}")   
else()
  set(MC_ENV_DESCRIPTION_PATH "${mc_env_description_INSTALL_PREFIX}")
endif()
find_package(roscpp QUIET)
if(${ROSCPP_FOUND})
  add_required_dependency(tf2_ros)
  addUnitTest(RobotProblem RobotPublisher_ros.cpp)
  pkg_config_use_dependency(RobotProblem roscpp)
  pkg_config_use_dependency(RobotProblem tf2_ros)
else()
  addUnitTest(RobotProblem RobotPublisher_noros.cpp)
endif()
#pkg_config_use_dependency(RobotProblem mc_rbdyn_urdf)
target_compile_options(RobotProblem PRIVATE -DHRP2_DRC_DESCRIPTION_PATH=\"${HRP2_DRC_DESCRIPTION_PATH}\" -DMC_ENV_DESCRIPTION_PATH=\"${MC_ENV_DESCRIPTION_PATH}\")

set(TESTSOURCES
    main.cpp
    Mockup.cpp
    SolverBasicTest.cpp
    SolverTestFunctions.cpp
)

set(TESTHEADERS
    Mockup.h
    SolverTestFunctions.h
)

add_executable(main ${TESTSOURCES} ${TESTHEADERS})
target_link_libraries(main TVM)
GENERATE_MSVC_DOT_USER_FILE(main "${CMAKE_INSTALL_PREFIX}/bin")
#pkg_config_use_dependency(main eigen-lssol)

add_executable(TestDisabled TestDisabled.cpp)
target_link_libraries(TestDisabled TVM)
add_test(TestDisabled TestDisabled)
GENERATE_MSVC_DOT_USER_FILE(TestDisabled "${CMAKE_INSTALL_PREFIX}/bin")

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
  GENERATE_MSVC_DOT_USER_FILE(${name} "${CMAKE_INSTALL_PREFIX}/bin")
endmacro()

addBenchmark(TestData)

find_package(Tasks QUIET)
if(${Tasks_FOUND})
  find_package(Boost REQUIRED COMPONENTS system timer)
  find_package(mc_rbdyn_urdf REQUIRED)
  addBenchmark(CompareTasks)
  if(NOT WIN32)
    target_link_libraries(CompareTasks PUBLIC Boost::system)
  endif()
  target_compile_options(CompareTasks PRIVATE -DHRP2_DRC_DESCRIPTION_PATH=\"${HRP2_DRC_DESCRIPTION_PATH}\" -DMC_ENV_DESCRIPTION_PATH=\"${MC_ENV_DESCRIPTION_PATH}\")
  target_link_libraries(CompareTasks PUBLIC Tasks::Tasks mc_rbdyn_urdf::mc_rbdyn_urdf)
endif()
