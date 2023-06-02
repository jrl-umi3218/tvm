set(TVM_SOURCES
    Clock.cpp
    ControlProblem.cpp
    LinearizedControlProblem.cpp
    Space.cpp
    Task.cpp
    Variable.cpp
    VariableVector.cpp
    constraint/BasicLinearConstraint.cpp
    constraint/Constraint.cpp
    constraint/LinearConstraint.cpp
    constraint/LinearizedTaskConstraint.cpp
    constraint/RHSVectors.cpp
    diagnostic/details.cpp
    diagnostic/GraphProbe.cpp
    event/Listener.cpp
    event/Source.cpp
    function/BasicLinearFunction.cpp
    function/Function.cpp
    function/IdentityFunction.cpp
    function/LinearFunction.cpp
    geometry/Plane.cpp
    graph/CallGraph.cpp
    graph/DependencyGraph.cpp
    graph/Inputs.cpp
    graph/Log.cpp
    graph/Logger.cpp
    hint/AutoCalculator.cpp
    hint/DiagonalCalculator.cpp
    hint/GenericCalculator.cpp
    hint/Substitution.cpp
    hint/SubstitutionCalculator.cpp
    hint/SubstitutionCalculatorImpl.cpp
    hint/Substitutions.cpp
    hint/SubstitutionUnit.cpp
    internal/FirstOrderProvider.cpp
    internal/MatrixProperties.cpp
    internal/ObjWithId.cpp
    internal/RangeCounting.cpp
    internal/VariableCountingVector.cpp
    scheme/Assignment.cpp
    scheme/AssignmentTarget.cpp
    scheme/helpers.cpp
    scheme/HierarchicalLeastSquares.cpp
    scheme/MatrixAssignment.cpp
    scheme/ResolutionScheme.cpp
    scheme/SchemeAbilities.cpp
    scheme/WeightedLeastSquares.cpp
    solver/defaultLeastSquareSolver.cpp
    solver/HierarchicalLeastSquareSolver.cpp
    solver/LeastSquareSolver.cpp
    task_dynamics/Constant.cpp
    task_dynamics/None.cpp
    task_dynamics/OneStepToZero.cpp
    task_dynamics/Proportional.cpp
    task_dynamics/ProportionalDerivative.cpp
    task_dynamics/Reference.cpp
    task_dynamics/TaskDynamics.cpp
    task_dynamics/TaskDynamicsImpl.cpp
    task_dynamics/VelocityDamper.cpp
    utils/UpdatelessFunction.cpp
    utils/checkFunction.cpp
    utils/memoryChecks.cpp)

set(TVM_ROBOT_SOURCES
    Robot.cpp
    robot/CollisionFunction.cpp
    robot/CoMFunction.cpp
    robot/CoMInConvexFunction.cpp
    robot/ConvexHull.cpp
    robot/Contact.cpp
    robot/DynamicFunction.cpp
    robot/Frame.cpp
    robot/FrictionCone.cpp
    robot/GeometricContactFunction.cpp
    robot/JointsSelector.cpp
    robot/OrientationFunction.cpp
    robot/PositionFunction.cpp
    robot/PostureFunction.cpp
    robot/utils.cpp
    utils/sch.cpp)

set(TVM_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include/tvm)
set(TVM_HEADERS
    ${TVM_INCLUDE_DIR}/api.h
    ${TVM_INCLUDE_DIR}/Clock.h
    ${TVM_INCLUDE_DIR}/ControlProblem.h
    ${TVM_INCLUDE_DIR}/defs.h
    ${TVM_INCLUDE_DIR}/LinearizedControlProblem.h
    ${TVM_INCLUDE_DIR}/Range.h
    ${TVM_INCLUDE_DIR}/Space.h
    ${TVM_INCLUDE_DIR}/Task.h
    ${TVM_INCLUDE_DIR}/Variable.h
    ${TVM_INCLUDE_DIR}/VariableVector.h
    ${TVM_INCLUDE_DIR}/constraint/abstract/Constraint.h
    ${TVM_INCLUDE_DIR}/constraint/abstract/LinearConstraint.h
    ${TVM_INCLUDE_DIR}/constraint/BasicLinearConstraint.h
    ${TVM_INCLUDE_DIR}/constraint/enums.h
    ${TVM_INCLUDE_DIR}/constraint/internal/LinearizedTaskConstraint.h
    ${TVM_INCLUDE_DIR}/constraint/internal/RHSVectors.h
    ${TVM_INCLUDE_DIR}/diagnostic/matrix.h
    ${TVM_INCLUDE_DIR}/diagnostic/GraphProbe.h
    ${TVM_INCLUDE_DIR}/diagnostic/internal/probe.h
    ${TVM_INCLUDE_DIR}/diagnostic/internal/traits.h
    ${TVM_INCLUDE_DIR}/event/enums.h
    ${TVM_INCLUDE_DIR}/event/Listener.h
    ${TVM_INCLUDE_DIR}/event/Source.h
    ${TVM_INCLUDE_DIR}/exception/exceptions.h
    ${TVM_INCLUDE_DIR}/function/abstract/Function.h
    ${TVM_INCLUDE_DIR}/function/abstract/LinearFunction.h
    ${TVM_INCLUDE_DIR}/function/BasicLinearFunction.h
    ${TVM_INCLUDE_DIR}/function/IdentityFunction.h
    ${TVM_INCLUDE_DIR}/geometry/Plane.h
    ${TVM_INCLUDE_DIR}/graph/abstract/Node.h
    ${TVM_INCLUDE_DIR}/graph/abstract/Node.hpp
    ${TVM_INCLUDE_DIR}/graph/abstract/OutputSelector.h
    ${TVM_INCLUDE_DIR}/graph/abstract/Outputs.h
    ${TVM_INCLUDE_DIR}/graph/CallGraph.h
    ${TVM_INCLUDE_DIR}/graph/internal/AbstractNode.h
    ${TVM_INCLUDE_DIR}/graph/internal/DependencyGraph.h
    ${TVM_INCLUDE_DIR}/graph/internal/Inputs.h
    ${TVM_INCLUDE_DIR}/graph/internal/Inputs.hpp
    ${TVM_INCLUDE_DIR}/graph/internal/Log.h
    ${TVM_INCLUDE_DIR}/graph/internal/Logger.h
    ${TVM_INCLUDE_DIR}/hint/Substitution.h
    ${TVM_INCLUDE_DIR}/hint/abstract/SubstitutionCalculator.h
    ${TVM_INCLUDE_DIR}/hint/abstract/SubstitutionCalculatorImpl.h
    ${TVM_INCLUDE_DIR}/hint/internal/AutoCalculator.h
    ${TVM_INCLUDE_DIR}/hint/internal/DiagonalCalculator.h
    ${TVM_INCLUDE_DIR}/hint/internal/GenericCalculator.h
    ${TVM_INCLUDE_DIR}/hint/internal/Substitutions.h
    ${TVM_INCLUDE_DIR}/hint/internal/SubstitutionUnit.h
    ${TVM_INCLUDE_DIR}/internal/CallbackManager.h
    ${TVM_INCLUDE_DIR}/internal/enums.h
    ${TVM_INCLUDE_DIR}/internal/FirstOrderProvider.h
    ${TVM_INCLUDE_DIR}/internal/IdProvider.h
    ${TVM_INCLUDE_DIR}/internal/MatrixProperties.h
    ${TVM_INCLUDE_DIR}/internal/MatrixWithProperties.h
    ${TVM_INCLUDE_DIR}/internal/meta.h
    ${TVM_INCLUDE_DIR}/internal/ObjWithId.h
    ${TVM_INCLUDE_DIR}/internal/PairElementToken.h
    ${TVM_INCLUDE_DIR}/internal/RangeCounting.h
    ${TVM_INCLUDE_DIR}/internal/traits.h
    ${TVM_INCLUDE_DIR}/internal/VariableCountingVector.h
    ${TVM_INCLUDE_DIR}/internal/VariableVectorPartition.h
    ${TVM_INCLUDE_DIR}/requirements/abstract/SingleSolvingRequirement.h
    ${TVM_INCLUDE_DIR}/requirements/AnisotropicWeight.h
    ${TVM_INCLUDE_DIR}/requirements/PriorityLevel.h
    ${TVM_INCLUDE_DIR}/requirements/SolvingRequirements.h
    ${TVM_INCLUDE_DIR}/requirements/ViolationEvaluation.h
    ${TVM_INCLUDE_DIR}/requirements/Weight.h
    ${TVM_INCLUDE_DIR}/scheme/abstract/ResolutionScheme.h
    ${TVM_INCLUDE_DIR}/scheme/internal/Assignment.h
    ${TVM_INCLUDE_DIR}/scheme/internal/Assignment.hpp
    ${TVM_INCLUDE_DIR}/scheme/internal/AssignmentTarget.h
    ${TVM_INCLUDE_DIR}/scheme/internal/CompiledAssignment.h
    ${TVM_INCLUDE_DIR}/scheme/internal/CompiledAssignmentWrapper.h
    ${TVM_INCLUDE_DIR}/scheme/internal/helpers.h
    ${TVM_INCLUDE_DIR}/scheme/internal/helpers.hpp
    ${TVM_INCLUDE_DIR}/scheme/internal/LinearizedProblemComputationData.h
    ${TVM_INCLUDE_DIR}/scheme/internal/MatrixAssignment.h
    ${TVM_INCLUDE_DIR}/scheme/internal/ProblemComputationData.h
    ${TVM_INCLUDE_DIR}/scheme/internal/ProblemDefinitionEvent.h
    ${TVM_INCLUDE_DIR}/scheme/internal/ResolutionSchemeBase.h
    ${TVM_INCLUDE_DIR}/scheme/internal/SchemeAbilities.h
    ${TVM_INCLUDE_DIR}/scheme/HierarchicalLeastSquares.h
    ${TVM_INCLUDE_DIR}/scheme/WeightedLeastSquares.h
    ${TVM_INCLUDE_DIR}/solver/abstract/HierarchicalLeastSquareSolver.h
    ${TVM_INCLUDE_DIR}/solver/abstract/LeastSquareSolver.h
    ${TVM_INCLUDE_DIR}/solver/internal/Option.h
    ${TVM_INCLUDE_DIR}/solver/internal/SolverEvents.h
    ${TVM_INCLUDE_DIR}/solver/defaultLeastSquareSolver.h
    ${TVM_INCLUDE_DIR}/task_dynamics/abstract/TaskDynamics.h
    ${TVM_INCLUDE_DIR}/task_dynamics/abstract/TaskDynamicsImpl.h
    ${TVM_INCLUDE_DIR}/task_dynamics/Clamped.h
    ${TVM_INCLUDE_DIR}/task_dynamics/Constant.h
    ${TVM_INCLUDE_DIR}/task_dynamics/enums.h
    ${TVM_INCLUDE_DIR}/task_dynamics/FeedForward.h
    ${TVM_INCLUDE_DIR}/task_dynamics/None.h
    ${TVM_INCLUDE_DIR}/task_dynamics/OneStepToZero.h
    ${TVM_INCLUDE_DIR}/task_dynamics/ProportionalDerivative.h
    ${TVM_INCLUDE_DIR}/task_dynamics/Proportional.h
    ${TVM_INCLUDE_DIR}/task_dynamics/Reference.h
    ${TVM_INCLUDE_DIR}/task_dynamics/VelocityDamper.h
    ${TVM_INCLUDE_DIR}/utils/AffineExpr.h
    ${TVM_INCLUDE_DIR}/utils/checkFunction.h
    ${TVM_INCLUDE_DIR}/utils/graph.h
    ${TVM_INCLUDE_DIR}/utils/ProtoTask.h
    ${TVM_INCLUDE_DIR}/utils/UpdatelessFunction.h
    ${TVM_INCLUDE_DIR}/utils/internal/AffineExprDetail.h
    ${TVM_INCLUDE_DIR}/utils/internal/AffineExprOperators.h
    ${TVM_INCLUDE_DIR}/utils/internal/BufferedMatrix.h
    ${TVM_INCLUDE_DIR}/utils/internal/graphDetails.h
    ${TVM_INCLUDE_DIR}/utils/internal/map.h
    ${TVM_INCLUDE_DIR}/utils/internal/MapWithVariableAsKey.h
    ${TVM_INCLUDE_DIR}/utils/internal/ProtoTaskDetails.h
    ${TVM_INCLUDE_DIR}/utils/memoryChecks.h)

set(TVM_ROBOT_HEADERS
    ${TVM_INCLUDE_DIR}/Robot.h
    ${TVM_INCLUDE_DIR}/robot/internal/DynamicFunction.h
    ${TVM_INCLUDE_DIR}/robot/internal/FrictionCone.h
    ${TVM_INCLUDE_DIR}/robot/internal/GeometricContactFunction.h
    ${TVM_INCLUDE_DIR}/robot/CollisionFunction.h
    ${TVM_INCLUDE_DIR}/robot/CoMFunction.h
    ${TVM_INCLUDE_DIR}/robot/CoMInConvexFunction.h
    ${TVM_INCLUDE_DIR}/robot/ConvexHull.h
    ${TVM_INCLUDE_DIR}/robot/Contact.h
    ${TVM_INCLUDE_DIR}/robot/Frame.h
    ${TVM_INCLUDE_DIR}/robot/JointsSelector.h
    ${TVM_INCLUDE_DIR}/robot/OrientationFunction.h
    ${TVM_INCLUDE_DIR}/robot/PositionFunction.h
    ${TVM_INCLUDE_DIR}/robot/PostureFunction.h
    ${TVM_INCLUDE_DIR}/robot/utils.h
    ${TVM_INCLUDE_DIR}/utils/sch.h)

set(TVM_HEADERS_INSTALL_EXCLUDE_PATTERN)

if(TVM_WITH_ROBOT)
  list(APPEND TVM_SOURCES ${TVM_ROBOT_SOURCES})
  list(APPEND TVM_HEADERS ${TVM_ROBOT_HEADERS})
endif()

macro(tvm_handle_solver_option OPTION)
  if(${OPTION})
    foreach(SRC ${ARGN})
      list(APPEND TVM_SOURCES ${SRC}.cpp)
      list(APPEND TVM_HEADERS "${TVM_INCLUDE_DIR}/${SRC}.h")
    endforeach()
  else()
    foreach(SRC ${ARGN})
      list(APPEND TVM_HEADERS_INSTALL_EXCLUDE_PATTERN PATTERN "${SRC}.h"
           EXCLUDE)
    endforeach()
  endif()
endmacro()

tvm_handle_solver_option(TVM_WITH_LSSOL solver/LSSOLLeastSquareSolver)
tvm_handle_solver_option(TVM_WITH_QLD solver/QLDLeastSquareSolver)
tvm_handle_solver_option(TVM_WITH_QUADPROG solver/QuadprogLeastSquareSolver)
tvm_handle_solver_option(
  TVM_WITH_LEXLS solver/LexLSHierarchicalLeastSquareSolver
  solver/LexLSLeastSquareSolver)
if(TVM_WITH_LEXLS)
  list(APPEND TVM_HEADERS "${TVM_INCLUDE_DIR}/solver/LexLSSolverOptions.h")
else()
  list(APPEND TVM_HEADERS_INSTALL_EXCLUDE_PATTERN PATTERN
       "solver/LexLSSolverOptions.h" EXCLUDE)
endif()

set(TVM_INCLUDE_BINARY_DIR "${PROJECT_BINARY_DIR}/include/")
configure_file("${PROJECT_SOURCE_DIR}/include/tvm/supported_solvers.in.h"
               "${TVM_INCLUDE_BINARY_DIR}/tvm/supported_solvers.h")
list(APPEND TVM_HEADERS "${TVM_INCLUDE_BINARY_DIR}/tvm/supported_solvers.h")

# Helper functions to assign files to groups in IDE, mimicking the organization
# of files on disk Adapted from https://stackoverflow.com/a/31423421
function(assign_source_group folder)
  foreach(_source IN ITEMS ${ARGN})
    if(IS_ABSOLUTE "${_source}")
      file(RELATIVE_PATH _source_rel "${CMAKE_SOURCE_DIR}" "${_source}")
    else()
      set(_source_rel "${_source}")
    endif()
    # remove include/tvm/ if present in the path
    string(REPLACE "include/tvm/" "" _source_rel_modified "${_source_rel}")
    get_filename_component(_source_path "${_source_rel_modified}" PATH)
    string(REPLACE "/" "\\" _source_path_msvc "${_source_path}")
    source_group("${folder}\\${_source_path_msvc}" FILES "${_source}")
  endforeach()
endfunction(assign_source_group)

assign_source_group("sources" ${TVM_SOURCES})
assign_source_group("headers" ${TVM_HEADERS})

add_library(TVM SHARED ${TVM_SOURCES} ${TVM_HEADERS})
target_link_libraries(TVM PUBLIC ${SOLVER_LIBS} tvm_3rd-party_mpark-variant)
target_link_libraries(TVM PUBLIC ${SOLVER_LIBS})
if(TVM_WITH_ROBOT)
  target_link_libraries(TVM PUBLIC RBDyn::RBDyn RBDyn::Parsers
                                   sch-core::sch-core)
endif()
target_include_directories(
  TVM
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
         $<BUILD_INTERFACE:${TVM_INCLUDE_BINARY_DIR}>
         $<INSTALL_INTERFACE:include>)
set_target_properties(TVM PROPERTIES COMPILE_FLAGS
                                     "-DTVM_EXPORTS -DEIGEN_RUNTIME_NO_MALLOC")
set_target_properties(TVM PROPERTIES SOVERSION ${PROJECT_VERSION_MAJOR}
                                     VERSION ${PROJECT_VERSION})
target_compile_features(TVM PUBLIC cxx_std_17)

if(GENERATE_COVERAGE)
  target_compile_options(TVM PRIVATE "--coverage")
  set_target_properties(TVM PROPERTIES LINK_FLAGS "--coverage")
endif()

if(TVM_TREAT_WARNINGS_AS_ERRORS)
  if(WIN32)
    target_compile_options(TVM PUBLIC $<BUILD_INTERFACE:/WX>)
  else()
    target_compile_options(TVM PUBLIC $<BUILD_INTERFACE:-Werror>)
  endif()
endif()

install(
  TARGETS TVM
  EXPORT "${TARGETS_EXPORT_NAME}"
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
if(NOT TVM_WITH_ROBOT)
  list(
    APPEND
    TVM_HEADERS_INSTALL_EXCLUDE_PATTERN
    PATTERN
    "robot*"
    EXCLUDE
    PATTERN
    "utils/sch.h"
    EXCLUDE
    PATTERN
    Robot.h
    EXCLUDE)
endif()

install(
  DIRECTORY ${TVM_INCLUDE_DIR}
  DESTINATION ${INCLUDE_INSTALL_DIR}
  PATTERN "supported_solvers.in.h" EXCLUDE
          ${TVM_HEADERS_INSTALL_EXCLUDE_PATTERN})

install(FILES "${TVM_INCLUDE_BINARY_DIR}/tvm/supported_solvers.h"
        DESTINATION ${INCLUDE_INSTALL_DIR}/tvm)
