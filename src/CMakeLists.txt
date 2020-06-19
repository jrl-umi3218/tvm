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

  event/Listener.cpp
  event/Source.cpp

  function/BasicLinearFunction.cpp
  function/Function.cpp
  function/IdentityFunction.cpp
  function/LinearFunction.cpp

  geometry/Plane.cpp

  graph/CallGraph.cpp
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

  requirements/SolvingRequirements.cpp

  scheme/Assignment.cpp
  scheme/AssignmentTarget.cpp
  scheme/helpers.cpp
  scheme/ResolutionScheme.cpp
  scheme/SchemeAbilities.cpp
  scheme/WeightedLeastSquares.cpp

  solver/defaultLeastSquareSolver.cpp
  solver/LeastSquareSolver.cpp

  task_dynamics/Constant.cpp
  task_dynamics/None.cpp
  task_dynamics/Proportional.cpp
  task_dynamics/ProportionalDerivative.cpp
  task_dynamics/Reference.cpp
  task_dynamics/TaskDynamics.cpp
  task_dynamics/TaskDynamicsImpl.cpp
  task_dynamics/VelocityDamper.cpp

  utils/UpdatelessFunction.cpp
  utils/checkFunction.cpp
)

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

  utils/sch.cpp
)

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

  ${TVM_INCLUDE_DIR}/internal/enums.h
  ${TVM_INCLUDE_DIR}/internal/FirstOrderProvider.h
  ${TVM_INCLUDE_DIR}/internal/IdProvider.h
  ${TVM_INCLUDE_DIR}/internal/MatrixProperties.h
  ${TVM_INCLUDE_DIR}/internal/MatrixWithProperties.h
  ${TVM_INCLUDE_DIR}/internal/meta.h
  ${TVM_INCLUDE_DIR}/internal/ObjWithId.h

  ${TVM_INCLUDE_DIR}/requirements/abstract/SingleSolvingRequirement.h
  ${TVM_INCLUDE_DIR}/requirements/AnisotropicWeight.h
  ${TVM_INCLUDE_DIR}/requirements/PriorityLevel.h
  ${TVM_INCLUDE_DIR}/requirements/SolvingRequirements.h
  ${TVM_INCLUDE_DIR}/requirements/ViolationEvaluation.h
  ${TVM_INCLUDE_DIR}/requirements/Weight.h

  ${TVM_INCLUDE_DIR}/scheme/abstract/ResolutionScheme.h
  ${TVM_INCLUDE_DIR}/scheme/internal/Assignment.h
  ${TVM_INCLUDE_DIR}/scheme/internal/AssignmentTarget.h
  ${TVM_INCLUDE_DIR}/scheme/internal/CompiledAssignment.h
  ${TVM_INCLUDE_DIR}/scheme/internal/CompiledAssignmentWrapper.h
  ${TVM_INCLUDE_DIR}/scheme/internal/helpers.h
  ${TVM_INCLUDE_DIR}/scheme/internal/helpers.hpp
  ${TVM_INCLUDE_DIR}/scheme/internal/ProblemComputationData.h
  ${TVM_INCLUDE_DIR}/scheme/internal/ResolutionSchemeBase.h
  ${TVM_INCLUDE_DIR}/scheme/internal/SchemeAbilities.h
  ${TVM_INCLUDE_DIR}/scheme/WeightedLeastSquares.h

  ${TVM_INCLUDE_DIR}/solver/abstract/LeastSquareSolver.h
  ${TVM_INCLUDE_DIR}/solver/internal/Option.h
  ${TVM_INCLUDE_DIR}/solver/defaultLeastSquareSolver.h

  ${TVM_INCLUDE_DIR}/task_dynamics/abstract/TaskDynamics.h
  ${TVM_INCLUDE_DIR}/task_dynamics/abstract/TaskDynamicsImpl.h
  ${TVM_INCLUDE_DIR}/task_dynamics/Clamped.h
  ${TVM_INCLUDE_DIR}/task_dynamics/Constant.h
  ${TVM_INCLUDE_DIR}/task_dynamics/enums.h
  ${TVM_INCLUDE_DIR}/task_dynamics/FeedForward.h
  ${TVM_INCLUDE_DIR}/task_dynamics/None.h
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
  ${TVM_INCLUDE_DIR}/utils/internal/ProtoTaskDetails.h
)

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

  ${TVM_INCLUDE_DIR}/utils/sch.h
)

if(TVM_WITH_ROBOT)
  list(APPEND TVM_SOURCES ${TVM_ROBOT_SOURCES})
  list(APPEND TVM_HEADERS ${TVM_ROBOT_HEADERS})
endif()

if(TVM_WITH_LSSOL)
  list(APPEND TVM_SOURCES solver/LSSOLLeastSquareSolver.cpp)
  list(APPEND TVM_HEADERS ${TVM_INCLUDE_DIR}/solver/LSSOLLeastSquareSolver.h)
endif()
if(TVM_WITH_QLD)
  list(APPEND TVM_SOURCES solver/QLDLeastSquareSolver.cpp)
  list(APPEND TVM_HEADERS ${TVM_INCLUDE_DIR}/solver/QLDLeastSquareSolver.h)
endif()
if(TVM_WITH_QUADPROG)
  list(APPEND TVM_SOURCES solver/QuadprogLeastSquareSolver.cpp)
  list(APPEND TVM_HEADERS ${TVM_INCLUDE_DIR}/solver/QuadprogLeastSquareSolver.h)
endif()

#Helper functions to assign files to groups in IDE, mimicking the organization of files on disk
#Adapted from https://stackoverflow.com/a/31423421
function(assign_source_group folder)
  foreach(_source IN ITEMS ${ARGN})
    if (IS_ABSOLUTE "${_source}")
        file(RELATIVE_PATH _source_rel "${CMAKE_SOURCE_DIR}" "${_source}")
    else()
        set(_source_rel "${_source}")
    endif()
    #remove include/tvm/ if present in the path
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
  target_link_libraries(TVM PUBLIC RBDyn::RBDyn RBDyn::Parsers sch-core::sch-core)
endif()
target_include_directories(TVM PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../include> $<INSTALL_INTERFACE:include>)
set_target_properties(TVM PROPERTIES COMPILE_FLAGS "-DTVM_EXPORTS")
set_target_properties(TVM PROPERTIES SOVERSION 0 VERSION 0.8.0)
target_compile_features(TVM PUBLIC cxx_std_17)
target_compile_definitions(TVM PUBLIC ${SOLVER_FLAGS})

if(GENERATE_COVERAGE)
  target_compile_options(TVM PRIVATE "--coverage")
  set_target_properties(TVM PROPERTIES LINK_FLAGS "--coverage")
endif()


install(TARGETS TVM
  EXPORT "${TARGETS_EXPORT_NAME}"
    LIBRARY DESTINATION "lib"
    ARCHIVE DESTINATION "lib"
    RUNTIME DESTINATION "bin"
)
if(TVM_WITH_ROBOT)
  install(DIRECTORY ${TVM_INCLUDE_DIR} DESTINATION ${INCLUDE_INSTALL_DIR})
else()
  install(DIRECTORY ${TVM_INCLUDE_DIR} DESTINATION ${INCLUDE_INSTALL_DIR}
    PATTERN "robot*" EXCLUDE
    PATTERN "utils/sch.h" EXCLUDE
    PATTERN Robot.h EXCLUDE)
endif()
