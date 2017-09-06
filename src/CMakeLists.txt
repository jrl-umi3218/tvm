set(TVM_SOURCES
  data/Inputs.cpp
  Assignment.cpp
  AssignmentTarget.cpp
  CallGraph.cpp
  Constraint.cpp
#  ControlProblem.cpp
  EventGraph.cpp
  FirstOrderProvider.cpp
  Function.cpp
  LinearConstraint.cpp
  LinearizedTaskConstraint.cpp
  MatrixProperties.cpp
  ResolutionScheme.cpp
  Robot.cpp
  SolvingRequirements.cpp
  Task.cpp
  TaskDynamics.cpp
  Time.cpp
  Variable.cpp
  VariableVector.cpp)

set(TVM_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../include/tvm)
set(TVM_HEADERS
  ${TVM_INCLUDE_DIR}/api.h
  ${TVM_INCLUDE_DIR}/CallGraph.h
  ${TVM_INCLUDE_DIR}/data/AbstractNode.h
  ${TVM_INCLUDE_DIR}/data/Inputs.h
  ${TVM_INCLUDE_DIR}/data/Inputs.hpp
  ${TVM_INCLUDE_DIR}/data/Node.h
  ${TVM_INCLUDE_DIR}/data/Node.hpp
  ${TVM_INCLUDE_DIR}/data/Outputs.h
  ${TVM_INCLUDE_DIR}/utils/enums.h

# Temporary
  ${TVM_INCLUDE_DIR}/../Assignment.h
  ${TVM_INCLUDE_DIR}/../AssignmentTarget.h
  ${TVM_INCLUDE_DIR}/../Constraint.h
  ${TVM_INCLUDE_DIR}/../ConstraintEnums.h
#  ${TVM_INCLUDE_DIR}/../ControlProblem.h
  ${TVM_INCLUDE_DIR}/../defs.h
  ${TVM_INCLUDE_DIR}/../exceptions.h
  ${TVM_INCLUDE_DIR}/../EventGraph.h
  ${TVM_INCLUDE_DIR}/../FirstOrderProvider.h
  ${TVM_INCLUDE_DIR}/../Function.h
  ${TVM_INCLUDE_DIR}/../LinearConstraint.h
  ${TVM_INCLUDE_DIR}/../LinearizedTaskConstraint.h
  ${TVM_INCLUDE_DIR}/../MatrixProperties.h
  ${TVM_INCLUDE_DIR}/../ResolutionScheme.h
  ${TVM_INCLUDE_DIR}/../Robot.h
  ${TVM_INCLUDE_DIR}/../SolvingRequirements.h
  ${TVM_INCLUDE_DIR}/../Task.h
  ${TVM_INCLUDE_DIR}/../TaskDynamics.h
  ${TVM_INCLUDE_DIR}/../Time.h
  #${TVM_INCLUDE_DIR}/../UpdateChecker.h
  ${TVM_INCLUDE_DIR}/../Variable.h
  ${TVM_INCLUDE_DIR}/../VariableVector.h
  ${TVM_INCLUDE_DIR}/utils/CompiledAssignment.h
  ${TVM_INCLUDE_DIR}/utils/CompiledAssignmentWrapper.h)

add_library(TVM SHARED ${TVM_SOURCES} ${TVM_HEADERS})
set_target_properties(TVM PROPERTIES COMPILE_FLAGS "-DTVM_EXPORTS")

install(TARGETS TVM
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)

install(DIRECTORY ${TVM_INCLUDE_DIR} DESTINATION include
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")
