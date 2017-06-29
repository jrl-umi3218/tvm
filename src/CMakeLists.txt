set(SOURCES
  Constraint.cpp
  ControlProblem.cpp
  DataGraph.cpp
  EventGraph.cpp
  FirstOrderProvider.cpp
  Function.cpp
  ResolutionScheme.cpp
  Robot.cpp
  Time.cpp
  Variable.cpp)
SET(VM_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
set(HEADERS 
  ${VM_INCLUDE_DIR}/CompiledAssignment.h
  ${VM_INCLUDE_DIR}/Constraint.h
  ${VM_INCLUDE_DIR}/ControlProblem.h
  ${VM_INCLUDE_DIR}/DataEnums.h
  ${VM_INCLUDE_DIR}/DataGraph.h
  ${VM_INCLUDE_DIR}/errors.h
  ${VM_INCLUDE_DIR}/EventGraph.h
  ${VM_INCLUDE_DIR}/FirstOrderProvider.h
  ${VM_INCLUDE_DIR}/Function.h
  ${VM_INCLUDE_DIR}/ResolutionScheme.h
  ${VM_INCLUDE_DIR}/Robot.h
  ${VM_INCLUDE_DIR}/Task.h
  ${VM_INCLUDE_DIR}/Time.h
  ${VM_INCLUDE_DIR}/tvm_api.h
  #${VM_INCLUDE_DIR}/UpdateChecker.h
  ${VM_INCLUDE_DIR}/Variable.h)

add_library(VariableManagement SHARED ${SOURCES} ${HEADERS})
