set(TVM_SOURCES

# Temporary
  EventGraph.cpp
  Variable.cpp
)

set(TVM_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../include/tvm)
set(TVM_HEADERS
  ${TVM_INCLUDE_DIR}/api.h
  ${TVM_INCLUDE_DIR}/data/AbstractNode.h
  ${TVM_INCLUDE_DIR}/data/Inputs.h
  ${TVM_INCLUDE_DIR}/data/Inputs.hpp
  ${TVM_INCLUDE_DIR}/data/Node.h
  ${TVM_INCLUDE_DIR}/data/Node.hpp
  ${TVM_INCLUDE_DIR}/data/Outputs.h
  ${TVM_INCLUDE_DIR}/utils/enums.h

# Temporary
  ${TVM_INCLUDE_DIR}/../EventGraph.h
  ${TVM_INCLUDE_DIR}/../tvm_api.h
  ${TVM_INCLUDE_DIR}/../Variable.h
)

add_library(TVM SHARED ${TVM_SOURCES} ${TVM_HEADERS})

install(TARGETS TVM
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)

install(DIRECTORY ${TVM_INCLUDE_DIR} DESTINATION include
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")
