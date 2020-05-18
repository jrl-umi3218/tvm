set(HEADERS_INCLUDE_DIR $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>$<INSTALL_INTERFACE:include/tvm/3rd-party/mpark>)
set(MPARK_VARIANT_HEADERS
  ${HEADERS_INCLUDE_DIR}/variant.hpp
)
install(FILES ${MPARK_VARIANT_HEADERS} DESTINATION "${CMAKE_INSTALL_PREFIX}/include/tvm/3rd-party/mpark}")
add_library(tvm_3rd-party_mpark-variant INTERFACE)
target_sources(tvm_3rd-party_mpark-variant INTERFACE ${MPARK_VARIANT_HEADERS})
target_include_directories(tvm_3rd-party_mpark-variant INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/3rd-party> $<INSTALL_INTERFACE:include/tvm/3rd-party/mpark>)
install(TARGETS tvm_3rd-party_mpark-variant EXPORT "${TARGETS_EXPORT_NAME}")