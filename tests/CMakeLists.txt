find_package(benchmark)

set(TESTSOURCES
    CompiledAssignmentTest.cpp
    main.cpp
    Mockup.cpp)

set(TESTHEADERS
    Mockup.h
)

add_executable(main ${TESTSOURCES} ${TESTHEADERS})
target_link_libraries(main PUBLIC TVM)
GENERATE_MSVC_DOT_USER_FILE(main)

add_executable(TestData TestData.cpp)
target_link_libraries(TestData PUBLIC TVM benchmark)
if(NOT WIN32)
  target_link_libraries(TestData PUBLIC pthread)
endif()
add_test(TestData TestData)
GENERATE_MSVC_DOT_USER_FILE(TestData)

add_executable(TestDisabled TestDisabled.cpp)
target_link_libraries(TestDisabled PUBLIC TVM)
add_test(TestDisabled TestDisabled)
GENERATE_MSVC_DOT_USER_FILE(TestDisabled)
