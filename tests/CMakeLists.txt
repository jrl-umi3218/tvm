find_package(benchmark)
if(WIN32)
  #temporary hard code
  include_directories("F:/Duplicate_work/Work/Code/Dependencies/install/msvc14/include")
  link_directories("F:/Duplicate_work/Work/Code/Dependencies/install/msvc14/lib")
endif()

set(TESTSOURCES
    CompiledAssignmentTest.cpp
    ConstraintTest.cpp
    main.cpp
    Mockup.cpp
    OutputSelectorTest.cpp
    SolvingRequirementsTest.cpp)

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
else()
  target_link_libraries(TestData PUBLIC shlwapi)
endif()
add_test(TestData TestData)
GENERATE_MSVC_DOT_USER_FILE( TestData)

add_executable(TestDisabled TestDisabled.cpp)
target_link_libraries(TestDisabled PUBLIC TVM)
add_test(TestDisabled TestDisabled)
GENERATE_MSVC_DOT_USER_FILE(TestDisabled)
