set(TESTSOURCES
    main.cpp
    Mockup.cpp)

set(TESTHEADERS
    Mockup.h
)

add_executable(Main ${TESTSOURCES} ${TESTHEADERS})
target_link_libraries(Main PUBLIC VariableManagement)
GENERATE_MSVC_DOT_USER_FILE(Main)
