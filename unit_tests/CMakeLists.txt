# GoogleTest, fetched rather than found: the version is then a property of the
# repo instead of the machine, and it lands under builds/ with everything else.
include(FetchContent)

FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
# GoogleTest is not ours, so it does not get boatforge::options' -Wall -Wpedantic.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(boatforge_tests
    npz_field_test.cpp
)

target_link_libraries(boatforge_tests
    PRIVATE npy_tools::npy_tools boatforge::options GTest::gtest_main
)

# The fixtures are read from the source tree rather than copied next to the
# binary: regenerating them with make_fixtures.py then takes effect without a
# rebuild, and there is only ever one copy to keep straight.
target_compile_definitions(boatforge_tests
    PRIVATE BOATFORGE_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures"
)

include(GoogleTest)
gtest_discover_tests(boatforge_tests)
