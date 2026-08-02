# GoogleTest, fetched rather than found: the version is then a property of the
# repo instead of the machine, and it lands under builds/ with everything else.
# Guarded because unit_tests/ declares the same dependency, and whichever
# directory CMake reaches first is the one that pays for it.
if(NOT TARGET GTest::gtest_main)
    include(FetchContent)
    FetchContent_Declare(googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    # GoogleTest is not ours, so it does not get -Wall -Wpedantic.
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

# The npz tests read real numpy savez output rather than anything generated
# here -- see npz_test.cpp for why. The fixtures that already exist for the
# field tests are exactly the right shape, so they are borrowed instead of
# duplicated; a standalone consumer can point this elsewhere.
set(NPY_TOOLS_FIXTURE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../unit_tests/fixtures"
    CACHE PATH "Directory holding the .npz fixtures the npy_tools tests read")

add_executable(npy_tools_tests
    npy_test.cpp
    npz_test.cpp
)

target_link_libraries(npy_tools_tests
    PRIVATE npy_tools::npy_tools GTest::gtest_main
)

if(TARGET boatforge::options)
    target_link_libraries(npy_tools_tests PRIVATE boatforge::options)
endif()

# Read from the source tree rather than copied next to the binary: regenerating
# them then takes effect without a rebuild, and there is only ever one copy to
# keep straight.
target_compile_definitions(npy_tools_tests
    PRIVATE NPY_TOOLS_FIXTURE_DIR="${NPY_TOOLS_FIXTURE_DIR}"
)

include(GoogleTest)
gtest_discover_tests(npy_tools_tests)
