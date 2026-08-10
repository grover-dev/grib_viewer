cmake_minimum_required(VERSION 3.24)

project(boatforge
    VERSION 0.1.0
    DESCRIPTION "Weather routing / map generation"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# x86-64 target, GCC toolchain
if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    message(WARNING "boatforge targets x86-64; building on ${CMAKE_SYSTEM_PROCESSOR}")
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Everything (build tree, binaries, libs, fetched deps) lives under the build
# directory, which the presets place at <repo>/builds/<preset>/.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR "In-source builds are not allowed; use `cmake --preset release`.")
endif()

find_package(libzip REQUIRED)

# yaml-cpp, fetched rather than found, for the same reason as GoogleTest: the
# version is a property of the repo, not of the machine.
include(FetchContent)
FetchContent_Declare(yaml-cpp
    URL https://github.com/jbeder/yaml-cpp/archive/refs/tags/yaml-cpp-0.9.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
# Not ours, so it does not get boatforge::options' -Wall -Wpedantic.
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_INSTALL OFF CACHE BOOL "" FORCE)
set(YAML_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(yaml-cpp)

# Project-wide compile settings, linked privately by every target we own.
add_library(boatforge_options INTERFACE)
add_library(boatforge::options ALIAS boatforge_options)
target_compile_options(boatforge_options INTERFACE
    -march=x86-64-v2
    -Wall -Wextra -Wpedantic

    # Release optimisation. CMake's own Release flags are already -O3 -DNDEBUG
    # on GCC, but stating it here means the level is a property of the repo
    # rather than of whichever CMake default happens to be in play.
    #
    # -fno-math-errno lets GCC inline sqrt and friends as single instructions
    # instead of calls that have to set errno; nothing here reads errno after
    # a maths call. It is the one piece of -ffast-math that costs no accuracy:
    # the rest (reassociation, finite-math-only) is deliberately left off,
    # since great-circle and interpolation code wants IEEE semantics.
    $<$<CONFIG:Release>:-O3;-fno-math-errno;-fno-semantic-interposition>
)

# Link-time optimisation for Release, so inlining crosses the npy_tools /
# boatforge boundary. Checked rather than assumed: a toolchain without a
# working LTO plugin should warn, not fail to link.
include(CheckIPOSupported)
check_ipo_supported(RESULT boatforge_ipo_supported OUTPUT boatforge_ipo_error)
if(boatforge_ipo_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
else()
    message(WARNING "LTO unavailable, building Release without it: ${boatforge_ipo_error}")
endif()

add_subdirectory(npy_tools)
add_subdirectory(include)
add_subdirectory(src)

# Tests are opt-out so a consumer adding this by add_subdirectory doesn't pay
# for GoogleTest, but a developer building the repo directly always gets them.
option(BOATFORGE_BUILD_TESTS "Build the C++ unit tests" ${PROJECT_IS_TOP_LEVEL})
if(BOATFORGE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(unit_tests)
endif()
