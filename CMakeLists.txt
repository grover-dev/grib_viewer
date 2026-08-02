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

# Project-wide compile settings, linked privately by every target we own.
add_library(boatforge_options INTERFACE)
add_library(boatforge::options ALIAS boatforge_options)
target_compile_options(boatforge_options INTERFACE
    -march=x86-64-v2
    -Wall -Wextra -Wpedantic
)

add_subdirectory(include)
add_subdirectory(src)

# Tests are opt-out so a consumer adding this by add_subdirectory doesn't pay
# for GoogleTest, but a developer building the repo directly always gets them.
option(BOATFORGE_BUILD_TESTS "Build the C++ unit tests" ${PROJECT_IS_TOP_LEVEL})
if(BOATFORGE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(unit_tests)
endif()
