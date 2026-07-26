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

# All build artifacts land under <repo>/builds/
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/builds/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/builds/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/builds/lib)

include(cmake/cnpy.cmake)

add_executable(boatforge
    src/main.cpp
)

target_include_directories(boatforge PRIVATE include)

target_compile_options(boatforge PRIVATE
    -march=x86-64-v2
    -Wall -Wextra -Wpedantic
)

target_link_libraries(boatforge PRIVATE cnpy::cnpy)
