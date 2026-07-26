# cnpy — .npy/.npz reader/writer (https://github.com/rogersce/cnpy)
#
# Upstream ships a CMakeLists.txt with `cmake_minimum_required(VERSION 2.6)`,
# which CMake >= 4 refuses to process. We therefore fetch the sources only
# (SOURCE_SUBDIR points at a nonexistent dir so add_subdirectory is skipped)
# and define our own target here.

include(FetchContent)

find_package(ZLIB REQUIRED)

FetchContent_Declare(cnpy
    GIT_REPOSITORY https://github.com/rogersce/cnpy.git
    GIT_TAG        4e8810b1a8637695171ed346ce68f6984e585ef4
    GIT_SHALLOW    FALSE
    SOURCE_SUBDIR  cmake-build-disabled
)
FetchContent_MakeAvailable(cnpy)

add_library(cnpy STATIC ${cnpy_SOURCE_DIR}/cnpy.cpp)
add_library(cnpy::cnpy ALIAS cnpy)

target_include_directories(cnpy SYSTEM PUBLIC ${cnpy_SOURCE_DIR})
target_link_libraries(cnpy PUBLIC ZLIB::ZLIB)
set_target_properties(cnpy PROPERTIES
    CXX_STANDARD 23
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)
# Third-party: don't hold it to our warning level.
target_compile_options(cnpy PRIVATE -w)
