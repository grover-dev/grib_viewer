# Public headers. A header-only INTERFACE target so consumers pick up the
# include path (and the header list, for IDEs and install rules) by linking it.

add_library(npy_tools_headers INTERFACE)
add_library(npy_tools::headers ALIAS npy_tools_headers)

target_sources(npy_tools_headers INTERFACE
    FILE_SET HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
    FILES
        npy_tools/npy.h
        npy_tools/npz.h
        npy_tools/npz_field.h
        npy_tools/npz_recorder.h
)
