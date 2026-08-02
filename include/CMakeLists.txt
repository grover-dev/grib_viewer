# Public headers. A header-only INTERFACE target so consumers pick up the
# include path (and the header list, for IDEs and install rules) by linking it.

add_library(boatforge_headers INTERFACE)
add_library(boatforge::headers ALIAS boatforge_headers)

target_sources(boatforge_headers INTERFACE
    FILE_SET HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
    FILES
        boatforge/version.h
)

# npz-backed fields appear in boatforge's own headers, so the npy_tools include
# path has to travel with them.
target_link_libraries(boatforge_headers INTERFACE npy_tools::headers)
