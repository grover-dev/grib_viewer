# Public headers. A header-only INTERFACE target so consumers pick up the
# include path (and the header list, for IDEs and install rules) by linking it.

add_library(boatforge_headers INTERFACE)
add_library(boatforge::headers ALIAS boatforge_headers)

target_sources(boatforge_headers INTERFACE
    FILE_SET HEADERS
    BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
    FILES
        boatforge/npy.h
        boatforge/npz.h
        boatforge/solar.h
        boatforge/version.h
)
