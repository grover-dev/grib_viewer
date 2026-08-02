add_library(npy_tools
    npy.cpp
    npz.cpp
    npz_field.cpp
)
add_library(npy_tools::npy_tools ALIAS npy_tools)

target_link_libraries(npy_tools
    PUBLIC  npy_tools::headers
    PRIVATE libzip::zip
)

# The warning set is the parent project's, so it only applies when npy_tools is
# built as part of it; standalone, the consumer's own settings stand.
if(TARGET boatforge::options)
    target_link_libraries(npy_tools PRIVATE boatforge::options)
endif()
