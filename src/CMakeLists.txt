# npy/npz reading, usable on its own and by future tools.
add_library(boatforge_core
    npy.cpp
    npz.cpp
    npz_field.cpp
)
add_library(boatforge::core ALIAS boatforge_core)

target_link_libraries(boatforge_core
    PUBLIC  boatforge::headers
    PRIVATE libzip::zip boatforge::options
)

# Demo driver: dump the contents of a .npz.
add_executable(boatforge
    main.cpp
)

target_link_libraries(boatforge
    PRIVATE boatforge::core boatforge::options
)
