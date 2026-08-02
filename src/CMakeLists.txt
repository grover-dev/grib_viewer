# Demo driver: dump the contents of a .npz. boatforge has no compiled sources
# of its own any more -- npy/npz reading moved out to npy_tools/ -- so there is
# no boatforge::core library left to link.
add_executable(boatforge
    main.cpp
)

target_link_libraries(boatforge
    PRIVATE boatforge::headers npy_tools::npy_tools boatforge::options
)
