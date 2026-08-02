# npy/npz reading, split out so it stands on its own: nothing in here knows
# about boats, routing or weather, and it depends only on libzip.

add_subdirectory(include)
add_subdirectory(src)

# Opt-out, matching BOATFORGE_BUILD_TESTS: a consumer pulling this in by
# add_subdirectory does not pay for GoogleTest, a developer building the repo
# directly always gets them.
option(NPY_TOOLS_BUILD_TESTS "Build the npy_tools unit tests" ${PROJECT_IS_TOP_LEVEL})
if(NPY_TOOLS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
