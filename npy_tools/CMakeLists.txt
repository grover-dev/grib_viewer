# npy/npz reading, split out so it stands on its own: nothing in here knows
# about boats, routing or weather, and it depends only on libzip.

add_subdirectory(include)
add_subdirectory(src)
