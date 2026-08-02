#pragma once

#include <filesystem>
#include <map>
#include <string>

#include <boatforge/npy.h>

namespace boatforge {

// Arrays keyed by name, with the ".npy" suffix stripped — matching what
// numpy.load() exposes as `.files`. std::less<> so lookups accept string_view.
using NpzArchive = std::map<std::string, NpyArray, std::less<>>;

// Load every array from a .npz archive. Handles ZIP64 (which numpy's savez
// emits unconditionally) and both stored and deflated members.
NpzArchive load_npz(const std::filesystem::path& path);

}  // namespace boatforge
