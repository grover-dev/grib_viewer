#pragma once

#include <filesystem>
#include <map>
#include <memory>
#include <string>

#include <npy_tools/npy.h>

namespace boatforge {

// Arrays keyed by name, with the ".npy" suffix stripped — matching what
// numpy.load() exposes as `.files`. std::less<> so lookups accept string_view.
using NpzArchive = std::map<std::string, NpyArray, std::less<>>;

// Load every array from a .npz archive. Handles ZIP64 (which numpy's savez
// emits unconditionally) and both stored and deflated members.
NpzArchive load_npz(const std::filesystem::path& path);

// Store matches numpy.savez, Deflate matches numpy.savez_compressed. Either
// loads with a plain numpy.load; the trade is write time against file size,
// and for the float columns a sim dumps, deflate typically wins little.
enum class Compression {
    Store,
    Deflate,
};

// Writes a .npz: a zip of .npy members, which is all numpy.savez produces.
//
// Arrays are added one at a time and the archive is committed by close() (or by
// the destructor). Nothing reaches disk before then, so a run that throws
// half-way through leaves no half-written file to mistake for a complete one.
//
//     NpzWriter out{"track.npz"};
//     out.add("lat", latitudes);      // any contiguous range -> 1-d array
//     out.add("t0", ...);
//     out.close();
class NpzWriter {
public:
    explicit NpzWriter(std::filesystem::path path,
                       Compression compression = Compression::Store);

    // Commits if close() was not called, and swallows any failure -- a
    // destructor cannot report one. Call close() explicitly if the output
    // matters, which for anything you intend to read back it does.
    ~NpzWriter();

    NpzWriter(NpzWriter&&) noexcept;
    NpzWriter& operator=(NpzWriter&&) noexcept;

    // Add a pre-serialized .npy image under `name` (the ".npy" suffix is added
    // for you). Throws if the name is already taken.
    void add_image(std::string_view name, std::vector<std::byte> image);

    void add(std::string_view name, const NpyArray& array) {
        add_image(name, write_npy(array));
    }

    // A contiguous range becomes a 1-d array of its element type.
    template <std::ranges::contiguous_range R>
    void add(std::string_view name, const R& values) {
        add_image(name, write_npy(values));
    }

    template <std::ranges::contiguous_range R>
    void add(std::string_view name, const R& values,
             const std::vector<std::size_t>& shape, bool fortran_order = false) {
        add_image(name, write_npy(values, shape, fortran_order));
    }

    // A single value becomes a 0-d array, the way the gridded-field npz stores
    // its axis constants.
    template <typename T>
    void add_scalar(std::string_view name, const T& value) {
        add_image(name, write_npy_scalar(value));
    }

    // Write the archive out. Throws on any I/O failure; calling it twice is a
    // no-op, so a close() in a success path and one in the destructor agree.
    void close();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// One-shot equivalent for an archive you already hold, the inverse of
// load_npz(): load_npz(save_npz(x)) gives back x.
void save_npz(const std::filesystem::path& path, const NpzArchive& arrays,
              Compression compression = Compression::Store);

}  // namespace boatforge
