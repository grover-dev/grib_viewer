#include <boatforge/npz.hpp>

#include <format>
#include <memory>
#include <string_view>

#include <zip.h>

namespace boatforge {
namespace {

struct ZipCloser {
    void operator()(zip_t* z) const { zip_close(z); }
};
struct ZipFileCloser {
    void operator()(zip_file_t* f) const { zip_fclose(f); }
};

using ZipHandle = std::unique_ptr<zip_t, ZipCloser>;
using ZipFileHandle = std::unique_ptr<zip_file_t, ZipFileCloser>;

std::string error_from_code(int code) {
    zip_error_t error;
    zip_error_init_with_code(&error, code);
    std::string message = zip_error_strerror(&error);
    zip_error_fini(&error);
    return message;
}

// Read one archive member in full.
std::vector<std::byte> read_member(zip_t* archive, zip_uint64_t index,
                                   const char* name, zip_uint64_t size) {
    ZipFileHandle file{zip_fopen_index(archive, index, 0)};
    if (!file) {
        throw NpyError{std::format("cannot open '{}': {}", name,
                                   zip_strerror(archive))};
    }

    std::vector<std::byte> buffer(size);
    zip_uint64_t offset = 0;
    while (offset < size) {
        const zip_int64_t n =
            zip_fread(file.get(), buffer.data() + offset, size - offset);
        if (n < 0) {
            throw NpyError{std::format("error reading '{}': {}", name,
                                       zip_file_strerror(file.get()))};
        }
        if (n == 0) {
            throw NpyError{std::format(
                "short read on '{}': expected {} bytes, got {}", name, size,
                offset)};
        }
        offset += static_cast<zip_uint64_t>(n);
    }
    return buffer;
}

}  // namespace

NpzArchive load_npz(const std::filesystem::path& path) {
    int code = 0;
    ZipHandle archive{zip_open(path.c_str(), ZIP_RDONLY, &code)};
    if (!archive) {
        throw NpyError{std::format("cannot open {}: {}", path.string(),
                                   error_from_code(code))};
    }

    const zip_int64_t entries = zip_get_num_entries(archive.get(), 0);
    if (entries < 0) {
        throw NpyError{std::format("cannot read {}: not a zip archive",
                                   path.string())};
    }

    NpzArchive arrays;
    for (zip_int64_t i = 0; i < entries; ++i) {
        zip_stat_t stat;
        if (zip_stat_index(archive.get(), static_cast<zip_uint64_t>(i), 0,
                           &stat) != 0) {
            throw NpyError{std::format("cannot stat entry {} of {}: {}", i,
                                       path.string(), zip_strerror(archive.get()))};
        }
        if ((stat.valid & ZIP_STAT_NAME) == 0 ||
            (stat.valid & ZIP_STAT_SIZE) == 0) {
            throw NpyError{std::format("entry {} of {} has no name or size", i,
                                       path.string())};
        }

        std::string_view name{stat.name};
        if (name.ends_with('/')) {
            continue;  // directory entry
        }
        if (!name.ends_with(".npy")) {
            continue;  // numpy writes only .npy members; ignore anything else
        }
        name.remove_suffix(4);

        const auto image = read_member(archive.get(),
                                       static_cast<zip_uint64_t>(i), stat.name,
                                       stat.size);
        try {
            arrays.emplace(name, parse_npy(image));
        } catch (const NpyError& e) {
            throw NpyError{std::format("{}: {}", name, e.what())};
        }
    }

    return arrays;
}

}  // namespace boatforge
