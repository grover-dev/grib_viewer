#include <npy_tools/npz.h>

#include <format>
#include <memory>
#include <string_view>
#include <utility>

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

namespace {

// Writing uses zip_discard rather than zip_close on the error path: zip_close
// is what commits the archive, so unwinding through it would leave a file that
// looks finished and is not.
struct ZipDiscarder {
    void operator()(zip_t* z) const { zip_discard(z); }
};
using ZipWriteHandle = std::unique_ptr<zip_t, ZipDiscarder>;

using Member = std::pair<std::string, std::vector<std::byte>>;

void write_archive(const std::filesystem::path& path,
                   const std::vector<Member>& members, Compression compression) {
    int code = 0;
    ZipWriteHandle archive{zip_open(path.c_str(), ZIP_CREATE | ZIP_TRUNCATE, &code)};
    if (!archive) {
        throw NpyError{std::format("cannot write {}: {}", path.string(),
                                   error_from_code(code))};
    }

    for (const auto& [name, image] : members) {
        const std::string entry = name + ".npy";

        // freep = 0: libzip reads straight out of `image`, which lives in the
        // writer until after zip_close.
        zip_source_t* source =
            zip_source_buffer(archive.get(), image.data(), image.size(), 0);
        if (source == nullptr) {
            throw NpyError{std::format("cannot buffer '{}': {}", entry,
                                       zip_strerror(archive.get()))};
        }

        const zip_int64_t index = zip_file_add(archive.get(), entry.c_str(),
                                               source, ZIP_FL_ENC_UTF_8);
        if (index < 0) {
            zip_source_free(source);  // ownership only transfers on success
            throw NpyError{std::format("cannot add '{}': {}", entry,
                                       zip_strerror(archive.get()))};
        }

        const zip_uint32_t method = compression == Compression::Deflate
                                        ? ZIP_CM_DEFLATE
                                        : ZIP_CM_STORE;
        if (zip_set_file_compression(archive.get(),
                                     static_cast<zip_uint64_t>(index), method,
                                     0) != 0) {
            throw NpyError{std::format("cannot set compression for '{}': {}",
                                       entry, zip_strerror(archive.get()))};
        }
    }

    zip_t* committing = archive.release();
    if (zip_close(committing) != 0) {
        const std::string message = zip_strerror(committing);
        zip_discard(committing);
        throw NpyError{
            std::format("cannot write {}: {}", path.string(), message)};
    }
}

}  // namespace

struct NpzWriter::Impl {
    std::filesystem::path path;
    Compression compression;
    std::vector<Member> members;  // insertion order, as savez writes them
    bool closed = false;
};

NpzWriter::NpzWriter(std::filesystem::path path, Compression compression)
    : impl_{std::make_unique<Impl>(std::move(path), compression)} {}

NpzWriter::~NpzWriter() {
    try {
        close();
    } catch (const NpyError&) {
        // Nothing useful to do here; close() explicitly to see the failure.
    }
}

NpzWriter::NpzWriter(NpzWriter&&) noexcept = default;
NpzWriter& NpzWriter::operator=(NpzWriter&&) noexcept = default;

void NpzWriter::add_image(std::string_view name, std::vector<std::byte> image) {
    if (name.empty()) {
        throw NpyError{"npz member name is empty"};
    }
    if (impl_->closed) {
        throw NpyError{std::format("cannot add '{}': {} is already written",
                                   name, impl_->path.string())};
    }
    for (const auto& [existing, _] : impl_->members) {
        if (existing == name) {
            throw NpyError{std::format("npz already has a member named '{}'",
                                       name)};
        }
    }
    impl_->members.emplace_back(std::string{name}, std::move(image));
}

void NpzWriter::close() {
    if (impl_ == nullptr || impl_->closed) {
        return;
    }
    impl_->closed = true;
    write_archive(impl_->path, impl_->members, impl_->compression);
}

void save_npz(const std::filesystem::path& path, const NpzArchive& arrays,
              Compression compression) {
    NpzWriter writer{path, compression};
    for (const auto& [name, array] : arrays) {
        writer.add(name, array);
    }
    writer.close();
}

}  // namespace boatforge
