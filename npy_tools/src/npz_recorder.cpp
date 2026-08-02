#include <npy_tools/npz_recorder.h>

#include <algorithm>
#include <format>
#include <map>
#include <vector>

namespace boatforge {
namespace {

struct Column {
    DType dtype;
    std::size_t count = 0;
    std::vector<std::byte> data;
};

}  // namespace

struct NpzRecorder::Impl {
    // Sorted, like NpzArchive; `order` keeps the archive's member order the
    // order the fields were first recorded in, which is the order they appear
    // in the step function and so the order someone reading the file expects.
    std::map<std::string, Column, std::less<>> columns;
    std::map<std::string, Column, std::less<>> scalars;
    std::vector<std::string> order;

    Column& column(std::string_view name, DType dtype, const char* kind,
                   std::map<std::string, Column, std::less<>>& into) {
        const auto it = into.find(name);
        if (it != into.end()) {
            if (it->second.dtype != dtype) {
                throw NpyError{std::format(
                    "'{}' was recorded as {}, now {} -- a column cannot change "
                    "type mid-run",
                    name, to_string(it->second.dtype), to_string(dtype))};
            }
            return it->second;
        }

        auto& other = (&into == &columns) ? scalars : columns;
        if (other.contains(name)) {
            throw NpyError{std::format(
                "'{}' is already a {}; it cannot also be a {}", name,
                (&into == &columns) ? "scalar" : "column", kind)};
        }

        order.push_back(std::string{name});
        return into.emplace(std::string{name}, Column{dtype, 0, {}})
            .first->second;
    }
};

NpzRecorder::NpzRecorder() : impl_{std::make_unique<Impl>()} {}
NpzRecorder::~NpzRecorder() = default;
NpzRecorder::NpzRecorder(NpzRecorder&&) noexcept = default;
NpzRecorder& NpzRecorder::operator=(NpzRecorder&&) noexcept = default;

void NpzRecorder::append(std::string_view name, DType dtype,
                         std::span<const std::byte> value) {
    Column& column = impl_->column(name, dtype, "column", impl_->columns);
    column.data.insert(column.data.end(), value.begin(), value.end());
    ++column.count;
}

void NpzRecorder::set_scalar_bytes(std::string_view name, DType dtype,
                                   std::span<const std::byte> value) {
    Column& scalar = impl_->column(name, dtype, "scalar", impl_->scalars);
    scalar.data.assign(value.begin(), value.end());
    scalar.count = 1;
}

std::size_t NpzRecorder::columns() const { return impl_->columns.size(); }

std::size_t NpzRecorder::rows() const {
    std::size_t rows = 0;
    for (const auto& [name, column] : impl_->columns) {
        rows = std::max(rows, column.count);
    }
    return rows;
}

std::size_t NpzRecorder::rows(std::string_view name) const {
    const auto it = impl_->columns.find(name);
    return it == impl_->columns.end() ? 0 : it->second.count;
}

void NpzRecorder::save(const std::filesystem::path& path,
                       Compression compression) const {
    const std::size_t expected = rows();
    for (const auto& [name, column] : impl_->columns) {
        if (column.count != expected) {
            throw NpyError{std::format(
                "column '{}' has {} samples but others have {} -- every field "
                "must be recorded on every step",
                name, column.count, expected)};
        }
    }

    NpzWriter writer{path, compression};
    for (const std::string& name : impl_->order) {
        if (const auto it = impl_->columns.find(name); it != impl_->columns.end()) {
            writer.add_image(name, write_npy(it->second.data, it->second.dtype,
                                             {it->second.count}));
        } else if (const auto s = impl_->scalars.find(name);
                   s != impl_->scalars.end()) {
            writer.add_image(name, write_npy(s->second.data, s->second.dtype, {}));
        }
    }
    writer.close();
}

void NpzRecorder::clear() {
    impl_->columns.clear();
    impl_->scalars.clear();
    impl_->order.clear();
}

}  // namespace boatforge
