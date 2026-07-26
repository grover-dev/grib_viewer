#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace boatforge {

// Thrown for malformed or unsupported .npy / .npz input.
class NpyError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

enum class DType {
    Bool,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float16,
    Float32,
    Float64,
};

// Human-readable numpy-style name, e.g. "float64".
std::string_view to_string(DType dtype);

// Element width in bytes.
std::size_t word_size(DType dtype);

// The DType corresponding to a C++ arithmetic type, for checked access.
template <typename T>
constexpr DType dtype_of() {
    if constexpr (std::is_same_v<T, bool>) return DType::Bool;
    else if constexpr (std::is_same_v<T, std::int8_t>) return DType::Int8;
    else if constexpr (std::is_same_v<T, std::int16_t>) return DType::Int16;
    else if constexpr (std::is_same_v<T, std::int32_t>) return DType::Int32;
    else if constexpr (std::is_same_v<T, std::int64_t>) return DType::Int64;
    else if constexpr (std::is_same_v<T, std::uint8_t>) return DType::UInt8;
    else if constexpr (std::is_same_v<T, std::uint16_t>) return DType::UInt16;
    else if constexpr (std::is_same_v<T, std::uint32_t>) return DType::UInt32;
    else if constexpr (std::is_same_v<T, std::uint64_t>) return DType::UInt64;
    else if constexpr (std::is_same_v<T, float>) return DType::Float32;
    else if constexpr (std::is_same_v<T, double>) return DType::Float64;
    else static_assert(false, "no numpy dtype corresponds to this type");
}

class NpyArray {
public:
    NpyArray() = default;
    NpyArray(std::vector<std::size_t> shape, DType dtype, bool fortran_order,
             std::vector<std::byte> data);

    const std::vector<std::size_t>& shape() const { return shape_; }
    DType dtype() const { return dtype_; }
    bool fortran_order() const { return fortran_order_; }

    // Number of elements. A 0-d array (numpy scalar) has size 1.
    std::size_t size() const { return size_; }
    std::span<const std::byte> bytes() const { return data_; }

    // Typed view over the buffer. Throws NpyError if T does not match dtype.
    template <typename T>
    std::span<const T> as() const {
        if (dtype_of<T>() != dtype_) {
            throw NpyError{"array is " + std::string{to_string(dtype_)} +
                           ", not " + std::string{to_string(dtype_of<T>())}};
        }
        return {reinterpret_cast<const T*>(data_.data()), size_};
    }

    // Shape rendered numpy-style, e.g. "(383,)" or "()".
    std::string shape_string() const;

private:
    std::vector<std::size_t> shape_;
    DType dtype_ = DType::Float64;
    bool fortran_order_ = false;
    std::size_t size_ = 0;
    std::vector<std::byte> data_;
};

// Parse a complete .npy file image (header + payload).
NpyArray parse_npy(std::span<const std::byte> image);

}  // namespace boatforge
