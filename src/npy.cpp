#include <boatforge/npy.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cstring>
#include <format>

namespace boatforge {
namespace {

constexpr std::array<char, 6> kMagic{'\x93', 'N', 'U', 'M', 'P', 'Y'};

// Read a little-endian integer of type T from `src`.
template <typename T>
T read_le(std::span<const std::byte> src) {
    T value{};
    std::memcpy(&value, src.data(), sizeof(T));
    if constexpr (std::endian::native == std::endian::big) {
        value = std::byteswap(value);
    }
    return value;
}

std::string_view trim(std::string_view s) {
    const auto begin = s.find_first_not_of(" \t\r\n");
    if (begin == std::string_view::npos) {
        return {};
    }
    return s.substr(begin, s.find_last_not_of(" \t\r\n") - begin + 1);
}

// Extract the value following "'key':" in the header dict. Returns everything
// up to the matching close of the value, which is enough for the three keys
// numpy writes (descr, fortran_order, shape).
std::string_view dict_value(std::string_view header, std::string_view key) {
    const std::string quoted = std::format("'{}'", key);
    auto pos = header.find(quoted);
    if (pos == std::string_view::npos) {
        throw NpyError{std::format("npy header missing key '{}'", key)};
    }
    pos = header.find(':', pos + quoted.size());
    if (pos == std::string_view::npos) {
        throw NpyError{std::format("npy header key '{}' has no value", key)};
    }
    ++pos;

    // Scan to the comma that ends this value, ignoring commas nested inside a
    // tuple/list (shape) or a quoted string (descr).
    int depth = 0;
    char quote = '\0';
    std::size_t end = pos;
    for (; end < header.size(); ++end) {
        const char c = header[end];
        if (quote != '\0') {
            if (c == quote) quote = '\0';
        } else if (c == '\'' || c == '"') {
            quote = c;
        } else if (c == '(' || c == '[') {
            ++depth;
        } else if (c == ')' || c == ']') {
            --depth;
        } else if (c == ',' && depth == 0) {
            break;
        } else if (c == '}' && depth == 0) {
            break;
        }
    }
    return trim(header.substr(pos, end - pos));
}

DType parse_descr(std::string_view descr) {
    descr = trim(descr);
    if (descr.size() < 2 || (descr.front() != '\'' && descr.front() != '"')) {
        throw NpyError{std::format(
            "unsupported npy descr {} (structured dtypes are not supported)",
            descr)};
    }
    descr = descr.substr(1, descr.size() - 2);

    // Leading byte-order character: '<' little, '>' big, '=' native, '|' n/a.
    char order = '|';
    if (!descr.empty() && (descr.front() == '<' || descr.front() == '>' ||
                           descr.front() == '=' || descr.front() == '|')) {
        order = descr.front();
        descr.remove_prefix(1);
    }
    if (descr.size() < 2) {
        throw NpyError{std::format("unsupported npy dtype '{}'", descr)};
    }

    const char kind = descr.front();
    std::size_t width = 0;
    const auto digits = descr.substr(1);
    if (std::from_chars(digits.data(), digits.data() + digits.size(), width)
            .ec != std::errc{}) {
        throw NpyError{std::format("unsupported npy dtype '{}'", descr)};
    }

    if (order == '>' && width > 1) {
        throw NpyError{"big-endian npy data is not supported"};
    }
    if (std::endian::native != std::endian::little) {
        throw NpyError{"boatforge requires a little-endian host"};
    }

    switch (kind) {
        case 'b':
            if (width == 1) return DType::Bool;
            break;
        case 'i':
            switch (width) {
                case 1: return DType::Int8;
                case 2: return DType::Int16;
                case 4: return DType::Int32;
                case 8: return DType::Int64;
                default: break;
            }
            break;
        case 'u':
            switch (width) {
                case 1: return DType::UInt8;
                case 2: return DType::UInt16;
                case 4: return DType::UInt32;
                case 8: return DType::UInt64;
                default: break;
            }
            break;
        case 'f':
            switch (width) {
                case 2: return DType::Float16;
                case 4: return DType::Float32;
                case 8: return DType::Float64;
                default: break;
            }
            break;
        default:
            break;
    }
    throw NpyError{std::format("unsupported npy dtype '{}{}'", kind, width)};
}

std::vector<std::size_t> parse_shape(std::string_view value) {
    value = trim(value);
    if (value.size() < 2 || value.front() != '(' || value.back() != ')') {
        throw NpyError{std::format("malformed npy shape {}", value)};
    }
    value = trim(value.substr(1, value.size() - 2));

    std::vector<std::size_t> shape;
    while (!value.empty()) {
        const auto comma = value.find(',');
        const auto item = trim(value.substr(0, comma));
        if (!item.empty()) {
            std::size_t dim = 0;
            if (std::from_chars(item.data(), item.data() + item.size(), dim).ec
                != std::errc{}) {
                throw NpyError{std::format("malformed npy shape entry '{}'", item)};
            }
            shape.push_back(dim);
        }
        if (comma == std::string_view::npos) {
            break;
        }
        value.remove_prefix(comma + 1);
    }
    return shape;
}

bool parse_bool(std::string_view value) {
    value = trim(value);
    if (value == "True") return true;
    if (value == "False") return false;
    throw NpyError{std::format("malformed npy boolean '{}'", value)};
}

}  // namespace

std::string_view to_string(DType dtype) {
    switch (dtype) {
        case DType::Bool: return "bool";
        case DType::Int8: return "int8";
        case DType::Int16: return "int16";
        case DType::Int32: return "int32";
        case DType::Int64: return "int64";
        case DType::UInt8: return "uint8";
        case DType::UInt16: return "uint16";
        case DType::UInt32: return "uint32";
        case DType::UInt64: return "uint64";
        case DType::Float16: return "float16";
        case DType::Float32: return "float32";
        case DType::Float64: return "float64";
    }
    return "<unknown>";
}

std::size_t word_size(DType dtype) {
    switch (dtype) {
        case DType::Bool:
        case DType::Int8:
        case DType::UInt8: return 1;
        case DType::Int16:
        case DType::UInt16:
        case DType::Float16: return 2;
        case DType::Int32:
        case DType::UInt32:
        case DType::Float32: return 4;
        case DType::Int64:
        case DType::UInt64:
        case DType::Float64: return 8;
    }
    return 0;
}

NpyArray::NpyArray(std::vector<std::size_t> shape, DType dtype,
                   bool fortran_order, std::vector<std::byte> data)
    : shape_{std::move(shape)},
      dtype_{dtype},
      fortran_order_{fortran_order},
      data_{std::move(data)} {
    size_ = 1;
    for (const std::size_t dim : shape_) {
        size_ *= dim;
    }
}

std::string NpyArray::shape_string() const {
    if (shape_.empty()) {
        return "()";  // 0-d array (numpy scalar)
    }
    std::string out = "(";
    for (std::size_t i = 0; i < shape_.size(); ++i) {
        out += std::format("{}{}", i > 0 ? ", " : "", shape_[i]);
    }
    // numpy renders a 1-d shape with a trailing comma.
    out += shape_.size() == 1 ? ",)" : ")";
    return out;
}

NpyArray parse_npy(std::span<const std::byte> image) {
    if (image.size() < 10 ||
        !std::equal(kMagic.begin(), kMagic.end(), image.begin(),
                    [](char a, std::byte b) { return std::byte(a) == b; })) {
        throw NpyError{"not a .npy file (bad magic)"};
    }

    const auto major = std::to_integer<unsigned>(image[6]);
    std::size_t header_len = 0;
    std::size_t header_start = 0;
    if (major == 1) {
        header_len = read_le<std::uint16_t>(image.subspan(8, 2));
        header_start = 10;
    } else if (major == 2 || major == 3) {
        if (image.size() < 12) {
            throw NpyError{"truncated .npy header"};
        }
        header_len = read_le<std::uint32_t>(image.subspan(8, 4));
        header_start = 12;
    } else {
        throw NpyError{std::format("unsupported .npy format version {}", major)};
    }

    if (image.size() < header_start + header_len) {
        throw NpyError{"truncated .npy header"};
    }

    const std::string_view header{
        reinterpret_cast<const char*>(image.data() + header_start), header_len};

    const DType dtype = parse_descr(dict_value(header, "descr"));
    const bool fortran_order = parse_bool(dict_value(header, "fortran_order"));
    std::vector<std::size_t> shape = parse_shape(dict_value(header, "shape"));

    std::size_t count = 1;
    for (const std::size_t dim : shape) {
        count *= dim;
    }
    const std::size_t payload_bytes = count * word_size(dtype);

    const auto payload = image.subspan(header_start + header_len);
    if (payload.size() < payload_bytes) {
        throw NpyError{std::format("truncated .npy payload: expected {} bytes, got {}",
                                   payload_bytes, payload.size())};
    }

    std::vector<std::byte> data(payload.begin(), payload.begin() + payload_bytes);
    return NpyArray{std::move(shape), dtype, fortran_order, std::move(data)};
}

}  // namespace boatforge
