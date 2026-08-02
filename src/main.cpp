#include <cstddef>
#include <exception>
#include <print>
#include <string>

#include <boatforge/npz.h>
#include <boatforge/version.h>

namespace {

constexpr std::size_t kPreviewCount = 6;

template <typename T>
void print_values(const boatforge::NpyArray& array) {
    const auto values = array.as<T>();
    const std::size_t shown = std::min(values.size(), kPreviewCount);

    std::print("    [");
    for (std::size_t i = 0; i < shown; ++i) {
        std::print("{}{}", i > 0 ? ", " : "", values[i]);
    }
    if (shown < values.size()) {
        std::print(", ... {} more", values.size() - shown);
    }
    std::println("]");
}

void print_preview(const boatforge::NpyArray& array) {
    using boatforge::DType;
    switch (array.dtype()) {
        case DType::Bool: print_values<bool>(array); break;
        case DType::Int8: print_values<std::int8_t>(array); break;
        case DType::Int16: print_values<std::int16_t>(array); break;
        case DType::Int32: print_values<std::int32_t>(array); break;
        case DType::Int64: print_values<std::int64_t>(array); break;
        case DType::UInt8: print_values<std::uint8_t>(array); break;
        case DType::UInt16: print_values<std::uint16_t>(array); break;
        case DType::UInt32: print_values<std::uint32_t>(array); break;
        case DType::UInt64: print_values<std::uint64_t>(array); break;
        case DType::Float32: print_values<float>(array); break;
        case DType::Float64: print_values<double>(array); break;
        case DType::Float16:
            std::println("    <float16 not previewed>");
            break;
    }
}

}  // namespace



int main(int argc, char** argv) {
    /**
     * How to start?
     * - Map has been created, lets architect the iteration loop
     *   - Will probably need to optimize, problem for later
     *
     *
     *
     */


    const std::string path = argc > 1 ? argv[1] : "track.npz";

    std::println("boatforge {} — reading {}", boatforge::kVersion, path);

    try {
        const boatforge::NpzArchive npz = boatforge::load_npz(path);

        std::println("{} array(s):", npz.size());
        for (const auto& [name, array] : npz) {
            std::println("  {:<12} {:<8} shape={:<10} order={}", name,
                         boatforge::to_string(array.dtype()),
                         array.shape_string(),
                         array.fortran_order() ? 'F' : 'C');
            print_preview(array);
        }
    } catch (const std::exception& e) {
        std::println(stderr, "error: {}", e.what());
        return 1;
    }

    return 0;
}
