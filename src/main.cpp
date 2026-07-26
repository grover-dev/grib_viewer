#include <cstddef>
#include <exception>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <cnpy.h>

#include <boatforge/version.hpp>

namespace {

constexpr std::size_t kPreviewCount = 6;

std::string format_shape(const std::vector<std::size_t>& shape) {
    if (shape.empty()) {
        return "()";  // 0-d array (numpy scalar)
    }
    std::string out = "(";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            out += ", ";
        }
        out += std::to_string(shape[i]);
    }
    out += ")";
    return out;
}

void print_preview(const cnpy::NpyArray& array) {
    // cnpy discards the numpy type char when parsing the header, so word size
    // is all we have to go on. Everything in track.npz is float64.
    if (array.word_size != sizeof(double)) {
        std::println("    <{}-byte elements, not previewed>", array.word_size);
        return;
    }

    const std::span values{array.data<double>(), array.num_vals};
    const std::size_t shown = std::min(values.size(), kPreviewCount);

    std::print("    [");
    for (std::size_t i = 0; i < shown; ++i) {
        std::print("{}{:.4f}", i > 0 ? ", " : "", values[i]);
    }
    if (shown < values.size()) {
        std::print(", ... ({} more)", values.size() - shown);
    }
    std::println("]");
}

}  // namespace

int main(int argc, char** argv) {
    const std::string path = argc > 1 ? argv[1] : "track.npz";

    std::println("boatforge {} — reading {}", boatforge::kVersion, path);

    try {
        const cnpy::npz_t npz = cnpy::npz_load(path);

        std::println("{} array(s):", npz.size());
        for (const auto& [name, array] : npz) {
            std::println("  {:<12} shape={:<10} word_size={} fortran_order={}",
                         name, format_shape(array.shape), array.word_size,
                         array.fortran_order);
            print_preview(array);
        }
    } catch (const std::exception& e) {
        std::println(stderr, "error: {}", e.what());
        return 1;
    }

    return 0;
}
