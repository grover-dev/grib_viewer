#include <npy_tools/npz_field.h>

#include <cmath>
#include <format>
#include <limits>
#include <string_view>
#include <utility>

#include <npy_tools/npz.h>

namespace boatforge
{
namespace
{

constexpr int32_t supported_version = 1;
constexpr float not_a_number = std::numeric_limits<float>::quiet_NaN();

const NpyArray& member(const NpzArchive& npz, std::string_view name)
{
    const auto it = npz.find(name);
    if (it == npz.end())
    {
        throw NpyError{std::format("not a gridded field npz: no '{}' array", name)};
    }
    return it->second;
}

// A 0-d member, read as a scalar. The writer emits these with fixed dtypes, so
// a mismatch means the file is not what it claims and is worth failing on.
template <typename T>
T scalar(const NpzArchive& npz, std::string_view name)
{
    const NpyArray& array = member(npz, name);
    if (array.size() != 1)
    {
        throw NpyError{std::format("'{}' should be a scalar, has shape {}", name, array.shape_string())};
    }
    try
    {
        return array.as<T>()[0];
    }
    catch (const NpyError& e)
    {
        throw NpyError{std::format("'{}': {}", name, e.what())};
    }
}

// Positive count, narrowed once so the sampler can index without casting.
std::size_t extent(const NpzArchive& npz, std::string_view name)
{
    const int64_t n = scalar<int64_t>(npz, name);
    if (n < 2)
    {
        throw NpyError{std::format("'{}' is {}; the grid needs at least 2 points per axis to interpolate", name, n)};
    }
    return static_cast<std::size_t>(n);
}

}  // namespace

NpzField NpzField::load(const std::filesystem::path& path)
{
    NpzArchive npz = load_npz(path);

    const int32_t version = scalar<int32_t>(npz, "version");
    if (version != supported_version)
    {
        throw NpyError{
            std::format("field npz is version {}, this build reads version {}; regenerate it "
                        "with scripts/grib_npz.py",
                        version, supported_version)};
    }

    NpzField field;
    field.t0_ = std::chrono::seconds{scalar<int64_t>(npz, "t0")};
    field.step_ = std::chrono::seconds{scalar<int64_t>(npz, "dt")};
    field.nt_ = extent(npz, "nt");
    field.lat0_ = scalar<double>(npz, "lat0");
    field.dlat_ = scalar<double>(npz, "dlat");
    field.nlat_ = extent(npz, "nlat");
    field.lon0_ = scalar<double>(npz, "lon0");
    field.dlon_ = scalar<double>(npz, "dlon");
    field.nlon_ = extent(npz, "nlon");
    field.wrap_ = scalar<int32_t>(npz, "lon_wrap") != 0;

    if (field.step_ <= std::chrono::seconds{0})
    {
        throw NpyError{std::format("time step is {} s; must be positive", field.step_.count())};
    }
    if (!(field.dlat_ > 0.0) || !(field.dlon_ > 0.0))
    {
        throw NpyError{
            std::format("grid steps must be positive and ascending, got dlat={} dlon={}", field.dlat_, field.dlon_)};
    }

    // Moved, not copied: a global month is gigabytes and the archive is dead
    // after this point anyway.
    const auto data_it = npz.find("data");
    if (data_it == npz.end())
    {
        throw NpyError{"not a gridded field npz: no 'data' array"};
    }
    field.data_ = std::move(data_it->second);

    const std::vector<std::size_t>& shape = field.data_.shape();
    if (shape.size() != 3 || shape[0] != field.nt_ || shape[1] != field.nlat_ || shape[2] != field.nlon_)
    {
        throw NpyError{std::format("'data' has shape {}, but the axes say ({}, {}, {})", field.data_.shape_string(),
                                   field.nt_, field.nlat_, field.nlon_)};
    }
    if (field.data_.fortran_order())
    {
        throw NpyError{"'data' is Fortran-ordered; the sampler indexes it C-order"};
    }

    // The dtype the writer chose is what says whether values are quantised;
    // scale/offset/fill are only meaningful for the integer payload.
    switch (field.data_.dtype())
    {
        case DType::UInt16:
            field.quantised_ = true;
            field.scale_ = scalar<double>(npz, "scale");
            field.offset_ = scalar<double>(npz, "offset");
            field.fill_ = static_cast<uint16_t>(scalar<int64_t>(npz, "fill"));
            break;
        case DType::Float32:
            field.quantised_ = false;
            break;
        default:
            throw NpyError{
                std::format("'data' is {}; expected uint16 (quantised) or float32", to_string(field.data_.dtype()))};
    }

    return field;
}

float NpzField::at(std::size_t i, std::size_t j, std::size_t k) const
{
    const std::size_t index = (i * nlat_ + j) * nlon_ + k;
    if (!quantised_)
    {
        return data_.as<float>()[index];
    }
    const uint16_t raw = data_.as<uint16_t>()[index];
    if (raw == fill_)
    {
        return not_a_number;
    }
    return static_cast<float>(static_cast<double>(raw) * scale_ + offset_);
}

bool NpzField::sample(std::chrono::seconds when, double lat, double lon, float& value) const
{
    if (nt_ == 0)
    {
        return false;
    }

    // --- fractional indices, by arithmetic ---------------------------------
    const double ti = static_cast<double>((when - t0_).count()) / static_cast<double>(step_.count());
    if (!(ti >= 0.0) || ti > static_cast<double>(nt_ - 1))
    {
        return false;  // also rejects NaN input, via the negated comparison
    }

    const double yi = (lat - lat0_) / dlat_;
    if (!(yi >= 0.0) || yi > static_cast<double>(nlat_ - 1))
    {
        return false;
    }

    // Measuring longitude as an offset east of the grid origin, modulo 360,
    // makes every input frame (-180..180, 0..360, or unwrapped) land on the
    // same index -- and puts a query just west of the origin far past the east
    // edge, where the bounds check below correctly rejects it on a regional
    // grid. It is the same trick grib_utils.crop_message uses to pick columns.
    double east = std::fmod(lon - lon0_, 360.0);
    if (east < 0.0)
    {
        east += 360.0;
    }
    const double xi = east / dlon_;
    if (!(xi >= 0.0) || (!wrap_ && xi > static_cast<double>(nlon_ - 1)))
    {
        return false;
    }

    // --- corners -----------------------------------------------------------
    const std::size_t i0 = static_cast<std::size_t>(ti);
    const std::size_t j0 = static_cast<std::size_t>(yi);
    const std::size_t k0 = static_cast<std::size_t>(xi);

    const std::size_t i1 = i0 + 1 < nt_ ? i0 + 1 : i0;
    const std::size_t j1 = j0 + 1 < nlat_ ? j0 + 1 : j0;
    // On a global axis the cell after the last one is the first one again, so
    // the seam blends instead of clamping.
    const std::size_t k1 = k0 + 1 < nlon_ ? k0 + 1 : (wrap_ ? 0 : k0);

    const double ft = ti - static_cast<double>(i0);
    const double fy = yi - static_cast<double>(j0);
    const double fx = xi - static_cast<double>(k0);

    const double c000 = at(i0, j0, k0), c001 = at(i0, j0, k1);
    const double c010 = at(i0, j1, k0), c011 = at(i0, j1, k1);
    const double c100 = at(i1, j0, k0), c101 = at(i1, j0, k1);
    const double c110 = at(i1, j1, k0), c111 = at(i1, j1, k1);

    // A hole anywhere in the cell makes the blend meaningless, so it propagates
    // rather than being silently filled from the corners that do have data.
    const double c00 = c000 + (c001 - c000) * fx;
    const double c01 = c010 + (c011 - c010) * fx;
    const double c10 = c100 + (c101 - c100) * fx;
    const double c11 = c110 + (c111 - c110) * fx;

    const double c0 = c00 + (c01 - c00) * fy;
    const double c1 = c10 + (c11 - c10) * fy;

    const float blended = static_cast<float>(c0 + (c1 - c0) * ft);
    /* A hole anywhere in the cell reaches here as a NaN, and it is a miss for
     * the same reason being off the grid is. */
    if (std::isnan(blended))
    {
        return false;
    }

    value = blended;
    return true;
}

}  // namespace boatforge
