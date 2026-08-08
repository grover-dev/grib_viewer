#include <npy_tools/npz_field.h>

#include <algorithm>
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

// Everything about a part except its payload. Read on its own so a directory
// can be indexed without inflating a single cube.
struct Axes
{
    std::chrono::seconds t0{0};
    std::chrono::seconds step{0};
    std::size_t nt = 0;
    double lat0 = 0.0, dlat = 0.0;
    std::size_t nlat = 0;
    double lon0 = 0.0, dlon = 0.0;
    std::size_t nlon = 0;
    bool wrap = false;
};

Axes read_axes(const NpzArchive& npz)
{
    const int32_t version = scalar<int32_t>(npz, "version");
    if (version != supported_version)
    {
        throw NpyError{
            std::format("field npz is version {}, this build reads version {}; regenerate it "
                        "with scripts/grib_npz.py",
                        version, supported_version)};
    }

    Axes axes;
    axes.t0 = std::chrono::seconds{scalar<int64_t>(npz, "t0")};
    axes.step = std::chrono::seconds{scalar<int64_t>(npz, "dt")};
    axes.nt = extent(npz, "nt");
    axes.lat0 = scalar<double>(npz, "lat0");
    axes.dlat = scalar<double>(npz, "dlat");
    axes.nlat = extent(npz, "nlat");
    axes.lon0 = scalar<double>(npz, "lon0");
    axes.dlon = scalar<double>(npz, "dlon");
    axes.nlon = extent(npz, "nlon");
    axes.wrap = scalar<int32_t>(npz, "lon_wrap") != 0;

    if (axes.step <= std::chrono::seconds{0})
    {
        throw NpyError{std::format("time step is {} s; must be positive", axes.step.count())};
    }
    if (!(axes.dlat > 0.0) || !(axes.dlon > 0.0))
    {
        throw NpyError{
            std::format("grid steps must be positive and ascending, got dlat={} dlon={}", axes.dlat, axes.dlon)};
    }
    return axes;
}

// Just the axis constants: every member except the payload, which is the one
// that costs anything to read.
NpzArchive load_axes_only(const std::filesystem::path& path)
{
    return load_npz(path, [](std::string_view name) { return name != "data"; });
}

// The window a part covers, last frame inclusive.
std::chrono::seconds last_frame(const Axes& axes)
{
    return axes.t0 + axes.step * static_cast<int64_t>(axes.nt - 1);
}

}  // namespace

NpzField NpzField::load(const std::filesystem::path& path)
{
    const Axes axes = read_axes(load_axes_only(path));

    NpzField field;
    field.step_ = axes.step;
    field.lat0_ = axes.lat0;
    field.dlat_ = axes.dlat;
    field.nlat_ = axes.nlat;
    field.lon0_ = axes.lon0;
    field.dlon_ = axes.dlon;
    field.nlon_ = axes.nlon;
    field.wrap_ = axes.wrap;
    field.parts_.push_back(Part{path, axes.t0, last_frame(axes), axes.nt});

    // Loaded here rather than on first use: a single-file field has nothing to
    // defer to, and a caller that got a field back has always been able to
    // assume the file was readable.
    field.make_resident(0);
    return field;
}

NpzField NpzField::load_directory(const std::filesystem::path& dir)
{
    if (!std::filesystem::is_directory(dir))
    {
        throw NpyError{std::format("{} is not a directory", dir.string())};
    }

    std::vector<std::filesystem::path> files;
    for (const auto& entry : std::filesystem::directory_iterator{dir})
    {
        if (entry.is_regular_file() && entry.path().extension() == ".npz")
        {
            files.push_back(entry.path());
        }
    }
    if (files.empty())
    {
        throw NpyError{std::format("{} holds no .npz parts", dir.string())};
    }
    // directory_iterator has no defined order; parts are named to sort, but the
    // real ordering happens on t0 below, so a renamed part still lands right.
    std::sort(files.begin(), files.end());

    NpzField field;
    bool first = true;
    for (const auto& path : files)
    {
        Axes axes;
        try
        {
            axes = read_axes(load_axes_only(path));
        }
        catch (const NpyError& e)
        {
            throw NpyError{std::format("{}: {}", path.string(), e.what())};
        }

        if (first)
        {
            field.step_ = axes.step;
            field.lat0_ = axes.lat0;
            field.dlat_ = axes.dlat;
            field.nlat_ = axes.nlat;
            field.lon0_ = axes.lon0;
            field.dlon_ = axes.dlon;
            field.nlon_ = axes.nlon;
            field.wrap_ = axes.wrap;
            first = false;
        }
        else if (axes.step != field.step_ || axes.lat0 != field.lat0_ || axes.dlat != field.dlat_ ||
                 axes.nlat != field.nlat_ || axes.lon0 != field.lon0_ || axes.dlon != field.dlon_ ||
                 axes.nlon != field.nlon_ || axes.wrap != field.wrap_)
        {
            // Only the time window may differ between parts. Anything else and
            // the parts are not one field, and a sample would silently mean a
            // different thing depending on which one happened to be resident.
            throw NpyError{std::format("{} does not match the other parts in {}: the grid and time step "
                                       "must be identical across a split field",
                                       path.string(), dir.string())};
        }

        field.parts_.push_back(Part{path, axes.t0, last_frame(axes), axes.nt});
    }

    std::sort(field.parts_.begin(), field.parts_.end(),
              [](const Part& a, const Part& b) { return a.t0 < b.t0; });

    // Deliberately not loaded: the point of a directory is that the first
    // sample decides which part is worth reading.
    return field;
}

void NpzField::make_resident(std::size_t index) const
{
    if (index == resident_)
    {
        return;
    }

    const Part& part = parts_[index];
    NpzArchive npz = load_npz(part.path);

    const auto data_it = npz.find("data");
    if (data_it == npz.end())
    {
        throw NpyError{std::format("{}: not a gridded field npz: no 'data' array", part.path.string())};
    }

    // Assigned, not swapped in beside the old one: the previous payload is
    // released here, so peak memory stays at one part rather than two. Moved
    // rather than copied because the archive is dead after this point anyway.
    data_ = std::move(data_it->second);
    t0_ = part.t0;
    nt_ = part.nt;

    const std::vector<std::size_t>& shape = data_.shape();
    if (shape.size() != 3 || shape[0] != nt_ || shape[1] != nlat_ || shape[2] != nlon_)
    {
        resident_ = npos;
        throw NpyError{std::format("{}: 'data' has shape {}, but the axes say ({}, {}, {})", part.path.string(),
                                   data_.shape_string(), nt_, nlat_, nlon_)};
    }
    if (data_.fortran_order())
    {
        resident_ = npos;
        throw NpyError{std::format("{}: 'data' is Fortran-ordered; the sampler indexes it C-order",
                                   part.path.string())};
    }

    // The dtype the writer chose is what says whether values are quantised;
    // scale/offset/fill are only meaningful for the integer payload. Read per
    // part, not once for the field: each part carries its own, and while
    // grib_npz.py quantises a split field against one shared range, a directory
    // assembled by other means need not.
    switch (data_.dtype())
    {
        case DType::UInt16:
            quantised_ = true;
            scale_ = scalar<double>(npz, "scale");
            offset_ = scalar<double>(npz, "offset");
            fill_ = static_cast<uint16_t>(scalar<int64_t>(npz, "fill"));
            break;
        case DType::Float32:
            quantised_ = false;
            break;
        default:
            resident_ = npos;
            throw NpyError{std::format("{}: 'data' is {}; expected uint16 (quantised) or float32",
                                       part.path.string(), to_string(data_.dtype()))};
    }

    resident_ = index;
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
    // --- which part answers this ------------------------------------------
    // The resident part is tried first, and on a forward walk it keeps
    // answering until the field steps off its end -- so the scan below, and the
    // load after it, happen once per boundary crossed rather than once per
    // sample. The scan being linear costs nothing next to the load it precedes.
    if (resident_ == npos || when < t0_ || when > parts_[resident_].end)
    {
        std::size_t found = npos;
        for (std::size_t i = 0; i < parts_.size(); ++i)
        {
            if (when >= parts_[i].t0 && when <= parts_[i].end)
            {
                found = i;
                break;
            }
        }
        if (found == npos)
        {
            // Before the first part, after the last, or in a gap between two.
            // A miss either way, and the same miss as falling off the end of a
            // single-file field -- nothing is loaded to find that out.
            return false;
        }
        make_resident(found);
    }

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
