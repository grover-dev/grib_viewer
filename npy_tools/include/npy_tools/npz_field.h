#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

#include <npy_tools/npy.h>

namespace boatforge {

// An environment field on a regular lat/lon grid and a uniform time axis, in
// the npz layout scripts/grib_npz.py writes. Nothing below is specific to
// radiation -- any field cut to that layout loads and samples identically.
//
// Sampling costs no search. The grid is regular and the time axis uniform, so
// a (t, lat, lon) query resolves to an index by arithmetic -- three divides,
// then eight strided loads for the trilinear blend. There is no tree to walk
// and no axis to binary-search, which is the whole reason the npz stores the
// axis constants alongside the cube.
//
// The field may live in one npz or in a directory of them split along time --
// see load() and load_directory(). Sampling reads the same either way; the
// difference is only how much of the field is in memory at once.
class NpzField {
public:
    // Throws NpyError if the archive is missing members, has the wrong dtypes,
    // or was written by a newer format version.
    static NpzField load(const std::filesystem::path& path);

    // A field spread over a directory of parts, as scripts/grib_npz.py writes
    // when one cube would exceed its byte limit. Every .npz directly in `dir`
    // is a part: same grid, same time step, its own window of the time axis.
    //
    // Nothing is loaded here. The constructor reads each part's axis constants
    // -- a few hundred bytes, skipping the payload entirely -- and keeps the
    // window each file covers. The first sample() loads the part that answers
    // it, and a sample outside every loaded part loads the one that holds it.
    // So the resident cost is `cached_parts` parts, not the whole field,
    // whatever the directory adds up to.
    //
    // `cached_parts` is what that costs against what it saves. One part is the
    // least memory and is enough for a single walk forward through time: each
    // boundary is crossed once. It is the wrong answer for callers that
    // interleave several walks at different times through one field -- Sim
    // does, stepping every run round-robin -- because each alternation across a
    // boundary would evict the part the other walk is about to want, and every
    // eviction costs a full decompression. Sizing the cache to the number of
    // distinct windows in flight turns that back into one load per part.
    //
    // Throws NpyError if `dir` holds no .npz, if any part is not a field npz,
    // or if the parts disagree on the grid or the time step -- they have to
    // interchange under one set of axis constants to be one field. Also throws
    // if `cached_parts` is 0.
    static NpzField load_directory(const std::filesystem::path& dir, std::size_t cached_parts = 2);

    // Which parts back this field, in time order. One entry naming the file
    // itself for a single-file load, so the two cases report the same shape.
    struct Part
    {
        std::filesystem::path path;
        std::chrono::seconds t0{0};   // first frame
        std::chrono::seconds end{0};  // last frame, inclusive
        std::size_t nt = 0;
    };

    const std::vector<Part>& parts() const
    {
        return parts_;
    }

    static constexpr std::size_t npos = static_cast<std::size_t>(-1);

    // Index into parts() of the part the last sample was answered from, or npos
    // before the first one -- which is the state a directory field starts in.
    std::size_t resident() const
    {
        return active_ == npos ? npos : cache_[active_].part;
    }

    // How many parts are in memory now, and the ceiling on that.
    std::size_t cached() const
    {
        return cache_.size();
    }

    std::size_t cache_limit() const
    {
        return cache_limit_;
    }

    // The field's value at a point, trilinear in (time, latitude, longitude),
    // in whatever unit the writer settled on. For the solar npz that is mean
    // downward short-wave irradiance in W/m^2, already converted from the J/m^2
    // accumulations ERA5 ships, so a panel model can use it directly.
    //
    // `when` is time since the Unix epoch, UTC -- which is what the npz time
    // axis holds. Coarser durations convert implicitly, so a blackboard's
    // std::chrono::minutes passes straight in.
    //
    // Longitude is accepted in any frame -- -180..180, 0..360, or unwrapped
    // past either -- since it is reduced modulo 360 against the grid origin. On
    // a global grid that also makes the antimeridian interpolate correctly
    // rather than fall off the end.
    //
    // False outside coverage, or where the source had no data; `value` is
    // written only when it returns true. The miss is a return value rather than
    // a NaN a caller has to remember to test for, so a sim that walks off the
    // end of the field stops on the false instead of carrying a NaN through its
    // state for the rest of the run.
    // For a directory field this may load a part, so it is const only in the
    // sense that the field it represents does not change -- not in the sense
    // that it is cheap or that two threads may call it on one instance. It
    // throws NpyError if the part it needs cannot be read (deleted or truncated
    // since the directory was indexed).
    bool sample(std::chrono::seconds when, double lat, double lon, float& value) const;

private:
    // Dequantised value at a grid node. NaN where the source marked no data.
    float at(std::size_t i, std::size_t j, std::size_t k) const;

    // Makes parts_[index] the part sampling reads from, loading it if it is not
    // already cached. No-op if it is already active, so the common case costs
    // one comparison.
    void make_active(std::size_t index) const;

    // Grid and time step, shared by every part -- checked at index time, since
    // parts that disagree cannot be sampled through one set of constants.
    std::chrono::seconds step_{0};  // between consecutive frames
    double lat0_ = 0.0, dlat_ = 0.0;
    std::size_t nlat_ = 0;
    double lon0_ = 0.0, dlon_ = 0.0;
    std::size_t nlon_ = 0;
    bool wrap_ = false;        // longitude axis spans the globe

    // Every part's window, ascending by t0 and never empty. Held by value so a
    // swap needs no second directory scan.
    std::vector<Part> parts_;

    // One part in memory, and what about the field is true only while it is.
    //
    // Payload is kept in the dtype the script wrote it in. A quantised cube is
    // half the size of the float one, and for a global month that difference
    // is gigabytes -- so it is dequantised per access rather than up front.
    // scale/offset/fill are read verbatim from the npz; quantised comes from
    // the payload's own dtype rather than a flag that could contradict it.
    struct Loaded
    {
        std::size_t part = npos;  // index into parts_
        std::chrono::seconds t0{0};
        std::size_t nt = 0;
        NpyArray data;
        bool quantised = false;
        double scale = 1.0, offset = 0.0;
        uint16_t fill = 0;
    };

    // A ring, not an LRU: a sim steps time forward, so parts are loaded in time
    // order and the oldest slot is always the one furthest behind the walk --
    // which is what an LRU would pick anyway, without the bookkeeping. A caller
    // that jumped around the time axis at random would get worse hit rates than
    // an LRU here, and should size the cache rather than expect the eviction
    // order to save it.
    //
    // Mutable because sampling a directory field is what triggers the load:
    // hiding that behind a const sample() keeps callers that hold a
    // `const NpzField&` -- and the single-file case, where nothing is ever
    // reloaded -- unchanged.
    mutable std::vector<Loaded> cache_;  // grows to cache_limit_, then is overwritten in place
    mutable std::size_t next_slot_ = 0;  // where the next load lands
    mutable std::size_t active_ = npos;  // slot the last sample read from
    std::size_t cache_limit_ = 1;
};

}  // namespace boatforge
