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
    // it; a later sample() outside that window loads the part that holds it and
    // drops the previous one. So the resident cost is one part, not one field,
    // whatever the directory adds up to.
    //
    // That makes a *forward* walk cheap -- one load per part crossed -- and a
    // walk that alternates across a boundary expensive, since each alternation
    // is a full decompression. Sims that share one field across runs at
    // different times (Sim does) will thrash if their windows differ; see the
    // note on sample().
    //
    // Throws NpyError if `dir` holds no .npz, if any part is not a field npz,
    // or if the parts disagree on the grid or the time step -- they have to
    // interchange under one set of axis constants to be one field.
    static NpzField load_directory(const std::filesystem::path& dir);

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

    // Index into parts() of the part currently in memory, or npos if none is --
    // which is the state a directory field starts in, before the first sample.
    static constexpr std::size_t npos = static_cast<std::size_t>(-1);

    std::size_t resident() const
    {
        return resident_;
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

    // Makes parts_[index] the resident part, replacing whatever was. No-op if
    // it already is, so the common case costs one comparison.
    void make_resident(std::size_t index) const;

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

    // The part in memory, and what about the field is true only while it is.
    // Mutable because sampling a directory field is what triggers the load:
    // hiding that behind a const sample() keeps callers that hold a
    // `const NpzField&` -- and the single-file case, where nothing is ever
    // reloaded -- unchanged.
    //
    // Payload is kept in the dtype the script wrote it in. A quantised cube is
    // half the size of the float one, and for a global month that difference
    // is gigabytes -- so it is dequantised per access rather than up front.
    // scale/offset/fill are read verbatim from the npz; quantised comes from
    // the payload's own dtype rather than a flag that could contradict it.
    mutable std::size_t resident_ = npos;
    mutable std::chrono::seconds t0_{0};  // frame 0 of the resident part
    mutable std::size_t nt_ = 0;
    mutable NpyArray data_;
    mutable bool quantised_ = false;
    mutable double scale_ = 1.0, offset_ = 0.0;
    mutable uint16_t fill_ = 0;
};

}  // namespace boatforge
