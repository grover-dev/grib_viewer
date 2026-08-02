#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>

#include <boatforge/npy.h>

namespace boatforge {

// Solar irradiance over a regular lat/lon grid on a uniform time axis, as
// written by scripts/solar_npz.py.
//
// Sampling costs no search. The grid is regular and the time axis uniform, so
// a (t, lat, lon) query resolves to an index by arithmetic -- three divides,
// then eight strided loads for the trilinear blend. There is no tree to walk
// and no axis to binary-search, which is the whole reason the npz stores the
// axis constants alongside the cube.
class SolarField {
public:
    // Throws NpyError if the archive is missing members, has the wrong dtypes,
    // or was written by a newer format version.
    static SolarField load(const std::filesystem::path& path);

    // Mean downward short-wave irradiance in W/m^2, trilinear in
    // (time, latitude, longitude). Already converted from the J/m^2
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
    // Returns NaN outside coverage, or where the source had no data.
    float sample(std::chrono::seconds when, double lat, double lon) const;

private:
    // Dequantised value at a grid node. NaN where the source marked no data.
    float at(std::size_t i, std::size_t j, std::size_t k) const;

    std::chrono::seconds t0_{0};    // frame 0, since the Unix epoch
    std::chrono::seconds step_{0};  // between consecutive frames
    std::size_t nt_ = 0;

    double lat0_ = 0.0, dlat_ = 0.0;
    std::size_t nlat_ = 0;
    double lon0_ = 0.0, dlon_ = 0.0;
    std::size_t nlon_ = 0;
    bool wrap_ = false;        // longitude axis spans the globe

    // Payload, kept in the dtype the script wrote it in. A quantised cube is
    // half the size of the float one, and for a global month that difference
    // is gigabytes -- so it is dequantised per access rather than up front.
    // scale_/offset_/fill_ are read verbatim from the npz; quantised_ comes
    // from the payload's own dtype rather than a flag that could contradict it.
    NpyArray data_;
    bool quantised_ = false;
    double scale_ = 1.0, offset_ = 0.0;
    uint16_t fill_ = 0;
};

}  // namespace boatforge
