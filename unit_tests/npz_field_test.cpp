// Tests for boatforge::NpzField against real ERA5 ssrd cut down to fixtures by
// unit_tests/fixtures/make_fixtures.py.
//
// Expected values are read out of the fixtures at runtime rather than baked in,
// so regenerating them from a different GRIB does not invalidate the suite --
// what the tests pin down is the *behaviour* (index arithmetic, interpolation,
// wrapping, bounds, dequantisation), not the particular numbers ERA5 happened
// to record over the Atlantic on 1 January 2025.

#include <boatforge/npz_field.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <string>

#include <boatforge/npy.h>
#include <boatforge/npz.h>
#include <gtest/gtest.h>

namespace {

using namespace std::chrono_literals;
using boatforge::NpyError;
using boatforge::NpzField;
using std::chrono::seconds;

std::filesystem::path fixture(const std::string& name) {
    return std::filesystem::path{BOATFORGE_FIXTURE_DIR} / name;
}

// The fixture's own contents, read straight through the npz reader. Tests use
// this as the oracle: it is the same bytes NpzField loads, but addressed by
// hand, so a bug in the index arithmetic cannot hide behind it.
struct Reference {
    boatforge::NpzArchive npz;

    int64_t t0, dt, nt;
    double lat0, dlat, lon0, dlon;
    int64_t nlat, nlon;
    bool wrap;
    double scale, offset;
    bool quantised;

    explicit Reference(const std::string& name)
        : npz{boatforge::load_npz(fixture(name))} {
        t0 = scalar<int64_t>("t0");
        dt = scalar<int64_t>("dt");
        nt = scalar<int64_t>("nt");
        lat0 = scalar<double>("lat0");
        dlat = scalar<double>("dlat");
        nlat = scalar<int64_t>("nlat");
        lon0 = scalar<double>("lon0");
        dlon = scalar<double>("dlon");
        nlon = scalar<int64_t>("nlon");
        wrap = scalar<int32_t>("lon_wrap") != 0;
        quantised = npz.at("data").dtype() == boatforge::DType::UInt16;
        scale = quantised ? scalar<double>("scale") : 1.0;
        offset = quantised ? scalar<double>("offset") : 0.0;
    }

    template <typename T>
    T scalar(const std::string& name) const {
        return npz.at(name).template as<T>()[0];
    }

    // Value at a grid node, dequantised the same way the writer quantised it.
    double node(int64_t i, int64_t j, int64_t k) const {
        const std::size_t index =
            static_cast<std::size_t>((i * nlat + j) * nlon + k);
        const boatforge::NpyArray& data = npz.at("data");
        if (!quantised) {
            return data.as<float>()[index];
        }
        return static_cast<double>(data.as<uint16_t>()[index]) * scale + offset;
    }

    seconds time_at(int64_t i) const { return seconds{t0 + i * dt}; }
    double lat_at(int64_t j) const { return lat0 + dlat * static_cast<double>(j); }
    double lon_at(int64_t k) const { return lon0 + dlon * static_cast<double>(k); }

    seconds mid_time() const { return time_at(nt / 2); }
    double mid_lat() const { return lat_at(nlat / 2); }
    double mid_lon() const { return lon_at(nlon / 2); }
};

// Quantisation error is bounded by half a step, and the sampler returns float,
// so anything at or below this is agreement rather than a discrepancy.
double tolerance(const Reference& ref) {
    return std::max(ref.scale, 1e-3);
}

// --------------------------------------------------------------------------
// Loading
// --------------------------------------------------------------------------

TEST(NpzFieldLoad, ReadsEachFixture) {
    for (const char* name : {"region_u16.npz", "region_f32.npz", "global_u16.npz"}) {
        SCOPED_TRACE(name);
        EXPECT_NO_THROW({ NpzField::load(fixture(name)); });
    }
}

TEST(NpzFieldLoad, MissingFileThrows) {
    EXPECT_THROW(NpzField::load(fixture("does_not_exist.npz")), NpyError);
}

TEST(NpzFieldLoad, NonArchiveThrows) {
    // The generator script is definitely not a zip.
    const auto not_an_npz =
        std::filesystem::path{BOATFORGE_FIXTURE_DIR} / "make_fixtures.py";
    ASSERT_TRUE(std::filesystem::exists(not_an_npz));
    EXPECT_THROW(NpzField::load(not_an_npz), NpyError);
}

TEST(NpzFieldLoad, ArchiveWithoutTheExpectedMembersThrows) {
    // track.npz is a real npz from this project, but it holds a route, not a
    // field -- so it must be rejected for missing members, not misread.
    const auto track = std::filesystem::path{BOATFORGE_FIXTURE_DIR} / ".." / ".." / "track.npz";
    if (!std::filesystem::exists(track)) {
        GTEST_SKIP() << "track.npz not present";
    }
    EXPECT_THROW(NpzField::load(track), NpyError);
}

// --------------------------------------------------------------------------
// Grid nodes: with every interpolation weight at 0 or 1, sampling must return
// the stored value exactly. This is what pins the index arithmetic.
// --------------------------------------------------------------------------

TEST(NpzFieldNodes, ExactAtEveryNodeOfTheRegionalGrid) {
    const Reference ref{"region_u16.npz"};
    const NpzField field = NpzField::load(fixture("region_u16.npz"));
    const double tol = tolerance(ref);

    std::size_t checked = 0;
    for (int64_t i = 0; i < ref.nt; ++i) {
        for (int64_t j = 0; j < ref.nlat; j += 7) {
            for (int64_t k = 0; k < ref.nlon; k += 7) {
                const float got =
                    field.sample(ref.time_at(i), ref.lat_at(j), ref.lon_at(k));
                ASSERT_NEAR(got, ref.node(i, j, k), tol)
                    << "node (" << i << ", " << j << ", " << k << ")";
                ++checked;
            }
        }
    }
    EXPECT_GT(checked, 100u);
}

TEST(NpzFieldNodes, ExactAtTheCornersOfCoverage) {
    const Reference ref{"region_u16.npz"};
    const NpzField field = NpzField::load(fixture("region_u16.npz"));
    const double tol = tolerance(ref);

    for (const int64_t i : {int64_t{0}, ref.nt - 1}) {
        for (const int64_t j : {int64_t{0}, ref.nlat - 1}) {
            for (const int64_t k : {int64_t{0}, ref.nlon - 1}) {
                EXPECT_NEAR(field.sample(ref.time_at(i), ref.lat_at(j), ref.lon_at(k)),
                            ref.node(i, j, k), tol)
                    << "corner (" << i << ", " << j << ", " << k << ")";
            }
        }
    }
}

// --------------------------------------------------------------------------
// Interpolation
// --------------------------------------------------------------------------

TEST(NpzFieldInterpolation, MidpointInTimeIsTheMeanOfTheNeighbouringFrames) {
    const Reference ref{"region_u16.npz"};
    const NpzField field = NpzField::load(fixture("region_u16.npz"));
    const int64_t j = ref.nlat / 3, k = ref.nlon / 3;

    for (int64_t i = 0; i + 1 < ref.nt; ++i) {
        const seconds half{ref.t0 + i * ref.dt + ref.dt / 2};
        EXPECT_NEAR(field.sample(half, ref.lat_at(j), ref.lon_at(k)),
                    0.5 * (ref.node(i, j, k) + ref.node(i + 1, j, k)),
                    tolerance(ref))
            << "between frames " << i << " and " << i + 1;
    }
}

TEST(NpzFieldInterpolation, MidpointInSpaceIsTheMeanOfTheFourSurroundingNodes) {
    const Reference ref{"region_u16.npz"};
    const NpzField field = NpzField::load(fixture("region_u16.npz"));
    const int64_t i = ref.nt / 2;

    for (int64_t j = 0; j + 1 < ref.nlat; j += 13) {
        for (int64_t k = 0; k + 1 < ref.nlon; k += 13) {
            const double expected =
                0.25 * (ref.node(i, j, k) + ref.node(i, j, k + 1) +
                        ref.node(i, j + 1, k) + ref.node(i, j + 1, k + 1));
            EXPECT_NEAR(field.sample(ref.time_at(i),
                                     ref.lat_at(j) + ref.dlat / 2,
                                     ref.lon_at(k) + ref.dlon / 2),
                        expected, tolerance(ref))
                << "cell (" << j << ", " << k << ")";
        }
    }
}

TEST(NpzFieldInterpolation, IsMonotoneBetweenTwoNodes) {
    // Along one edge of one cell the blend reduces to a lerp, so the samples
    // must move monotonically from one node's value to the other's.
    const Reference ref{"region_u16.npz"};
    const NpzField field = NpzField::load(fixture("region_u16.npz"));
    const int64_t i = ref.nt / 2, j = ref.nlat / 2, k = ref.nlon / 2;

    const double lo = ref.node(i, j, k), hi = ref.node(i, j, k + 1);
    ASSERT_NE(lo, hi) << "picked a flat cell; the test would be vacuous";

    float previous = field.sample(ref.time_at(i), ref.lat_at(j), ref.lon_at(k));
    for (int step = 1; step <= 10; ++step) {
        const double f = step / 10.0;
        const float got = field.sample(ref.time_at(i), ref.lat_at(j),
                                       ref.lon_at(k) + f * ref.dlon);
        EXPECT_NEAR(got, lo + (hi - lo) * f, tolerance(ref)) << "at f=" << f;
        if (hi > lo) {
            EXPECT_GE(got, previous - tolerance(ref)) << "at f=" << f;
        } else {
            EXPECT_LE(got, previous + tolerance(ref)) << "at f=" << f;
        }
        previous = got;
    }
}

// --------------------------------------------------------------------------
// Bounds. A query off the grid is NaN, not a clamped edge value -- silently
// returning the nearest coast would look plausible and be wrong.
// --------------------------------------------------------------------------

TEST(NpzFieldBounds, OutsideTheTimeAxisIsNaN) {
    const Reference ref{"region_u16.npz"};
    const NpzField field = NpzField::load(fixture("region_u16.npz"));
    const seconds last{ref.t0 + (ref.nt - 1) * ref.dt};

    EXPECT_TRUE(std::isnan(field.sample(seconds{ref.t0} - 1s, ref.mid_lat(), ref.mid_lon())));
    EXPECT_TRUE(std::isnan(field.sample(last + 1s, ref.mid_lat(), ref.mid_lon())));
    EXPECT_FALSE(std::isnan(field.sample(seconds{ref.t0}, ref.mid_lat(), ref.mid_lon())));
    EXPECT_FALSE(std::isnan(field.sample(last, ref.mid_lat(), ref.mid_lon())));
}

TEST(NpzFieldBounds, OutsideTheLatitudeAxisIsNaN) {
    const Reference ref{"region_u16.npz"};
    const NpzField field = NpzField::load(fixture("region_u16.npz"));
    const double north = ref.lat_at(ref.nlat - 1);

    EXPECT_TRUE(std::isnan(field.sample(ref.mid_time(), ref.lat0 - 0.01, ref.mid_lon())));
    EXPECT_TRUE(std::isnan(field.sample(ref.mid_time(), north + 0.01, ref.mid_lon())));
    EXPECT_FALSE(std::isnan(field.sample(ref.mid_time(), ref.lat0, ref.mid_lon())));
    EXPECT_FALSE(std::isnan(field.sample(ref.mid_time(), north, ref.mid_lon())));
}

TEST(NpzFieldBounds, OutsideTheLongitudeAxisIsNaNWhenTheGridDoesNotWrap) {
    const Reference ref{"region_u16.npz"};
    ASSERT_FALSE(ref.wrap);
    const NpzField field = NpzField::load(fixture("region_u16.npz"));
    const double east = ref.lon_at(ref.nlon - 1);

    EXPECT_TRUE(std::isnan(field.sample(ref.mid_time(), ref.mid_lat(), ref.lon0 - 0.01)));
    EXPECT_TRUE(std::isnan(field.sample(ref.mid_time(), ref.mid_lat(), east + 0.01)));
    // The far side of the planet is off this grid too, not wrapped onto it.
    EXPECT_TRUE(std::isnan(field.sample(ref.mid_time(), ref.mid_lat(), ref.lon0 + 180.0)));
    EXPECT_FALSE(std::isnan(field.sample(ref.mid_time(), ref.mid_lat(), ref.lon0)));
    EXPECT_FALSE(std::isnan(field.sample(ref.mid_time(), ref.mid_lat(), east)));
}

TEST(NpzFieldBounds, NaNQueriesAreNaN) {
    const Reference ref{"region_u16.npz"};
    const NpzField field = NpzField::load(fixture("region_u16.npz"));
    const double nan = std::nan("");

    EXPECT_TRUE(std::isnan(field.sample(ref.mid_time(), nan, ref.mid_lon())));
    EXPECT_TRUE(std::isnan(field.sample(ref.mid_time(), ref.mid_lat(), nan)));
}

// --------------------------------------------------------------------------
// Longitude handling
// --------------------------------------------------------------------------

TEST(NpzFieldLongitude, AnyFrameOfReferenceGivesTheSameSample) {
    // -180..180, 0..360 and unwrapped inputs must all land on the same cell.
    for (const char* name : {"region_u16.npz", "global_u16.npz"}) {
        SCOPED_TRACE(name);
        const Reference ref{name};
        const NpzField field = NpzField::load(fixture(name));

        for (int64_t k = 0; k + 1 < ref.nlon; k += 11) {
            const double lon = ref.lon_at(k) + 0.3 * ref.dlon;
            const float base = field.sample(ref.mid_time(), ref.mid_lat(), lon);
            ASSERT_FALSE(std::isnan(base)) << "lon " << lon;
            for (const double turns : {-720.0, -360.0, 360.0, 720.0}) {
                EXPECT_FLOAT_EQ(
                    field.sample(ref.mid_time(), ref.mid_lat(), lon + turns), base)
                    << "lon " << lon << " shifted by " << turns;
            }
        }
    }
}

TEST(NpzFieldLongitude, GlobalGridBlendsAcrossTheAntimeridian) {
    const Reference ref{"global_u16.npz"};
    ASSERT_TRUE(ref.wrap);
    const NpzField field = NpzField::load(fixture("global_u16.npz"));

    const int64_t i = ref.nt / 2, j = ref.nlat / 2;
    const int64_t last = ref.nlon - 1;
    const double west = ref.lon_at(last);           // final column
    const double lo = ref.node(i, j, last);         // ... and the first column,
    const double hi = ref.node(i, j, 0);            // which is its eastern neighbour

    for (const double f : {0.0, 0.25, 0.5, 0.75}) {
        EXPECT_NEAR(field.sample(ref.time_at(i), ref.lat_at(j), west + f * ref.dlon),
                    lo + (hi - lo) * f, tolerance(ref))
            << "fraction " << f << " across the seam";
    }
    // Stepping a full cell past the last column arrives back at the first one.
    EXPECT_NEAR(field.sample(ref.time_at(i), ref.lat_at(j), west + ref.dlon),
                hi, tolerance(ref));
}

TEST(NpzFieldLongitude, GlobalGridHasNoUnreachableLongitude) {
    const Reference ref{"global_u16.npz"};
    const NpzField field = NpzField::load(fixture("global_u16.npz"));

    for (double lon = -180.0; lon < 180.0; lon += 0.5) {
        EXPECT_FALSE(std::isnan(field.sample(ref.mid_time(), 0.0, lon)))
            << "lon " << lon;
    }
}

TEST(NpzFieldLongitude, PolesAreInsideCoverageButBeyondThemIsNot) {
    const Reference ref{"global_u16.npz"};
    const NpzField field = NpzField::load(fixture("global_u16.npz"));

    EXPECT_FALSE(std::isnan(field.sample(ref.mid_time(), -90.0, 0.0)));
    EXPECT_FALSE(std::isnan(field.sample(ref.mid_time(), 90.0, 0.0)));
    EXPECT_TRUE(std::isnan(field.sample(ref.mid_time(), -90.001, 0.0)));
    EXPECT_TRUE(std::isnan(field.sample(ref.mid_time(), 90.001, 0.0)));
}

// --------------------------------------------------------------------------
// Payload dtypes
// --------------------------------------------------------------------------

TEST(NpzFieldDtype, QuantisedAgreesWithFloat32OverTheSameWindow) {
    // Same GRIB window written both ways: --dtype u16 must not cost more than
    // half a quantisation step against the float it was derived from.
    const Reference quantised{"region_u16.npz"};
    ASSERT_TRUE(quantised.quantised);
    const Reference exact{"region_f32.npz"};
    ASSERT_FALSE(exact.quantised);
    ASSERT_EQ(quantised.nt, exact.nt);
    ASSERT_EQ(quantised.nlat, exact.nlat);
    ASSERT_EQ(quantised.nlon, exact.nlon);

    const NpzField a = NpzField::load(fixture("region_u16.npz"));
    const NpzField b = NpzField::load(fixture("region_f32.npz"));

    double worst = 0.0, peak = 0.0;
    for (int64_t i = 0; i < exact.nt; ++i) {
        for (int64_t j = 0; j < exact.nlat; j += 5) {
            for (int64_t k = 0; k < exact.nlon; k += 5) {
                const seconds t = exact.time_at(i);
                const double lat = exact.lat_at(j), lon = exact.lon_at(k);
                const double got = a.sample(t, lat, lon);
                const double want = b.sample(t, lat, lon);
                worst = std::max(worst, std::abs(got - want));
                peak = std::max(peak, want);
            }
        }
    }
    ASSERT_GT(peak, 1.0) << "window is entirely dark; the comparison proves nothing";
    EXPECT_LE(worst, quantised.scale) << "quantisation drifted beyond one step";
}

TEST(NpzFieldDtype, Float32PathReturnsStoredValuesVerbatim) {
    const Reference ref{"region_f32.npz"};
    const NpzField field = NpzField::load(fixture("region_f32.npz"));

    for (int64_t j = 0; j < ref.nlat; j += 9) {
        for (int64_t k = 0; k < ref.nlon; k += 9) {
            EXPECT_FLOAT_EQ(field.sample(ref.time_at(1), ref.lat_at(j), ref.lon_at(k)),
                            static_cast<float>(ref.node(1, j, k)));
        }
    }
}

// --------------------------------------------------------------------------
// Physical sanity of what the extractor produced. These would catch a unit
// error in solar_npz.py that all the arithmetic tests above would sail past.
// --------------------------------------------------------------------------

TEST(SolarPhysics, IrradianceIsNonNegativeAndBelowTheSolarConstant) {
    const Reference ref{"global_u16.npz"};
    const NpzField field = NpzField::load(fixture("global_u16.npz"));

    for (int64_t i = 0; i < ref.nt; ++i) {
        for (int64_t j = 0; j < ref.nlat; ++j) {
            for (int64_t k = 0; k < ref.nlon; ++k) {
                const float w = field.sample(ref.time_at(i), ref.lat_at(j), ref.lon_at(k));
                ASSERT_FALSE(std::isnan(w)) << "hole at (" << i << "," << j << "," << k << ")";
                ASSERT_GE(w, 0.0f);
                // Surface downward short-wave cannot exceed what arrives at the
                // top of the atmosphere; 1400 W/m^2 is that, rounded up.
                ASSERT_LE(w, 1400.0f) << "at lat " << ref.lat_at(j);
            }
        }
    }
}

TEST(SolarPhysics, JanuaryPolarNightIsDarkAndAntarcticSummerIsNot) {
    // The global fixture is 1 January. The Arctic is in polar night and the
    // Antarctic is in 24-hour daylight; if J/m^2 were divided by the wrong
    // interval both would still be ordered this way, but a sign or axis flip
    // -- latitude stored descending, say -- would invert it.
    const Reference ref{"global_u16.npz"};
    const NpzField field = NpzField::load(fixture("global_u16.npz"));

    double arctic = 0.0, antarctic = 0.0;
    for (int64_t i = 0; i < ref.nt; ++i) {
        for (double lon = -180.0; lon < 180.0; lon += 10.0) {
            arctic = std::max<double>(arctic, field.sample(ref.time_at(i), 85.0, lon));
            antarctic = std::max<double>(antarctic, field.sample(ref.time_at(i), -85.0, lon));
        }
    }
    EXPECT_LT(arctic, 1.0) << "the Arctic is lit in January";
    EXPECT_GT(antarctic, 100.0) << "the Antarctic is dark in January";
}

TEST(SolarPhysics, DaylightPeaksNearLocalNoon) {
    // Local solar noon at longitude L is 12:00 UTC minus L/15 hours. ERA5
    // stamps each accumulation at the *end* of the hour it covers, so the
    // extractor shifts the axis back half a step; skip that and the peak lands
    // consistently ~30 min late, which is exactly what this measures.
    const Reference ref{"diurnal_u16.npz"};
    ASSERT_GE(ref.nt, 24) << "fixture must span a full day";
    const NpzField field = NpzField::load(fixture("diurnal_u16.npz"));

    const double lat = ref.mid_lat(), lon = ref.mid_lon();
    const double local_noon_utc = 12.0 - lon / 15.0;

    // Sub-hourly, so the answer comes from the interpolation rather than from
    // whichever frame happens to be brightest.
    double best_hour = 0.0, best = -1.0;
    const int64_t span = (ref.nt - 1) * ref.dt;
    for (int64_t offset = 0; offset <= span; offset += 300) {
        const double w = field.sample(seconds{ref.t0 + offset}, lat, lon);
        ASSERT_FALSE(std::isnan(w));
        if (w > best) {
            best = w;
            best_hour = static_cast<double>(ref.t0 + offset) / 3600.0;
        }
    }
    ASSERT_GT(best, 50.0) << "no daylight in the window; nothing to locate";

    // Hour of day, UTC, of the brightest moment.
    const double peak_utc = std::fmod(std::fmod(best_hour, 24.0) + 24.0, 24.0);
    EXPECT_NEAR(peak_utc, local_noon_utc, 1.0)
        << "peak at " << peak_utc << "h UTC, local noon at " << local_noon_utc
        << "h for lon " << lon;
}

TEST(SolarPhysics, NightIsDarkAndDayIsNot) {
    const Reference ref{"diurnal_u16.npz"};
    const NpzField field = NpzField::load(fixture("diurnal_u16.npz"));
    const double lat = ref.mid_lat(), lon = ref.mid_lon();

    // Search the fixture's own frames rather than absolute clock times: the
    // window starts at whatever hour it starts at, and a target expressed in
    // UTC could land outside coverage entirely.
    const auto frame_nearest_local_hour = [&](double target) {
        int64_t best = 0;
        double best_gap = 1e9;
        for (int64_t i = 0; i < ref.nt; ++i) {
            const double local =
                std::fmod(static_cast<double>(ref.t0 + i * ref.dt) / 3600.0 + lon / 15.0,
                          24.0);
            double gap = std::abs(std::fmod(local - target + 36.0, 24.0) - 12.0);
            gap = 12.0 - gap;  // circular distance, 0..12 h
            if (gap < best_gap) {
                best_gap = gap;
                best = i;
            }
        }
        EXPECT_LT(best_gap, 1.5) << "no frame near local hour " << target;
        return ref.time_at(best);
    };

    EXPECT_LT(field.sample(frame_nearest_local_hour(0.0), lat, lon), 1.0)
        << "local midnight is not dark";
    EXPECT_GT(field.sample(frame_nearest_local_hour(12.0), lat, lon), 50.0)
        << "local noon is not lit";
}

}  // namespace
