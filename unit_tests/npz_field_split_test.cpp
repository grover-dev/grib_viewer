// Tests for the split form of boatforge::NpzField: a field spread over a
// directory of parts, loaded one part at a time.
//
// The parts are cut here at runtime from the same single-file fixture the rest
// of the suite uses, rather than checked in as a second set of fixtures. That
// makes the central property directly testable: a directory field must answer
// *identically* to the whole-cube field it was cut from, for every query. Any
// difference is a bug in the part selection, since nothing else about the
// sampler changed.

#include <npy_tools/npz_field.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

#include <npy_tools/npy.h>
#include <npy_tools/npz.h>
#include <gtest/gtest.h>

namespace {

using boatforge::NpyError;
using boatforge::NpzField;
using std::chrono::seconds;

std::filesystem::path fixture(const std::string& name) {
    return std::filesystem::path{BOATFORGE_FIXTURE_DIR} / name;
}

float sample(const NpzField& field, seconds when, double lat, double lon) {
    constexpr float nan = std::numeric_limits<float>::quiet_NaN();
    float value = nan;
    return field.sample(when, lat, lon, value) ? value : nan;
}

// A directory that cleans up after itself, so a failing test cannot leave parts
// behind for the next run to trip over.
class TempDir {
public:
    explicit TempDir(const std::string& name)
        : path_{std::filesystem::temp_directory_path() / ("npzfield_" + name)} {
        std::filesystem::remove_all(path_);
        std::filesystem::create_directories(path_);
    }
    ~TempDir() { std::filesystem::remove_all(path_); }

    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;

    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

// Cuts `source` into parts of `span` frames overlapping by `overlap`, the way
// scripts/grib_npz.py writes them, and returns the directory holding them.
// `overlap` is a parameter rather than a constant because the boundary
// behaviour with and without it is exactly what some of these tests are about.
void split_fixture(const std::string& source, const std::filesystem::path& dir,
                   std::size_t span, std::size_t overlap = 1) {
    const boatforge::NpzArchive npz = boatforge::load_npz(fixture(source));

    const auto scalar = [&npz](const std::string& name) {
        return npz.at(name).as<int64_t>()[0];
    };
    const int64_t t0 = scalar("t0"), dt = scalar("dt"), nt = scalar("nt");
    const int64_t nlat = scalar("nlat"), nlon = scalar("nlon");
    const auto frame = static_cast<std::size_t>(nlat * nlon);

    const boatforge::NpyArray& data = npz.at("data");
    const bool quantised = data.dtype() == boatforge::DType::UInt16;

    const std::size_t advance = span > overlap ? span - overlap : span;
    std::size_t index = 0;
    for (std::size_t lo = 0; lo < static_cast<std::size_t>(nt); lo += advance) {
        const std::size_t hi = std::min(lo + span, static_cast<std::size_t>(nt));
        const std::size_t count = hi - lo;

        char name[32];
        std::snprintf(name, sizeof(name), "part.%03zu.npz", index++);
        boatforge::NpzWriter out{dir / name};

        out.add_scalar("version", int32_t{1});
        out.add_scalar("t0", t0 + dt * static_cast<int64_t>(lo));
        out.add_scalar("dt", dt);
        out.add_scalar("nt", static_cast<int64_t>(count));
        for (const char* key : {"lat0", "dlat", "lon0", "dlon"}) {
            out.add_scalar(key, npz.at(key).as<double>()[0]);
        }
        out.add_scalar("nlat", nlat);
        out.add_scalar("nlon", nlon);
        out.add_scalar("lon_wrap", npz.at("lon_wrap").as<int32_t>()[0]);
        if (quantised) {
            out.add_scalar("scale", npz.at("scale").as<double>()[0]);
            out.add_scalar("offset", npz.at("offset").as<double>()[0]);
            out.add_scalar("fill", npz.at("fill").as<int64_t>()[0]);
        }

        const std::vector<std::size_t> shape{count, static_cast<std::size_t>(nlat),
                                             static_cast<std::size_t>(nlon)};
        if (quantised) {
            const auto values = data.as<uint16_t>();
            out.add("data",
                    std::vector<uint16_t>{values.begin() + static_cast<std::ptrdiff_t>(lo * frame),
                                          values.begin() + static_cast<std::ptrdiff_t>(hi * frame)},
                    shape);
        } else {
            const auto values = data.as<float>();
            out.add("data",
                    std::vector<float>{values.begin() + static_cast<std::ptrdiff_t>(lo * frame),
                                       values.begin() + static_cast<std::ptrdiff_t>(hi * frame)},
                    shape);
        }
        out.close();

        if (hi == static_cast<std::size_t>(nt)) {
            break;
        }
    }
}

// Every stamp on the time axis, plus the half-steps between them, so the checks
// below cover interpolation across a part boundary and not just the nodes.
std::vector<seconds> probe_times(const NpzField& whole, int64_t dt, int64_t nt) {
    const seconds t0 = whole.parts().front().t0;
    std::vector<seconds> times;
    for (int64_t i = 0; i < nt; ++i) {
        times.push_back(t0 + seconds{dt * i});
        if (i + 1 < nt) {
            times.push_back(t0 + seconds{dt * i + dt / 2});
        }
    }
    return times;
}

struct Grid {
    int64_t dt, nt;
    double lat, lon;
};

Grid grid_of(const std::string& name) {
    const boatforge::NpzArchive npz = boatforge::load_npz(fixture(name));
    // A point well inside coverage, so the comparison is about part selection
    // rather than about edge clamping.
    return Grid{npz.at("dt").as<int64_t>()[0], npz.at("nt").as<int64_t>()[0],
                npz.at("lat0").as<double>()[0] + npz.at("dlat").as<double>()[0] * 1.5,
                npz.at("lon0").as<double>()[0] + npz.at("dlon").as<double>()[0] * 1.5};
}

// --------------------------------------------------------------------------
// The central property: split and whole must be indistinguishable.
// --------------------------------------------------------------------------

TEST(NpzFieldSplit, AnswersIdenticallyToTheWholeCube) {
    const NpzField whole = NpzField::load(fixture("region_u16.npz"));
    const Grid grid = grid_of("region_u16.npz");

    // Several part sizes, including one that does not divide the axis evenly
    // and one small enough that most samples cross a boundary.
    for (const std::size_t span : {2u, 3u, 5u}) {
        SCOPED_TRACE(span);
        TempDir dir{"identical_" + std::to_string(span)};
        split_fixture("region_u16.npz", dir.path(), span);

        const NpzField split = NpzField::load_directory(dir.path());
        for (const seconds when : probe_times(whole, grid.dt, grid.nt)) {
            const float a = sample(whole, when, grid.lat, grid.lon);
            const float b = sample(split, when, grid.lat, grid.lon);
            ASSERT_EQ(std::isnan(a), std::isnan(b)) << when.count();
            if (!std::isnan(a)) {
                ASSERT_FLOAT_EQ(a, b) << when.count();
            }
        }
    }
}

TEST(NpzFieldSplit, Float32PartsAnswerIdenticallyToo) {
    const NpzField whole = NpzField::load(fixture("region_f32.npz"));
    const Grid grid = grid_of("region_f32.npz");

    TempDir dir{"identical_f32"};
    split_fixture("region_f32.npz", dir.path(), 3);

    const NpzField split = NpzField::load_directory(dir.path());
    for (const seconds when : probe_times(whole, grid.dt, grid.nt)) {
        const float a = sample(whole, when, grid.lat, grid.lon);
        const float b = sample(split, when, grid.lat, grid.lon);
        ASSERT_EQ(std::isnan(a), std::isnan(b)) << when.count();
        if (!std::isnan(a)) {
            ASSERT_FLOAT_EQ(a, b) << when.count();
        }
    }
}

// --------------------------------------------------------------------------
// Deferral and swapping: what the directory form exists to do.
// --------------------------------------------------------------------------

TEST(NpzFieldSplit, NothingIsResidentUntilTheFirstSample) {
    TempDir dir{"deferred"};
    split_fixture("region_u16.npz", dir.path(), 3);

    const NpzField split = NpzField::load_directory(dir.path(), 1);
    EXPECT_EQ(split.cached(), 0u);
    EXPECT_GT(split.parts().size(), 1u);
    EXPECT_EQ(split.resident(), NpzField::npos);

    const Grid grid = grid_of("region_u16.npz");
    sample(split, split.parts().front().t0, grid.lat, grid.lon);
    EXPECT_EQ(split.resident(), 0u);
}

TEST(NpzFieldSplit, ASingleFileFieldReportsOnePartAndKeepsItResident) {
    const NpzField field = NpzField::load(fixture("region_u16.npz"));
    ASSERT_EQ(field.parts().size(), 1u);
    EXPECT_EQ(field.resident(), 0u);
    EXPECT_EQ(field.parts().front().path, fixture("region_u16.npz"));
}

TEST(NpzFieldSplit, WalkingForwardSwapsPartsInOrderAndOnlyAtBoundaries) {
    TempDir dir{"walk"};
    split_fixture("region_u16.npz", dir.path(), 3);

    const NpzField split = NpzField::load_directory(dir.path(), 1);
    const Grid grid = grid_of("region_u16.npz");
    const seconds t0 = split.parts().front().t0;

    std::size_t swaps = 0, previous = NpzField::npos;
    for (int64_t i = 0; i < grid.nt; ++i) {
        sample(split, t0 + seconds{grid.dt * i}, grid.lat, grid.lon);
        if (split.resident() != previous) {
            // Never backwards: a forward walk must not revisit a part it left,
            // or the "one load per boundary" claim is false.
            EXPECT_TRUE(previous == NpzField::npos || split.resident() > previous);
            previous = split.resident();
            ++swaps;
        }
    }
    EXPECT_EQ(swaps, split.parts().size());
}

TEST(NpzFieldSplit, ResamplingInsideTheResidentPartLoadsNothing) {
    TempDir dir{"sticky"};
    split_fixture("region_u16.npz", dir.path(), 3);

    // One slot, so the parts deleted below cannot be answered from the cache.
    const NpzField split = NpzField::load_directory(dir.path(), 1);
    const Grid grid = grid_of("region_u16.npz");

    // Take the last part resident, then delete every file: further samples
    // inside that part must still work, which they only can if no load happens.
    // Addressed by its final frame, which is the one stamp no other part holds
    // -- at an instant inside the overlap the earlier part wins, by design.
    const auto& last = split.parts().back();
    sample(split, last.end, grid.lat, grid.lon);
    ASSERT_EQ(split.resident(), split.parts().size() - 1);

    for (const auto& part : split.parts()) {
        std::filesystem::remove(part.path);
    }

    EXPECT_FALSE(std::isnan(sample(split, last.end, grid.lat, grid.lon)));
    EXPECT_EQ(split.resident(), split.parts().size() - 1);

    // ...and stepping outside it now fails loudly rather than reading rubbish.
    EXPECT_THROW(sample(split, split.parts().front().t0, grid.lat, grid.lon), NpyError);
}

TEST(NpzFieldSplit, OutsideEveryPartIsAMissNotAThrow) {
    TempDir dir{"outside"};
    split_fixture("region_u16.npz", dir.path(), 3);

    const NpzField split = NpzField::load_directory(dir.path());
    const Grid grid = grid_of("region_u16.npz");

    EXPECT_TRUE(std::isnan(sample(split, split.parts().front().t0 - seconds{1}, grid.lat, grid.lon)));
    EXPECT_TRUE(std::isnan(sample(split, split.parts().back().end + seconds{1}, grid.lat, grid.lon)));
    // A miss must not have cost a load.
    EXPECT_EQ(split.resident(), NpzField::npos);
}

TEST(NpzFieldSplit, AGapBetweenPartsIsAMiss) {
    TempDir dir{"gap"};
    split_fixture("region_u16.npz", dir.path(), 3);

    NpzField split = NpzField::load_directory(dir.path());
    ASSERT_GE(split.parts().size(), 3u);

    // Drop a middle part and re-index: the window it covered is now a hole, and
    // a query inside it must miss rather than snap to a neighbouring part.
    const seconds hole = split.parts()[1].t0 + (split.parts()[1].end - split.parts()[1].t0) / 2;
    std::filesystem::remove(split.parts()[1].path);
    split = NpzField::load_directory(dir.path());

    const Grid grid = grid_of("region_u16.npz");
    // The overlap means the frames shared with the neighbours still resolve, so
    // probe strictly inside what only the removed part held.
    if (hole > split.parts().front().end && hole < split.parts()[1].t0) {
        EXPECT_TRUE(std::isnan(sample(split, hole, grid.lat, grid.lon)));
    }
}

// --------------------------------------------------------------------------
// Against the writer itself. Everything above cuts its own parts with
// NpzWriter, which mirrors what scripts/grib_npz.py does but is not it -- so
// on its own the suite would only prove this file agrees with itself. The
// region_split fixture is real output from the script's command line, manifest
// members and all.
// --------------------------------------------------------------------------

TEST(NpzFieldWriter, ReadsARealSplitFieldFromTheScript) {
    const auto dir = fixture("region_split");
    if (!std::filesystem::is_directory(dir)) {
        GTEST_SKIP() << "region_split not present; run make_fixtures.py";
    }

    const NpzField split = NpzField::load_directory(dir);
    EXPECT_GT(split.parts().size(), 1u);

    // The parts are the same field as the single-file fixture they were cut
    // from, so every query has to agree with it -- which is the actual contract
    // between the two languages.
    const NpzField whole = NpzField::load(fixture("region_u16.npz"));
    const Grid grid = grid_of("region_u16.npz");
    for (const seconds when : probe_times(whole, grid.dt, grid.nt)) {
        ASSERT_FLOAT_EQ(sample(whole, when, grid.lat, grid.lon),
                        sample(split, when, grid.lat, grid.lon))
            << when.count();
    }
}

TEST(NpzFieldWriter, ScriptPartsOverlapByOneFrame) {
    const auto dir = fixture("region_split");
    if (!std::filesystem::is_directory(dir)) {
        GTEST_SKIP() << "region_split not present; run make_fixtures.py";
    }

    const NpzField split = NpzField::load_directory(dir);
    const auto& parts = split.parts();
    ASSERT_GT(parts.size(), 1u);

    const seconds step{grid_of("region_u16.npz").dt};
    for (std::size_t i = 0; i + 1 < parts.size(); ++i) {
        // The next part starts on the frame this one ends with. That is what
        // makes the interval across a boundary interpolable, and it is a
        // property of what the script wrote, not of what this file assumed.
        EXPECT_EQ(parts[i + 1].t0, parts[i].end) << i;
        // ...and the two parts really do agree on the shared frame's value.
        const Grid grid = grid_of("region_u16.npz");
        const NpzField a = NpzField::load(parts[i].path);
        const NpzField b = NpzField::load(parts[i + 1].path);
        EXPECT_FLOAT_EQ(sample(a, parts[i].end, grid.lat, grid.lon),
                        sample(b, parts[i + 1].t0, grid.lat, grid.lon));
        EXPECT_LT(parts[i].t0, parts[i].end);
        EXPECT_EQ(parts[i].end - parts[i].t0,
                  step * static_cast<int64_t>(parts[i].nt - 1));
    }
}

TEST(NpzFieldWriter, ScriptPartsTileTheWholeWindow) {
    const auto dir = fixture("region_split");
    if (!std::filesystem::is_directory(dir)) {
        GTEST_SKIP() << "region_split not present; run make_fixtures.py";
    }

    const NpzField split = NpzField::load_directory(dir);
    const NpzField whole = NpzField::load(fixture("region_u16.npz"));
    const Grid grid = grid_of("region_u16.npz");

    // Same coverage as the unsplit field: no frame lost at a seam, and nothing
    // reachable past the end that was not reachable before.
    EXPECT_EQ(split.parts().front().t0, whole.parts().front().t0);
    EXPECT_EQ(split.parts().back().end, whole.parts().front().end);
    EXPECT_TRUE(std::isnan(
        sample(split, split.parts().back().end + seconds{1}, grid.lat, grid.lon)));
}

// --------------------------------------------------------------------------
// The cache: how many parts stay in memory, and what that buys.
// --------------------------------------------------------------------------

// Counts loads by watching resident() change is not enough -- a part answered
// from the cache also changes it. Files are made unreadable instead, so a load
// that does happen is unmistakable.
TEST(NpzFieldCache, HoldsUpToTheLimitAndNoMore) {
    TempDir dir{"limit"};
    split_fixture("region_u16.npz", dir.path(), 2);

    const NpzField split = NpzField::load_directory(dir.path(), 3);
    ASSERT_GT(split.parts().size(), 3u);
    EXPECT_EQ(split.cache_limit(), 3u);

    const Grid grid = grid_of("region_u16.npz");
    for (const auto& part : split.parts()) {
        sample(split, part.end, grid.lat, grid.lon);
        EXPECT_LE(split.cached(), 3u);
    }
    EXPECT_EQ(split.cached(), 3u);
}

// The case the depth exists for: two walks a few parts apart, stepped
// alternately the way Sim steps its runs round-robin. With one slot each
// alternation evicts what the other walk wants; with two, neither reloads.
TEST(NpzFieldCache, InterleavedWalksDoNotEvictEachOther) {
    TempDir dir{"interleaved"};
    split_fixture("region_u16.npz", dir.path(), 2);

    const NpzField split = NpzField::load_directory(dir.path(), 2);
    ASSERT_GE(split.parts().size(), 2u);
    const Grid grid = grid_of("region_u16.npz");

    // Both parts loaded, then every file removed: further alternation can only
    // be served from the cache.
    const seconds a = split.parts().front().t0;
    const seconds b = split.parts()[1].end;
    sample(split, a, grid.lat, grid.lon);
    sample(split, b, grid.lat, grid.lon);
    ASSERT_EQ(split.cached(), 2u);

    for (const auto& part : split.parts()) {
        std::filesystem::remove(part.path);
    }

    for (int i = 0; i < 8; ++i) {
        EXPECT_FALSE(std::isnan(sample(split, a, grid.lat, grid.lon)));
        EXPECT_FALSE(std::isnan(sample(split, b, grid.lat, grid.lon)));
    }
}

// The same alternation with one slot has to reload every time -- the property
// that makes the depth worth having.
TEST(NpzFieldCache, OneSlotReloadsOnEveryAlternation) {
    TempDir dir{"thrash"};
    split_fixture("region_u16.npz", dir.path(), 2);

    const NpzField split = NpzField::load_directory(dir.path(), 1);
    const Grid grid = grid_of("region_u16.npz");
    const seconds a = split.parts().front().t0;
    const seconds b = split.parts()[1].end;

    sample(split, a, grid.lat, grid.lon);
    sample(split, b, grid.lat, grid.lon);
    EXPECT_EQ(split.cached(), 1u);

    for (const auto& part : split.parts()) {
        std::filesystem::remove(part.path);
    }
    EXPECT_THROW(sample(split, a, grid.lat, grid.lon), NpyError);
}

// Eviction is by insertion order, and insertion order is time order, so the
// slot reused is always the one furthest behind the walk.
TEST(NpzFieldCache, EvictsInLoadOrder) {
    TempDir dir{"ring"};
    split_fixture("region_u16.npz", dir.path(), 2);

    const NpzField split = NpzField::load_directory(dir.path(), 2);
    ASSERT_GE(split.parts().size(), 3u);
    const Grid grid = grid_of("region_u16.npz");

    // Load parts 0 and 1, then 2 -- which must take part 0's slot.
    sample(split, split.parts()[0].t0, grid.lat, grid.lon);
    sample(split, split.parts()[1].end, grid.lat, grid.lon);
    sample(split, split.parts()[2].end, grid.lat, grid.lon);
    ASSERT_EQ(split.cached(), 2u);

    std::filesystem::remove(split.parts()[0].path);
    // 1 and 2 are still cached...
    EXPECT_FALSE(std::isnan(sample(split, split.parts()[1].end, grid.lat, grid.lon)));
    EXPECT_FALSE(std::isnan(sample(split, split.parts()[2].end, grid.lat, grid.lon)));
    // ...and 0 is the one that was dropped.
    EXPECT_THROW(sample(split, split.parts()[0].t0, grid.lat, grid.lon), NpyError);
}

TEST(NpzFieldCache, DepthDoesNotChangeAnyAnswer) {
    const NpzField whole = NpzField::load(fixture("region_u16.npz"));
    const Grid grid = grid_of("region_u16.npz");

    TempDir dir{"depths"};
    split_fixture("region_u16.npz", dir.path(), 2);

    for (const std::size_t depth : {1u, 2u, 4u, 99u}) {
        SCOPED_TRACE(depth);
        const NpzField split = NpzField::load_directory(dir.path(), depth);
        for (const seconds when : probe_times(whole, grid.dt, grid.nt)) {
            ASSERT_FLOAT_EQ(sample(whole, when, grid.lat, grid.lon),
                            sample(split, when, grid.lat, grid.lon))
                << when.count();
        }
    }
}

TEST(NpzFieldCache, IsCappedAtThePartCount) {
    TempDir dir{"oversized"};
    split_fixture("region_u16.npz", dir.path(), 3);

    const NpzField split = NpzField::load_directory(dir.path(), 1000);
    EXPECT_EQ(split.cache_limit(), split.parts().size());
}

TEST(NpzFieldCache, ZeroIsRejected) {
    TempDir dir{"zero"};
    split_fixture("region_u16.npz", dir.path(), 3);
    EXPECT_THROW(NpzField::load_directory(dir.path(), 0), NpyError);
}

// --------------------------------------------------------------------------
// A directory has to be one field, or sampling it means different things at
// different times.
// --------------------------------------------------------------------------

TEST(NpzFieldSplit, PartsOnDifferentGridsAreRejected) {
    TempDir dir{"mixed"};
    split_fixture("region_u16.npz", dir.path(), 3);
    std::filesystem::copy_file(fixture("global_u16.npz"), dir.path() / "part.999.npz");

    EXPECT_THROW(NpzField::load_directory(dir.path()), NpyError);
}

TEST(NpzFieldSplit, AnEmptyDirectoryThrows) {
    TempDir dir{"empty"};
    EXPECT_THROW(NpzField::load_directory(dir.path()), NpyError);
}

TEST(NpzFieldSplit, ADirectoryHoldingSomethingElseThrows) {
    TempDir dir{"notafield"};
    // A real npz, but a route rather than a field.
    boatforge::NpzWriter out{dir.path() / "track.npz"};
    out.add("lat", std::vector<double>{1.0, 2.0});
    out.close();

    EXPECT_THROW(NpzField::load_directory(dir.path()), NpyError);
}

TEST(NpzFieldSplit, LoadDirectoryOnAFileThrows) {
    EXPECT_THROW(NpzField::load_directory(fixture("region_u16.npz")), NpyError);
}

// A directory of one part is still a directory, and must behave like the
// single-file load of the same data.
TEST(NpzFieldSplit, ASinglePartDirectoryMatchesTheWholeFile) {
    const boatforge::NpzArchive npz = boatforge::load_npz(fixture("region_u16.npz"));
    const auto nt = static_cast<std::size_t>(npz.at("nt").as<int64_t>()[0]);

    TempDir dir{"onepart"};
    split_fixture("region_u16.npz", dir.path(), nt);

    const NpzField split = NpzField::load_directory(dir.path());
    ASSERT_EQ(split.parts().size(), 1u);

    const NpzField whole = NpzField::load(fixture("region_u16.npz"));
    const Grid grid = grid_of("region_u16.npz");
    for (const seconds when : probe_times(whole, grid.dt, grid.nt)) {
        ASSERT_FLOAT_EQ(sample(whole, when, grid.lat, grid.lon),
                        sample(split, when, grid.lat, grid.lon));
    }
}

}  // namespace
