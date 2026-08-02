// Tests for the .npz (zip-of-npy) reader.
//
// These run against the fixtures in unit_tests/fixtures rather than anything
// generated here. They are real numpy savez output, which is the only thing
// that pins down the parts of the format a hand-rolled zip would not exercise:
// ZIP64 end-of-directory records (savez emits them unconditionally), deflated
// members, and 0-d scalars stored as their own .npy files.
//
// Nothing below depends on the fixtures' *values*, only on their structure --
// the member names, dtypes and shapes that make them a gridded field. Those
// are asserted against each other where possible (data.size == nt*nlat*nlon)
// so regenerating the fixtures from different weather does not break anything.

#include <npy_tools/npz.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include <npy_tools/npy.h>
#include <gtest/gtest.h>

namespace {

using boatforge::DType;
using boatforge::load_npz;
using boatforge::NpyError;
using boatforge::NpzArchive;

std::filesystem::path fixture(std::string_view name) {
    return std::filesystem::path{NPY_TOOLS_FIXTURE_DIR} / name;
}

// std::map::at has no heterogeneous overload, so a string_view key has to go
// through find() -- which is the lookup being exercised anyway.
const boatforge::NpyArray& member(const NpzArchive& npz, std::string_view name) {
    const auto it = npz.find(name);
    EXPECT_NE(it, npz.end()) << "missing member '" << name << "'";
    static const boatforge::NpyArray kAbsent;
    return it == npz.end() ? kAbsent : it->second;
}

// Every fixture is one of these; the reader should not care which.
constexpr std::string_view kFixtures[]{
    "region_f32.npz", "region_u16.npz", "global_u16.npz",
    "diurnal_u16.npz", "signed_f32.npz",
};

TEST(LoadNpz, KeysDropTheNpySuffix) {
    const NpzArchive npz = load_npz(fixture("region_f32.npz"));

    // What numpy.load() exposes as `.files` -- no extensions, no directory
    // components.
    for (const auto& [name, array] : npz) {
        EXPECT_FALSE(name.ends_with(".npy")) << name;
        EXPECT_EQ(name.find('/'), std::string::npos) << name;
    }
    EXPECT_TRUE(npz.contains("data"));
    EXPECT_TRUE(npz.contains("lat"));
    EXPECT_FALSE(npz.contains("data.npy"));
}

TEST(LoadNpz, ReadsEveryMemberOfEveryFixture) {
    for (const std::string_view name : kFixtures) {
        const NpzArchive npz = load_npz(fixture(name));
        SCOPED_TRACE(name);

        for (const std::string_view key :
             {"version", "t0", "dt", "nt", "lat0", "dlat", "nlat", "lon0",
              "dlon", "nlon", "data", "time", "lat", "lon"}) {
            EXPECT_TRUE(npz.contains(key)) << key;
        }
    }
}

// The scalars are 0-d .npy members. They are the case most likely to be
// mishandled -- an empty shape tuple that still carries one element -- and
// every fixture has thirteen of them.
TEST(LoadNpz, ZeroDimensionalScalarsCarryExactlyOneElement) {
    const NpzArchive npz = load_npz(fixture("region_u16.npz"));

    for (const std::string_view key : {"t0", "dt", "nt", "nlat", "nlon"}) {
        const auto& array = member(npz, key);
        SCOPED_TRACE(key);
        EXPECT_TRUE(array.shape().empty());
        EXPECT_EQ(array.size(), 1u);
        EXPECT_EQ(array.shape_string(), "()");
        EXPECT_EQ(array.dtype(), DType::Int64);
        EXPECT_EQ(array.bytes().size(), 8u);
    }
    EXPECT_EQ(npz.at("version").dtype(), DType::Int32);
    EXPECT_EQ(npz.at("lat0").dtype(), DType::Float64);
}

// Shapes and payload sizes have to agree with the axis lengths stored beside
// them, which is the end-to-end check that headers were parsed against the
// right member and no payload was truncated on the way out of the zip.
TEST(LoadNpz, ShapesAgreeWithTheStoredAxisLengths) {
    for (const std::string_view name : kFixtures) {
        const NpzArchive npz = load_npz(fixture(name));
        SCOPED_TRACE(name);

        const auto nt = static_cast<std::size_t>(npz.at("nt").as<std::int64_t>()[0]);
        const auto nlat = static_cast<std::size_t>(npz.at("nlat").as<std::int64_t>()[0]);
        const auto nlon = static_cast<std::size_t>(npz.at("nlon").as<std::int64_t>()[0]);

        const auto& data = npz.at("data");
        EXPECT_EQ(data.shape(), (std::vector<std::size_t>{nt, nlat, nlon}));
        EXPECT_EQ(data.size(), nt * nlat * nlon);
        EXPECT_EQ(data.bytes().size(), data.size() * word_size(data.dtype()));
        EXPECT_FALSE(data.fortran_order());

        EXPECT_EQ(npz.at("time").shape(), (std::vector<std::size_t>{nt}));
        EXPECT_EQ(npz.at("lat").shape(), (std::vector<std::size_t>{nlat}));
        EXPECT_EQ(npz.at("lon").shape(), (std::vector<std::size_t>{nlon}));
    }
}

// global_u16 is ~100k elements, big enough that libzip hands the payload back
// over several reads. A short-read bug shows up here and nowhere else.
TEST(LoadNpz, LargeDeflatedMembersAreReadInFull) {
    const NpzArchive npz = load_npz(fixture("global_u16.npz"));
    const auto& data = npz.at("data");

    ASSERT_EQ(data.dtype(), DType::UInt16);
    ASSERT_GT(data.size(), 10000u);
    EXPECT_EQ(data.as<std::uint16_t>().size(), data.size());
    EXPECT_EQ(data.bytes().size(), data.size() * 2);
}

TEST(LoadNpz, BothFloatAndQuantisedPayloadsRoundTrip) {
    EXPECT_EQ(load_npz(fixture("region_f32.npz")).at("data").dtype(),
              DType::Float32);
    EXPECT_EQ(load_npz(fixture("region_u16.npz")).at("data").dtype(),
              DType::UInt16);
}

// std::less<> on the map, so a lookup does not have to allocate a std::string
// just to compare.
TEST(LoadNpz, LookupAcceptsStringView) {
    const NpzArchive npz = load_npz(fixture("region_f32.npz"));

    const std::string_view key = "data";
    ASSERT_NE(npz.find(key), npz.end());
    EXPECT_EQ(npz.find(key)->first, "data");
    EXPECT_EQ(npz.find(std::string_view{"nope"}), npz.end());
}

TEST(LoadNpz, ThrowsOnAMissingFile) {
    EXPECT_THROW(load_npz(fixture("no_such_fixture.npz")), NpyError);

    // The path is in the message; a bare "cannot open" is useless when the
    // caller assembled the path from a config file.
    try {
        load_npz(fixture("no_such_fixture.npz"));
        FAIL() << "expected NpyError";
    } catch (const NpyError& e) {
        EXPECT_NE(std::string{e.what()}.find("no_such_fixture.npz"),
                  std::string::npos)
            << e.what();
    }
}

TEST(LoadNpz, ThrowsOnAFileThatIsNotAZip) {
    // make_fixtures.py is right there and is definitively not a zip archive.
    EXPECT_THROW(load_npz(fixture("make_fixtures.py")), NpyError);
}

TEST(LoadNpz, ThrowsOnADirectory) {
    EXPECT_THROW(load_npz(std::filesystem::path{NPY_TOOLS_FIXTURE_DIR}), NpyError);
}

}  // namespace
