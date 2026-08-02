// Tests for the .npy / .npz writers and for NpzRecorder.
//
// Most of these round-trip through the reader, which is a weaker check than it
// looks -- a matched pair of bugs would cancel out. So the byte-level details
// numpy actually depends on (magic, version, 64-byte payload alignment, descr
// spelling) are asserted directly, and scripts/npz_roundtrip.py loads the same
// output with numpy itself for the check no self-consistent pair can pass.

#include <npy_tools/npz.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include <npy_tools/npy.h>
#include <npy_tools/npz_recorder.h>
#include <gtest/gtest.h>

namespace {

using boatforge::Compression;
using boatforge::DType;
using boatforge::load_npz;
using boatforge::NpyArray;
using boatforge::NpyError;
using boatforge::NpzArchive;
using boatforge::NpzRecorder;
using boatforge::NpzWriter;
using boatforge::parse_npy;
using boatforge::write_npy;
using boatforge::write_npy_scalar;

// A path under the test binary's temp dir, removed when the test ends. Writing
// to a fixed name would make the suite order-dependent under ctest's parallel
// runs.
class TempNpz {
public:
    explicit TempNpz(std::string_view stem)
        : path_{std::filesystem::temp_directory_path() /
                std::format("npy_tools_{}_{}.npz", stem,
                            ::testing::UnitTest::GetInstance()
                                ->current_test_info()
                                ->name())} {
        std::filesystem::remove(path_);
    }
    ~TempNpz() { std::filesystem::remove(path_); }

    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

std::string_view header_of(const std::vector<std::byte>& image) {
    const auto len = static_cast<std::size_t>(image[8]) |
                     (static_cast<std::size_t>(image[9]) << 8);
    return std::string_view{reinterpret_cast<const char*>(image.data()) + 10, len};
}

TEST(WriteNpy, EmitsTheHeaderNumpyExpects) {
    const std::vector<double> values{1.0, 2.0, 3.0};
    const auto image = write_npy(values);

    ASSERT_GE(image.size(), 10u);
    EXPECT_EQ(image[0], std::byte{0x93});
    EXPECT_EQ(std::string_view(reinterpret_cast<const char*>(image.data()) + 1, 5),
              "NUMPY");
    EXPECT_EQ(image[6], std::byte{1});  // version 1.0 while the header fits
    EXPECT_EQ(image[7], std::byte{0});

    const std::string_view header = header_of(image);
    EXPECT_TRUE(header.starts_with(
        "{'descr': '<f8', 'fortran_order': False, 'shape': (3,), }"))
        << header;
    EXPECT_TRUE(header.ends_with("\n")) << header;

    // The payload has to start on a 64-byte boundary; numpy's own reader is
    // relaxed about it but np.lib.format.write_array is not, and mmap of a
    // misaligned array is what the rule exists for.
    EXPECT_EQ((10 + header.size()) % 64, 0u) << header.size();
    EXPECT_EQ(image.size(), 10 + header.size() + 3 * sizeof(double));
}

TEST(WriteNpy, RoundTripsEverySupportedDType) {
    // std::array, not std::vector: vector<bool> is not a contiguous range, so
    // it is the one standard container write_npy cannot take.
    const auto check = [](auto value, DType expected, std::string_view descr) {
        using T = decltype(value);
        const std::array<T, 2> values{value, value};
        const auto image = write_npy(values);

        EXPECT_NE(header_of(image).find(descr), std::string_view::npos)
            << header_of(image);

        const NpyArray array = parse_npy(image);
        ASSERT_EQ(array.dtype(), expected);
        ASSERT_EQ(array.size(), 2u);
        EXPECT_EQ(array.template as<T>()[1], value);
    };

    check(true, DType::Bool, "|b1");
    check(std::int8_t{-128}, DType::Int8, "|i1");
    check(std::int16_t{-32768}, DType::Int16, "<i2");
    check(std::int32_t{-2000000000}, DType::Int32, "<i4");
    check(std::int64_t{-1LL << 62}, DType::Int64, "<i8");
    check(std::uint8_t{255}, DType::UInt8, "|u1");
    check(std::uint16_t{65535}, DType::UInt16, "<u2");
    check(std::uint32_t{4294967295u}, DType::UInt32, "<u4");
    check(std::uint64_t{~0ULL}, DType::UInt64, "<u8");
    check(0.5f, DType::Float32, "<f4");
    check(1e300, DType::Float64, "<f8");
}

TEST(WriteNpy, WritesShapesAndFortranOrder) {
    const std::vector<std::int32_t> values{0, 1, 2, 3, 4, 5};

    const NpyArray flat = parse_npy(write_npy(values));
    EXPECT_EQ(flat.shape(), (std::vector<std::size_t>{6}));

    const NpyArray grid = parse_npy(write_npy(values, {2, 3}));
    EXPECT_EQ(grid.shape(), (std::vector<std::size_t>{2, 3}));
    EXPECT_EQ(grid.shape_string(), "(2, 3)");
    EXPECT_FALSE(grid.fortran_order());

    const NpyArray column_major = parse_npy(write_npy(values, {2, 3}, true));
    EXPECT_TRUE(column_major.fortran_order());
    EXPECT_NE(header_of(write_npy(values, {2, 3}, true)).find("True"),
              std::string_view::npos);
}

TEST(WriteNpy, WritesZeroDimensionalScalars) {
    const NpyArray array = parse_npy(write_npy_scalar(7.25));

    EXPECT_TRUE(array.shape().empty());
    EXPECT_EQ(array.size(), 1u);
    EXPECT_EQ(array.shape_string(), "()");
    EXPECT_DOUBLE_EQ(array.as<double>()[0], 7.25);
    EXPECT_NE(header_of(write_npy_scalar(7.25)).find("'shape': ()"),
              std::string_view::npos);
}

TEST(WriteNpy, WritesEmptyArrays) {
    const std::vector<float> none;
    const NpyArray array = parse_npy(write_npy(none));

    EXPECT_EQ(array.size(), 0u);
    EXPECT_EQ(array.shape(), (std::vector<std::size_t>{0}));
    EXPECT_TRUE(array.bytes().empty());
}

TEST(WriteNpy, ReSerializesAParsedArray) {
    const std::vector<double> values{1.5, -2.5, 3.5};
    const NpyArray original = parse_npy(write_npy(values, {3, 1}));
    const NpyArray again = parse_npy(write_npy(original));

    EXPECT_EQ(again.shape(), original.shape());
    EXPECT_EQ(again.dtype(), original.dtype());
    EXPECT_EQ(again.fortran_order(), original.fortran_order());
    EXPECT_TRUE(std::ranges::equal(again.bytes(), original.bytes()));
}

TEST(WriteNpy, RejectsAPayloadThatDoesNotMatchTheShape) {
    const std::vector<double> values{1.0, 2.0, 3.0};
    EXPECT_THROW((void)write_npy(values, {2, 3}), NpyError);
    EXPECT_THROW((void)write_npy(values, {}), NpyError);
    EXPECT_NO_THROW((void)write_npy(values, {3}));

    // The message says what was expected, since the shape is usually the part
    // that is wrong.
    try {
        (void)write_npy(values, {4});
        FAIL() << "expected NpyError";
    } catch (const NpyError& e) {
        const std::string what = e.what();
        EXPECT_NE(what.find("(4,)"), std::string::npos) << what;
        EXPECT_NE(what.find("float64"), std::string::npos) << what;
    }
}

TEST(WriteNpz, RoundTripsThroughLoadNpz) {
    const TempNpz out{"roundtrip"};

    const std::vector<double> lat{51.5, 51.6, 51.7};
    const std::vector<float> power{100.0f, 250.5f, 0.0f};
    const std::vector<std::int64_t> time{1735689600, 1735693200, 1735696800};
    {
        NpzWriter writer{out.path()};
        writer.add("lat", lat);
        writer.add("solar_power_in_w", power);
        writer.add("time", time);
        writer.add_scalar("nt", std::int64_t{3});
        writer.close();
    }

    const NpzArchive npz = load_npz(out.path());
    ASSERT_EQ(npz.size(), 4u);

    EXPECT_EQ(npz.at("lat").dtype(), DType::Float64);
    EXPECT_TRUE(std::ranges::equal(npz.at("lat").as<double>(), lat));
    EXPECT_EQ(npz.at("solar_power_in_w").dtype(), DType::Float32);
    EXPECT_TRUE(std::ranges::equal(npz.at("solar_power_in_w").as<float>(), power));
    EXPECT_TRUE(std::ranges::equal(npz.at("time").as<std::int64_t>(), time));

    EXPECT_TRUE(npz.at("nt").shape().empty());
    EXPECT_EQ(npz.at("nt").as<std::int64_t>()[0], 3);
}

TEST(WriteNpz, MembersGetTheNpySuffixOnDisk) {
    const TempNpz out{"suffix"};
    {
        NpzWriter writer{out.path()};
        writer.add("data", std::vector<double>{1.0});
        writer.close();
    }

    // Keyed without the suffix on the way back in, exactly like savez output.
    const NpzArchive npz = load_npz(out.path());
    EXPECT_TRUE(npz.contains("data"));
    EXPECT_FALSE(npz.contains("data.npy"));
}

TEST(WriteNpz, DeflateAndStoreBothReadBackIdentically) {
    const std::vector<double> values(4096, 3.25);  // compresses to nearly nothing

    const TempNpz stored{"stored"};
    const TempNpz deflated{"deflated"};
    {
        NpzWriter a{stored.path(), Compression::Store};
        a.add("data", values);
        a.close();

        NpzWriter b{deflated.path(), Compression::Deflate};
        b.add("data", values);
        b.close();
    }

    EXPECT_LT(std::filesystem::file_size(deflated.path()),
              std::filesystem::file_size(stored.path()));
    EXPECT_TRUE(std::ranges::equal(load_npz(stored.path()).at("data").as<double>(),
                                   load_npz(deflated.path()).at("data").as<double>()));
}

TEST(WriteNpz, LargeArraysSurviveTheTrip) {
    const TempNpz out{"large"};
    std::vector<std::uint16_t> data(200000);
    for (std::size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<std::uint16_t>(i);
    }
    {
        NpzWriter writer{out.path(), Compression::Deflate};
        writer.add("data", data, {200, 1000});
        writer.close();
    }

    const NpzArchive npz = load_npz(out.path());
    const NpyArray& array = npz.at("data");
    EXPECT_EQ(array.shape(), (std::vector<std::size_t>{200, 1000}));
    EXPECT_TRUE(std::ranges::equal(array.as<std::uint16_t>(), data));
}

TEST(WriteNpz, SaveNpzIsTheInverseOfLoadNpz) {
    const TempNpz out{"inverse"};
    // A real fixture, so this covers 0-d scalars, i4/i8/f8 and a 3-d cube in
    // one pass -- and proves the two directions agree on all of them.
    const NpzArchive original =
        load_npz(std::filesystem::path{NPY_TOOLS_FIXTURE_DIR} / "region_u16.npz");

    boatforge::save_npz(out.path(), original, Compression::Deflate);
    const NpzArchive reloaded = load_npz(out.path());

    ASSERT_EQ(reloaded.size(), original.size());
    for (const auto& [name, array] : original) {
        SCOPED_TRACE(name);
        const NpyArray& copy = reloaded.at(name);
        EXPECT_EQ(copy.dtype(), array.dtype());
        EXPECT_EQ(copy.shape(), array.shape());
        EXPECT_EQ(copy.fortran_order(), array.fortran_order());
        EXPECT_TRUE(std::ranges::equal(copy.bytes(), array.bytes()));
    }
}

TEST(WriteNpz, NothingIsWrittenUntilClose) {
    const TempNpz out{"deferred"};
    {
        NpzWriter writer{out.path()};
        writer.add("data", std::vector<double>{1.0});
        EXPECT_FALSE(std::filesystem::exists(out.path()));
        writer.close();
    }
    EXPECT_TRUE(std::filesystem::exists(out.path()));
}

TEST(WriteNpz, TheDestructorCommits) {
    const TempNpz out{"destructor"};
    {
        NpzWriter writer{out.path()};
        writer.add("data", std::vector<double>{2.0});
    }  // no explicit close()
    EXPECT_DOUBLE_EQ(load_npz(out.path()).at("data").as<double>()[0], 2.0);
}

TEST(WriteNpz, CloseIsIdempotent) {
    const TempNpz out{"idempotent"};
    NpzWriter writer{out.path()};
    writer.add("data", std::vector<double>{1.0});
    writer.close();
    EXPECT_NO_THROW(writer.close());  // and again from the destructor
    EXPECT_EQ(load_npz(out.path()).size(), 1u);
}

TEST(WriteNpz, RejectsDuplicateAndEmptyNames) {
    const TempNpz out{"names"};
    NpzWriter writer{out.path()};

    writer.add("data", std::vector<double>{1.0});
    EXPECT_THROW(writer.add("data", std::vector<float>{2.0f}), NpyError);
    EXPECT_THROW(writer.add("", std::vector<double>{1.0}), NpyError);
    writer.close();
}

TEST(WriteNpz, RejectsAddAfterClose) {
    const TempNpz out{"closed"};
    NpzWriter writer{out.path()};
    writer.close();
    EXPECT_THROW(writer.add("late", std::vector<double>{1.0}), NpyError);
}

TEST(WriteNpz, ThrowsOnAnUnwritablePath) {
    NpzWriter writer{"/proc/definitely/not/writable.npz"};
    writer.add("data", std::vector<double>{1.0});
    EXPECT_THROW(writer.close(), NpyError);
}

TEST(Recorder, WritesOneColumnPerFieldPerStep) {
    const TempNpz out{"recorder"};

    NpzRecorder log;
    for (int step = 0; step < 5; ++step) {
        log.record("time", std::chrono::seconds{1735689600 + step * 3600});
        log.record("lat", 51.5 + step * 0.1);
        log.record("solar_power_in_w", 100.0f * static_cast<float>(step));
        log.record("steps", static_cast<std::uint32_t>(step));
    }
    EXPECT_EQ(log.columns(), 4u);
    EXPECT_EQ(log.rows(), 5u);
    EXPECT_EQ(log.rows("lat"), 5u);
    EXPECT_EQ(log.rows("nonexistent"), 0u);

    log.save(out.path());
    const NpzArchive npz = load_npz(out.path());
    ASSERT_EQ(npz.size(), 4u);

    // Each field keeps the width it was recorded at -- a float column stays
    // float32 rather than being widened on the way out.
    EXPECT_EQ(npz.at("time").dtype(), DType::Int64);
    EXPECT_EQ(npz.at("lat").dtype(), DType::Float64);
    EXPECT_EQ(npz.at("solar_power_in_w").dtype(), DType::Float32);
    EXPECT_EQ(npz.at("steps").dtype(), DType::UInt32);

    for (const auto& [name, array] : npz) {
        SCOPED_TRACE(name);
        EXPECT_EQ(array.shape(), (std::vector<std::size_t>{5}));
    }
    EXPECT_EQ(npz.at("time").as<std::int64_t>()[0], 1735689600);
    EXPECT_EQ(npz.at("time").as<std::int64_t>()[4], 1735689600 + 4 * 3600);
    EXPECT_DOUBLE_EQ(npz.at("lat").as<double>()[2], 51.7);
    EXPECT_FLOAT_EQ(npz.at("solar_power_in_w").as<float>()[3], 300.0f);
    EXPECT_EQ(npz.at("steps").as<std::uint32_t>()[4], 4u);
}

TEST(Recorder, DurationsAreRecordedAsWholeSeconds) {
    const TempNpz out{"durations"};

    NpzRecorder log;
    log.record("t", std::chrono::hours{2});
    log.record("t", std::chrono::minutes{90});
    log.record("t", std::chrono::milliseconds{1500});  // truncates, as documented
    log.save(out.path());

    const auto values = load_npz(out.path()).at("t").as<std::int64_t>();
    EXPECT_EQ(values[0], 7200);
    EXPECT_EQ(values[1], 5400);
    EXPECT_EQ(values[2], 1);
}

TEST(Recorder, ScalarsAreWrittenOnceAsZeroDimensionalArrays) {
    const TempNpz out{"scalars"};

    NpzRecorder log;
    log.set_scalar("surface_area_m", 1.0f);
    log.set_scalar("time_step", std::chrono::hours{1});
    log.set_scalar("surface_area_m", 2.0f);  // last one wins
    log.record("lat", 51.5);
    log.save(out.path());

    const NpzArchive npz = load_npz(out.path());
    EXPECT_TRUE(npz.at("surface_area_m").shape().empty());
    EXPECT_FLOAT_EQ(npz.at("surface_area_m").as<float>()[0], 2.0f);
    EXPECT_EQ(npz.at("time_step").as<std::int64_t>()[0], 3600);
    // Scalars are not columns, so they do not have to match the row count.
    EXPECT_EQ(npz.at("lat").shape(), (std::vector<std::size_t>{1}));
    EXPECT_EQ(log.columns(), 1u);
}

TEST(Recorder, RejectsAColumnThatChangesType) {
    NpzRecorder log;
    log.record("power", 1.0f);
    EXPECT_THROW(log.record("power", 1.0), NpyError);
    EXPECT_THROW(log.record("power", std::int32_t{1}), NpyError);
    EXPECT_NO_THROW(log.record("power", 2.0f));

    try {
        log.record("power", 1.0);
        FAIL() << "expected NpyError";
    } catch (const NpyError& e) {
        const std::string what = e.what();
        EXPECT_NE(what.find("float32"), std::string::npos) << what;
        EXPECT_NE(what.find("float64"), std::string::npos) << what;
    }
}

TEST(Recorder, RejectsANameUsedAsBothColumnAndScalar) {
    NpzRecorder log;
    log.record("lat", 51.5);
    EXPECT_THROW(log.set_scalar("lat", 51.5), NpyError);

    log.set_scalar("area", 1.0f);
    EXPECT_THROW(log.record("area", 1.0f), NpyError);
}

// The failure this exists to catch: a field recorded inside a branch, so the
// columns silently stop lining up and every plot after it is wrong.
TEST(Recorder, RejectsRaggedColumnsOnSave) {
    const TempNpz out{"ragged"};

    NpzRecorder log;
    for (int step = 0; step < 4; ++step) {
        log.record("lat", 51.5);
        if (step % 2 == 0) {
            log.record("only_sometimes", 1.0);
        }
    }
    EXPECT_EQ(log.rows(), 4u);
    EXPECT_EQ(log.rows("only_sometimes"), 2u);
    EXPECT_THROW(log.save(out.path()), NpyError);
    EXPECT_FALSE(std::filesystem::exists(out.path()));
}

TEST(Recorder, ClearResetsEverything) {
    const TempNpz out{"cleared"};

    NpzRecorder log;
    log.record("lat", 51.5);
    log.set_scalar("area", 1.0f);
    log.clear();
    EXPECT_EQ(log.columns(), 0u);
    EXPECT_EQ(log.rows(), 0u);

    // Including the recorded types, so a column can come back as something else.
    log.record("lat", 1.0f);
    log.save(out.path());
    const NpzArchive npz = load_npz(out.path());
    EXPECT_EQ(npz.size(), 1u);
    EXPECT_EQ(npz.at("lat").dtype(), DType::Float32);
}

TEST(Recorder, AnEmptyRecorderWritesAnEmptyArchive) {
    const TempNpz out{"empty"};

    NpzRecorder log;
    log.save(out.path());
    EXPECT_TRUE(std::filesystem::exists(out.path()));
    EXPECT_TRUE(load_npz(out.path()).empty());
}

}  // namespace
