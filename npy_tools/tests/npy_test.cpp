// Tests for the .npy parser.
//
// The images here are assembled byte by byte rather than read from fixtures.
// That is deliberate: parse_npy's job is mostly to reject things, and a file
// numpy is willing to write is by definition not one of them. Building the
// header inline is the only way to reach the truncation, bad-magic, wrong-
// endianness and unsupported-dtype paths, and it keeps the expected bytes
// visible next to the assertion that checks them.

#include <npy_tools/npy.h>

#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

namespace {

using boatforge::DType;
using boatforge::NpyArray;
using boatforge::NpyError;
using boatforge::parse_npy;

// A header dict as numpy writes it. `descr` and `shape` go in verbatim so a
// test can pass something malformed; the quotes around descr are the caller's.
std::string header_dict(std::string_view descr, std::string_view shape,
                        bool fortran) {
    return std::string{"{'descr': "} + std::string{descr} +
           ", 'fortran_order': " + (fortran ? "True" : "False") +
           ", 'shape': " + std::string{shape} + ", }";
}

template <typename T>
std::vector<std::byte> raw(std::initializer_list<T> values) {
    std::vector<std::byte> out(values.size() * sizeof(T));
    std::size_t i = 0;
    for (const T value : values) {
        std::memcpy(out.data() + i++ * sizeof(T), &value, sizeof(T));
    }
    return out;
}

// magic + version + header length + padded header + payload. numpy pads the
// header with spaces and a newline so the payload starts on a 64-byte
// boundary; matching that also means the parser's whitespace trimming is
// exercised on every test that goes through here.
std::vector<std::byte> npy_image(std::string_view dict,
                                 std::span<const std::byte> payload,
                                 int major = 1) {
    const std::size_t prefix = major == 1 ? 10 : 12;
    std::string header{dict};
    while ((prefix + header.size() + 1) % 64 != 0) {
        header.push_back(' ');
    }
    header.push_back('\n');

    std::vector<std::byte> image;
    for (const char c : std::string_view{"\x93NUMPY"}) {
        image.push_back(std::byte(c));
    }
    image.push_back(std::byte(major));
    image.push_back(std::byte{0});

    if (major == 1) {
        const auto len = static_cast<std::uint16_t>(header.size());
        image.push_back(std::byte(len & 0xff));
        image.push_back(std::byte((len >> 8) & 0xff));
    } else {
        const auto len = static_cast<std::uint32_t>(header.size());
        for (int shift = 0; shift < 32; shift += 8) {
            image.push_back(std::byte((len >> shift) & 0xff));
        }
    }

    for (const char c : header) {
        image.push_back(std::byte(c));
    }
    image.insert(image.end(), payload.begin(), payload.end());
    return image;
}

// The common case: a well-formed image for `values` of type T.
template <typename T>
std::vector<std::byte> image_of(std::string_view descr, std::string_view shape,
                                std::initializer_list<T> values,
                                bool fortran = false) {
    const auto payload = raw<T>(values);
    return npy_image(header_dict(descr, shape, fortran), payload);
}

TEST(ParseNpy, ReadsAOneDimensionalFloat64Array) {
    const auto image = image_of<double>("'<f8'", "(3,)", {1.5, -2.0, 1e300});
    const NpyArray array = parse_npy(image);

    EXPECT_EQ(array.dtype(), DType::Float64);
    EXPECT_FALSE(array.fortran_order());
    EXPECT_EQ(array.size(), 3u);
    EXPECT_EQ(array.shape(), (std::vector<std::size_t>{3}));
    EXPECT_EQ(array.shape_string(), "(3,)");

    const auto values = array.as<double>();
    ASSERT_EQ(values.size(), 3u);
    EXPECT_DOUBLE_EQ(values[0], 1.5);
    EXPECT_DOUBLE_EQ(values[1], -2.0);
    EXPECT_DOUBLE_EQ(values[2], 1e300);
}

TEST(ParseNpy, ReadsEverySupportedDType) {
    EXPECT_EQ(parse_npy(image_of<float>("'<f4'", "(2,)", {1.0f, -1.0f})).as<float>()[1],
              -1.0f);
    EXPECT_EQ(parse_npy(image_of<std::int8_t>("'|i1'", "(2,)", {-128, 127}))
                  .as<std::int8_t>()[0],
              -128);
    EXPECT_EQ(parse_npy(image_of<std::int16_t>("'<i2'", "(1,)", {-32768}))
                  .as<std::int16_t>()[0],
              -32768);
    EXPECT_EQ(parse_npy(image_of<std::int32_t>("'<i4'", "(1,)", {-2000000000}))
                  .as<std::int32_t>()[0],
              -2000000000);
    EXPECT_EQ(parse_npy(image_of<std::int64_t>("'<i8'", "(1,)", {-1LL << 62}))
                  .as<std::int64_t>()[0],
              -1LL << 62);
    EXPECT_EQ(parse_npy(image_of<std::uint8_t>("'|u1'", "(1,)", {255}))
                  .as<std::uint8_t>()[0],
              255);
    EXPECT_EQ(parse_npy(image_of<std::uint16_t>("'<u2'", "(1,)", {65535}))
                  .as<std::uint16_t>()[0],
              65535);
    EXPECT_EQ(parse_npy(image_of<std::uint32_t>("'<u4'", "(1,)", {4294967295u}))
                  .as<std::uint32_t>()[0],
              4294967295u);
    EXPECT_EQ(parse_npy(image_of<std::uint64_t>("'<u8'", "(1,)", {~0ULL}))
                  .as<std::uint64_t>()[0],
              ~0ULL);
}

TEST(ParseNpy, ReadsBooleanArrays) {
    // numpy stores bool as one byte per element, 0 or 1.
    const auto image = image_of<std::uint8_t>("'|b1'", "(3,)", {1, 0, 1});
    const NpyArray array = parse_npy(image);

    ASSERT_EQ(array.dtype(), DType::Bool);
    const auto values = array.as<bool>();
    ASSERT_EQ(values.size(), 3u);
    EXPECT_TRUE(values[0]);
    EXPECT_FALSE(values[1]);
    EXPECT_TRUE(values[2]);
}

// float16 has no C++ arithmetic type, so it parses but only bytes() can reach
// the payload. The point of supporting it at all is that a caller can detect
// and reject it by dtype instead of getting a size mismatch.
TEST(ParseNpy, ReadsFloat16AsOpaqueBytes) {
    const auto image = image_of<std::uint16_t>("'<f2'", "(4,)", {0, 1, 2, 3});
    const NpyArray array = parse_npy(image);

    EXPECT_EQ(array.dtype(), DType::Float16);
    EXPECT_EQ(array.size(), 4u);
    EXPECT_EQ(array.bytes().size(), 8u);
    EXPECT_THROW((void)array.as<float>(), NpyError);
}

TEST(ParseNpy, AcceptsNativeAndUnorderedDescrs) {
    EXPECT_EQ(parse_npy(image_of<double>("'=f8'", "(1,)", {2.0})).dtype(),
              DType::Float64);
    EXPECT_EQ(parse_npy(image_of<double>("'f8'", "(1,)", {2.0})).dtype(),
              DType::Float64);
    // Byte order is meaningless for a single byte, so '>' is fine at width 1.
    EXPECT_EQ(parse_npy(image_of<std::int8_t>("'>i1'", "(1,)", {5})).dtype(),
              DType::Int8);
}

TEST(ParseNpy, ZeroDimensionalArrayHoldsOneElement) {
    const auto image = image_of<double>("'<f8'", "()", {7.25});
    const NpyArray array = parse_npy(image);

    EXPECT_TRUE(array.shape().empty());
    EXPECT_EQ(array.size(), 1u);
    EXPECT_EQ(array.shape_string(), "()");
    EXPECT_DOUBLE_EQ(array.as<double>()[0], 7.25);
}

TEST(ParseNpy, EmptyArrayHasNoElements) {
    const auto image = npy_image(header_dict("'<f4'", "(0,)", false), {});
    const NpyArray array = parse_npy(image);

    EXPECT_EQ(array.size(), 0u);
    EXPECT_EQ(array.shape(), (std::vector<std::size_t>{0}));
    EXPECT_TRUE(array.bytes().empty());
    EXPECT_TRUE(array.as<float>().empty());
}

TEST(ParseNpy, MultiDimensionalShapeMultipliesOut) {
    const auto image =
        image_of<std::int32_t>("'<i4'", "(2, 3)", {0, 1, 2, 3, 4, 5});
    const NpyArray array = parse_npy(image);

    EXPECT_EQ(array.shape(), (std::vector<std::size_t>{2, 3}));
    EXPECT_EQ(array.size(), 6u);
    EXPECT_EQ(array.shape_string(), "(2, 3)");
    EXPECT_EQ(array.as<std::int32_t>()[5], 5);
}

TEST(ParseNpy, FortranOrderIsReportedNotApplied) {
    const auto image =
        image_of<double>("'<f8'", "(2, 2)", {1.0, 2.0, 3.0, 4.0}, true);
    const NpyArray array = parse_npy(image);

    EXPECT_TRUE(array.fortran_order());
    // The payload is handed back exactly as stored -- column-major, in this
    // case -- so a caller that ignores fortran_order() reads it transposed.
    EXPECT_DOUBLE_EQ(array.as<double>()[1], 2.0);
}

TEST(ParseNpy, AcceptsVersion2And3Headers) {
    // Versions 2 and 3 differ only in a 4-byte header length (and, for 3, utf-8
    // field names, which are irrelevant to the three keys we read).
    for (const int major : {2, 3}) {
        const auto payload = raw<double>({6.5});
        const auto image =
            npy_image(header_dict("'<f8'", "(1,)", false), payload, major);
        const NpyArray array = parse_npy(image);
        EXPECT_DOUBLE_EQ(array.as<double>()[0], 6.5) << "version " << major;
    }
}

TEST(ParseNpy, IgnoresBytesPastTheDeclaredPayload) {
    auto image = image_of<double>("'<f8'", "(2,)", {1.0, 2.0});
    image.resize(image.size() + 32, std::byte{0xab});

    const NpyArray array = parse_npy(image);
    EXPECT_EQ(array.size(), 2u);
    EXPECT_EQ(array.bytes().size(), 16u);
}

TEST(ParseNpy, TypedAccessRejectsAMismatchedType) {
    const auto image = image_of<double>("'<f8'", "(1,)", {1.0});
    const NpyArray array = parse_npy(image);

    EXPECT_NO_THROW((void)array.as<double>());
    EXPECT_THROW((void)array.as<float>(), NpyError);
    EXPECT_THROW((void)array.as<std::int64_t>(), NpyError);

    // The message names both dtypes, since "wrong type" alone is not enough to
    // find the mistake in a file with eighteen members.
    try {
        (void)array.as<float>();
        FAIL() << "expected NpyError";
    } catch (const NpyError& e) {
        const std::string what = e.what();
        EXPECT_NE(what.find("float64"), std::string::npos) << what;
        EXPECT_NE(what.find("float32"), std::string::npos) << what;
    }
}

TEST(ParseNpy, DefaultConstructedArrayIsEmpty) {
    const NpyArray array;
    EXPECT_EQ(array.size(), 0u);
    EXPECT_TRUE(array.shape().empty());
    EXPECT_TRUE(array.bytes().empty());
    EXPECT_EQ(array.shape_string(), "()");
}

TEST(ParseNpy, RejectsBadMagic) {
    auto image = image_of<double>("'<f8'", "(1,)", {1.0});
    image[1] = std::byte{'X'};
    EXPECT_THROW(parse_npy(image), NpyError);

    // Too short to even hold the magic and version.
    const std::vector<std::byte> stub(4, std::byte{0});
    EXPECT_THROW(parse_npy(stub), NpyError);
    EXPECT_THROW(parse_npy({}), NpyError);
}

TEST(ParseNpy, RejectsAnUnsupportedFormatVersion) {
    const auto payload = raw<double>({1.0});
    const auto image =
        npy_image(header_dict("'<f8'", "(1,)", false), payload, 4);
    EXPECT_THROW(parse_npy(image), NpyError);
}

TEST(ParseNpy, RejectsATruncatedHeader) {
    auto image = image_of<double>("'<f8'", "(1,)", {1.0});
    image.resize(20);  // header length still claims the full dict
    EXPECT_THROW(parse_npy(image), NpyError);

    // A version-2 image cut off inside its own 4-byte length field.
    auto v2 = npy_image(header_dict("'<f8'", "(1,)", false), raw<double>({1.0}), 2);
    v2.resize(11);
    EXPECT_THROW(parse_npy(v2), NpyError);
}

TEST(ParseNpy, RejectsATruncatedPayload) {
    auto image = image_of<double>("'<f8'", "(4,)", {1.0, 2.0, 3.0, 4.0});
    image.resize(image.size() - 1);
    EXPECT_THROW(parse_npy(image), NpyError);
}

TEST(ParseNpy, RejectsBigEndianData) {
    const auto image = image_of<double>("'>f8'", "(1,)", {1.0});
    EXPECT_THROW(parse_npy(image), NpyError);
}

TEST(ParseNpy, RejectsStructuredDtypes) {
    const auto payload = raw<std::int32_t>({1, 2});
    const auto dict = header_dict("[('a', '<i4'), ('b', '<i4')]", "(1,)", false);
    EXPECT_THROW(parse_npy(npy_image(dict, payload)), NpyError);
}

TEST(ParseNpy, RejectsDtypesWithNoCppEquivalent) {
    const auto payload = raw<std::uint8_t>({0, 0, 0, 0, 0, 0, 0, 0});
    for (const std::string_view descr : {"'<i3'", "'<c16'", "'|S5'", "'<f8x'"}) {
        const auto dict = header_dict(descr, "(1,)", false);
        EXPECT_THROW(parse_npy(npy_image(dict, payload)), NpyError) << descr;
    }
}

TEST(ParseNpy, RejectsAMalformedHeaderDict) {
    const auto payload = raw<double>({1.0});

    // Missing keys.
    EXPECT_THROW(parse_npy(npy_image("{'fortran_order': False, 'shape': (1,), }",
                                     payload)),
                 NpyError);
    EXPECT_THROW(parse_npy(npy_image("{'descr': '<f8', 'shape': (1,), }", payload)),
                 NpyError);
    EXPECT_THROW(
        parse_npy(npy_image("{'descr': '<f8', 'fortran_order': False, }", payload)),
        NpyError);

    // Present but unparseable.
    EXPECT_THROW(parse_npy(npy_image(header_dict("'<f8'", "1,", false), payload)),
                 NpyError);
    EXPECT_THROW(
        parse_npy(npy_image(header_dict("'<f8'", "(x,)", false), payload)),
        NpyError);
    EXPECT_THROW(parse_npy(npy_image("{'descr': '<f8', 'fortran_order': Maybe, "
                                     "'shape': (1,), }",
                                     payload)),
                 NpyError);
}

TEST(DTypeTraits, NameAndWidthAgreeForEveryDType) {
    struct Case {
        DType dtype;
        std::string_view name;
        std::size_t width;
    };
    constexpr Case kCases[]{
        {DType::Bool, "bool", 1},       {DType::Int8, "int8", 1},
        {DType::Int16, "int16", 2},     {DType::Int32, "int32", 4},
        {DType::Int64, "int64", 8},     {DType::UInt8, "uint8", 1},
        {DType::UInt16, "uint16", 2},   {DType::UInt32, "uint32", 4},
        {DType::UInt64, "uint64", 8},   {DType::Float16, "float16", 2},
        {DType::Float32, "float32", 4}, {DType::Float64, "float64", 8},
    };
    for (const auto& [dtype, name, width] : kCases) {
        EXPECT_EQ(boatforge::to_string(dtype), name);
        EXPECT_EQ(boatforge::word_size(dtype), width) << name;
    }
}

TEST(DTypeTraits, DtypeOfMapsTheCppArithmeticTypes) {
    static_assert(boatforge::dtype_of<bool>() == DType::Bool);
    static_assert(boatforge::dtype_of<std::int8_t>() == DType::Int8);
    static_assert(boatforge::dtype_of<std::int16_t>() == DType::Int16);
    static_assert(boatforge::dtype_of<std::int32_t>() == DType::Int32);
    static_assert(boatforge::dtype_of<std::int64_t>() == DType::Int64);
    static_assert(boatforge::dtype_of<std::uint8_t>() == DType::UInt8);
    static_assert(boatforge::dtype_of<std::uint16_t>() == DType::UInt16);
    static_assert(boatforge::dtype_of<std::uint32_t>() == DType::UInt32);
    static_assert(boatforge::dtype_of<std::uint64_t>() == DType::UInt64);
    static_assert(boatforge::dtype_of<float>() == DType::Float32);
    static_assert(boatforge::dtype_of<double>() == DType::Float64);
    SUCCEED();
}

TEST(ShapeString, MatchesNumpysRendering) {
    const auto shape_of = [](std::string_view shape) {
        const auto payload = raw<std::uint8_t>({0, 0, 0, 0, 0, 0, 0, 0});
        return parse_npy(npy_image(header_dict("'|u1'", shape, false), payload))
            .shape_string();
    };
    EXPECT_EQ(shape_of("()"), "()");
    EXPECT_EQ(shape_of("(1,)"), "(1,)");       // 1-d keeps the trailing comma
    EXPECT_EQ(shape_of("(2, 4)"), "(2, 4)");   // higher rank does not
    EXPECT_EQ(shape_of("(2, 2, 2)"), "(2, 2, 2)");
}

}  // namespace
