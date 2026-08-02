#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>

#include <npy_tools/npy.h>
#include <npy_tools/npz.h>

namespace boatforge {

// Accumulates a value per step per field and writes them out as parallel 1-d
// arrays -- one .npy member per field, all the same length. That is the layout
// numpy.load() and pandas.DataFrame(dict(np.load(...))) both expect, so a run
// dumped this way plots without any reshaping on the far side.
//
// Meant for a per-step struct like `blackboard`: call record() for each field
// you care about at the bottom of the step, then save() once at the end.
//
//     NpzRecorder log;
//     log.set_scalar("time_step_s", bb.time_step.count());   // written once
//     // ... each step:
//     log.record("time", bb.time);                           // seconds, int64
//     log.record("lat", bb.current_lat);                     // float64
//     log.record("solar_power_in_w", bb.solar_power_in_w);   // float32
//     // ... at the end:
//     log.save("track.npz");
//
// A field's type is fixed by its first record(): recording a double into a
// column that started as float throws rather than silently converting, since a
// column that changes width mid-run is a bug that only shows up in the plot.
class NpzRecorder {
public:
    NpzRecorder();
    ~NpzRecorder();
    NpzRecorder(NpzRecorder&&) noexcept;
    NpzRecorder& operator=(NpzRecorder&&) noexcept;

    // Append one sample. The element type picks the dtype, so a float field
    // stays float32 in the output instead of being widened on the way out.
    template <typename T>
    void record(std::string_view name, const T& value) {
        append(name, dtype_of<T>(), std::as_bytes(std::span<const T, 1>{&value, 1}));
    }

    // Durations are recorded as whole seconds, matching how the gridded-field
    // npz stores its time axis. Anything finer is truncated, so record the
    // count() yourself if sub-second resolution matters.
    template <typename Rep, typename Period>
    void record(std::string_view name, std::chrono::duration<Rep, Period> value) {
        const auto seconds =
            std::chrono::duration_cast<std::chrono::seconds>(value).count();
        record<std::int64_t>(name, static_cast<std::int64_t>(seconds));
    }

    // A value that does not vary per step -- a configuration constant, a run
    // id -- written once as a 0-d array. Setting the same name again replaces
    // it; using a name that is already a recorded column throws.
    template <typename T>
    void set_scalar(std::string_view name, const T& value) {
        set_scalar_bytes(name, dtype_of<T>(),
                         std::as_bytes(std::span<const T, 1>{&value, 1}));
    }

    template <typename Rep, typename Period>
    void set_scalar(std::string_view name, std::chrono::duration<Rep, Period> value) {
        const auto seconds =
            std::chrono::duration_cast<std::chrono::seconds>(value).count();
        set_scalar<std::int64_t>(name, static_cast<std::int64_t>(seconds));
    }

    // Number of per-step columns, and how many samples the longest one holds.
    std::size_t columns() const;
    std::size_t rows() const;

    // Samples recorded under `name` so far; 0 if there is no such column.
    std::size_t rows(std::string_view name) const;

    // Throws if the columns are not all the same length -- a field recorded on
    // only one branch of the step would otherwise produce an npz whose columns
    // silently do not line up.
    void save(const std::filesystem::path& path,
              Compression compression = Compression::Store) const;

    // Drop everything, columns and scalars alike, ready for the next run.
    void clear();

private:
    void append(std::string_view name, DType dtype, std::span<const std::byte> value);
    void set_scalar_bytes(std::string_view name, DType dtype,
                          std::span<const std::byte> value);

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace boatforge
