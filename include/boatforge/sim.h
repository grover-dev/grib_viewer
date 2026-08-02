#pragma once
#include <boatforge/dynamics.h>

#include <npy_tools/npz_recorder.h>
#include <filesystem>

class Sim
{
public:
    struct lat_lon
    {
        double lat;
        double lon;
    };
    // FIXME: add start lat/lon, end lat/lon
    Sim(const std::chrono::seconds start_time, const lat_lon start, const lat_lon end,
        const std::filesystem::path& path, std::filesystem::path output_path)
        : solar_field_(blackboard_, path),
          solver_(blackboard_),
          boat_(blackboard_),
          world_(blackboard_),
          info_(blackboard_),
          output_path_(std::move(output_path)),
          start_time_(start_time)
    {
        blackboard_.time = start_time;
        blackboard_.current_lat = start.lat;
        blackboard_.current_lon = start.lon;
        blackboard_.end_lat = end.lat;
        blackboard_.end_lon = end.lon;

        sample();
    }

    bool step()
    {
        solar_field_.sample();
        solver_.step();
        boat_.step();
        world_.step();

        info_.step();

        blackboard_.steps++;
        blackboard_.time += blackboard_.time_step;
        blackboard_.total_time += blackboard_.time_step;

        sample();
        count_--;
        return count_ > 0;  // FIXME: For now only do a fixed number of steps
    }

    void end()
    {
        recorder_.save(output_path_);
    }

private:
    /* One row of the track, in the shape scripts/vis_map.py ingests: `lat`,
     * `lng` and `time` are the three keys it requires, and every other column of
     * the same length is offered as a channel to shade the course by.
     *
     * `time` is hours since departure as a float, not an epoch stamp -- vis_map
     * reads it straight as hours ("101 points over 1736136000.0 h" is what an
     * epoch value looks like there). */
    void sample()
    {
        recorder_.record("lat", blackboard_.current_lat);
        recorder_.record("lng", blackboard_.current_lon);
        recorder_.record("time",
                         static_cast<double>(
                             std::chrono::duration_cast<std::chrono::hours>(blackboard_.time - start_time_).count()));

        /* Extra channels, selectable with --track-scalar. Note vis_map clamps
         * whatever it shades by to [0, 1], so a channel meant for colour has to
         * be a fraction; these are raw and will saturate. */
        recorder_.record("solar_power_in_w", blackboard_.solar_power_in_w);
        recorder_.record("power_stored_wh", blackboard_.power_stored_wh);
        recorder_.record("distance_to_end_km", blackboard_.distance_to_end / 1000.0);
    }

    blackboard blackboard_;
    SolarIsolationField solar_field_;
    Solver solver_;
    BoatState boat_;
    WorldPropogation world_;
    Info info_;
    /* By value, not by reference: main.cpp passes a temporary path built from a
     * std::string, so a reference member would dangle by the time end() runs. */
    const std::filesystem::path output_path_;

    const std::chrono::seconds start_time_;
    uint16_t count_ = 100;
    boatforge::NpzRecorder recorder_;
};
