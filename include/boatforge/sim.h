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
        const std::filesystem::path& path, const std::filesystem::path& output_path)
        : solar_field_(blackboard_, path),
          solver_(blackboard_),
          boat_(blackboard_),
          world_(blackboard_),
          info_(blackboard_),
          output_path_(output_path)
    {
        blackboard_.time = start_time;
        blackboard_.current_lat = start.lat;
        blackboard_.current_lon = start.lon;
        blackboard_.end_lat = end.lat;
        blackboard_.end_lon = end.lon;

        recorder_.record("lng", blackboard_.current_lon);
        recorder_.record("lat", blackboard_.current_lat);
        recorder_.record("time", blackboard_.time);
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

        recorder_.record("lng", blackboard_.current_lon);
        recorder_.record("lat", blackboard_.current_lat);
        recorder_.record("time", blackboard_.time);
        // FIXME: save output here -> to RAM, later to disk...
        count_--;
        return count_ > 0;  // FIXME: For now only do a fixed number of steps
    }

    void end()
    {
        recorder_.save(output_path_);
    }

private:
    blackboard blackboard_;
    SolarIsolationField solar_field_;
    Solver solver_;
    BoatState boat_;
    WorldPropogation world_;
    Info info_;
    const std::filesystem::path& output_path_;

    uint16_t count_ = 100;
    boatforge::NpzRecorder recorder_;
};
