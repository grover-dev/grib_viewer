#pragma once
#include <boatforge/dynamics.h>

#include <npy_tools/npz_recorder.h>
#include <filesystem>

#include <npy_tools/npz_field.h>
class Sim
{
public:
    // FIXME: move this into the blackboard def eventuall,y no need for special structs
    struct lat_lon
    {
        double lat;
        double lon;
    };

    struct run_t
    {
        const std::chrono::seconds start_time;
        const lat_lon start;
        const lat_lon end;
        boatforge::NpzField& solar_field;
        std::string output_name;
    };

    Sim(std::span<run_t> runs)
    {
        for (auto& run : runs)
        {
            instances.push_back(sim_t{run});
        }

        for (auto& instace : instances)
        {
            instace.sample();
        }
    }

    // TODO:
    // - run multi start day sims, track duration + energy, minimize for duration and off time
    // - stick to waypointed great circle distance for now, eentually move to something more intelligent
    //   - as a basic option can hand plot a few routes, then check their performance. avoid a lot of unnecessary search

    bool run()
    {
        bool done = true;

        for (auto& instace : instances)
        {
            done &= instace.step();
        }

        return done;
    }

    // FIXME: move to destructor
    void end()
    {
        // FIXME: rework the recorder to take in a  data dict eventually

        for (auto& instace : instances)
        {
            instace.end();
        }
    }

private:
    /* One row of the track, in the shape scripts/vis_map.py ingests: `lat`,
     * `lng` and `time` are the three keys it requires, and every other column of
     * the same length is offered as a channel to shade the course by.
     *
     * `time` is hours since departure as a float, not an epoch stamp -- vis_map
     * reads it straight as hours ("101 points over 1736136000.0 h" is what an
     * epoch value looks like there). */

    struct sim_t
    {
        run_t& run_;
        blackboard blackboard_;
        SolarIsolationField solar_field_;
        Solver solver_;
        BoatState boat_;
        WorldPropogation world_;
        Info info_;
        boatforge::NpzRecorder recorder_;
        uint16_t count_ = 100;

        sim_t(run_t& run)
            : run_(run),
              solar_field_(blackboard_, run_.solar_field),
              solver_(blackboard_),
              boat_(blackboard_),
              world_(blackboard_),
              info_(blackboard_)
        {
            blackboard_.time = run_.start_time;
            blackboard_.current_lat = run_.start.lat;
            blackboard_.current_lon = run_.start.lon;
            blackboard_.end_lat = run_.end.lat;
            blackboard_.end_lon = run_.end.lon;
        }

        void sample()
        {
            // FIXME: find a way to make blackboard auto add these... tbd
            recorder_.record("lat", blackboard_.current_lat);
            recorder_.record("lng", blackboard_.current_lon);
            recorder_.record(
                "time",
                static_cast<double>(
                    std::chrono::duration_cast<std::chrono::hours>(blackboard_.time - run_.start_time).count()));

            /* Extra channels, selectable with --track-scalar. Note vis_map clamps
             * whatever it shades by to [0, 1], so a channel meant for colour has to
             * be a fraction; these are raw and will saturate. */
            recorder_.record("solar_power_in_w", blackboard_.solar_power_in_w);
            recorder_.record("power_stored_wh", blackboard_.power_stored_wh);
            recorder_.record("distance_to_end_km", blackboard_.distance_to_end / 1000.0);
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
            // recorder_.save(output_path_);
        }
    };

    std::vector<sim_t> instances;

    /* By value, not by reference: main.cpp passes a temporary path built from a
     * std::string, so a reference member would dangle by the time end() runs. */
    // const std::filesystem::path output_path_;

    //
};
