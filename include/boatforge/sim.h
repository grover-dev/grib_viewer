#pragma once
#include <boatforge/dynamics.h>

#include <npy_tools/npz_recorder.h>
#include <chrono>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <map>
#include <print>
#include <string>
#include <vector>

#include <npy_tools/npz_field.h>
class Sim
{
public:
    // FIXME: move this into the blackboard def eventuall,y no need for special structs
    struct lat_lon
    {
        double lat = 0.0;
        double lon = 0.0;
    };

    /* One simulated voyage. Everything here is plain data with a default, so a
     * yaml node maps onto it field by field and a key left out of the file just
     * keeps the default rather than leaving a member uninitialised. */
    struct run_t
    {
        /* Names the output file, out_directory / name + ".npz", so it has to be
         * unique across the runs of one config. */
        std::string name = "run";

        /* Departure, seconds since the Unix epoch UTC -- the frame the solar
         * npz time axis uses. From yaml this is either an integer stamp or a
         * date string the loader converts. */
        std::chrono::seconds start_time{0};

        /* Degrees, WGS84 */
        lat_lon start{};
        lat_lon end{};

        /* npz written by scripts/grib_npz.py. Loaded by Sim, and shared between
         * runs that name the same file, so a sweep of start days over one
         * weather cube only pays for the load once. */
        std::filesystem::path solar_field{};

        /* Steps of blackboard::time_step to run before giving up on reaching
         * the end point. 0 means "no cap" once the sim can terminate on
         * arrival. */
        // FIXME: For now this is the only termination condition
        uint32_t max_steps = 100;
    };

    /* The whole of an invocation: what to run and where to put it. This is the
     * root of the yaml document. */
    struct config_t
    {
        std::filesystem::path out_directory = ".";
        std::vector<run_t> runs;
    };

    explicit Sim(config_t config) : config_(std::move(config))
    {
        for (auto& run : config_.runs)
        {
            /* Look up before loading so a repeated path loads once; map nodes
             * are stable, so the reference handed to the sim survives later
             * inserts. */
            auto field = solar_fields_.find(run.solar_field);
            if (field == solar_fields_.end())
            {
                field = solar_fields_.emplace(run.solar_field, boatforge::NpzField::load(run.solar_field)).first;
            }

            instances_.emplace_back(run, field->second, config_.out_directory);
        }

        for (auto& instace : instances_)
        {
            instace.sample();
        }
    }

    // TODO:
    // - run multi start day sims, track duration + energy, minimize for duration and off time
    // - stick to waypointed great circle distance for now, eentually move to something more intelligent
    //   - as a basic option can hand plot a few routes, then check their performance. avoid a lot of unnecessary search

    /* True while any instance still has stepping left to do. Instances that
     * have finished are stepped no further, so a short run does not hold back
     * -- or get dragged along by -- a long one. */
    bool run()
    {
        bool running = false;

        for (auto& instance : instances_)
        {
            if (instance.blackboard_.data_valid)
            {
                running |= instance.step();
            }
        }

        return running;
    }

    // FIXME: move to destructor
    void end()
    {
        // FIXME: rework the recorder to take in a  data dict eventually

        for (auto& instace : instances_)
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
        const run_t& run_;
        /* By value, not by reference: the config member it is built from
         * outlives us, but the joined path is a temporary. */
        const std::filesystem::path output_path_;
        blackboard blackboard_;
        SolarIsolationField solar_field_;
        Solver solver_;
        BoatState boat_;
        WorldPropogation world_;
        Info info_;
        boatforge::NpzRecorder recorder_;
        uint32_t steps_left_;

        sim_t(const run_t& run, boatforge::NpzField& solar_field, const std::filesystem::path& out_directory)
            : run_(run),
              output_path_(out_directory / (run.name + ".npz")),
              solar_field_(blackboard_, solar_field),
              solver_(blackboard_),
              boat_(blackboard_),
              world_(blackboard_),
              info_(blackboard_),
              steps_left_(run.max_steps)
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
            if (steps_left_ == 0)
            {
                return false;
            }

            solar_field_.sample();
            /* Off the end of the field's coverage: everything downstream would
             * be modelled on data that is not there, so the run ends here with
             * the track it has rather than a tail of NaNs. */
            if (!blackboard_.data_valid)
            {
                std::println("{}: no field data at step {}, ending run early", run_.name, blackboard_.steps);
                /* Retire the instance, or the next call would step it again --
                 * the sim as a whole runs until the longest run finishes. */
                steps_left_ = 0;
                return false;
            }

            solver_.step();
            boat_.step();
            world_.step();

            info_.step();

            blackboard_.steps++;
            blackboard_.time += blackboard_.time_step;
            blackboard_.total_time += blackboard_.time_step;

            sample();
            steps_left_--;
            return steps_left_ > 0;  // FIXME: For now only do a fixed number of steps
        }

        void end()
        {
            std::filesystem::create_directories(output_path_.parent_path());
            recorder_.save(output_path_);
        }
    };

    config_t config_;

    /* Keyed by path so runs sharing a field share one load. Node addresses are
     * stable across inserts, which is what lets sim_t hold a reference. */
    std::map<std::filesystem::path, boatforge::NpzField> solar_fields_;

    /* deque, not vector: sim_t's members hold references to its own blackboard,
     * so an instance that got relocated by a growing vector would leave every
     * model pointing at the old one. */
    std::deque<sim_t> instances_;
};
