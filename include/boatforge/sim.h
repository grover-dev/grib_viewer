#pragma once
#include <boatforge/dynamics.h>

#include <npy_tools/npz_recorder.h>
#include <chrono>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <map>
#include <optional>
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

        /* npz written by scripts/grib_npz.py, or a directory of the parts it
         * writes when one cube would exceed its byte limit -- which of the two
         * is decided by what is on disk, not by a second key. Loaded by Sim,
         * and shared between runs that name the same path, so a sweep of start
         * days over one weather cube only pays for the load once. A directory
         * additionally loads only the parts being sampled; see
         * config_t::field_cache_parts. */
        std::filesystem::path solar_field{};

        /* Ocean current, as the two components of the velocity rather than as a
         * speed and a bearing: `uo` eastward, `vo` northward, each its own
         * directory of parts written by scripts/netcdf4_npz.py (or a single
         * npz, decided by what is on disk). They are loaded and cached exactly
         * like solar_field, and a path repeated across runs is only loaded
         * once.
         *
         * The two want separate directories. A directory is read as the parts
         * of one field, and the components share a grid and a time step, so
         * both in one directory would load as a single field of duplicated
         * frames rather than being refused for disagreeing.
         *
         * Both or neither. A current with one component is not a current, so
         * the loader refuses a run that names only one; naming neither leaves
         * ocean_current_* at zero and simply models no current. */
        std::filesystem::path current_u_field{};
        std::filesystem::path current_v_field{};

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

        /* Parts of a split field kept in memory at once, per field. Runs are
         * stepped round-robin and share one field, so a sweep whose start days
         * fall in different parts has that many windows in flight at once --
         * and a cache smaller than that reloads a part on every alternation.
         * Two is enough for a sweep spanning one boundary; widen it for a
         * sweep spread over more, at one part of memory each.
         *
         * Ignored by a field that is a single npz: there is nothing to evict. */
        std::size_t field_cache_parts = 2;

        std::vector<run_t> runs;
    };

    explicit Sim(config_t config) : config_(std::move(config))
    {
        for (auto& run : config_.runs)
        {
            /* Every field a run names goes through one cache, so the two
             * current components and the solar cube share the same loading
             * rules -- and a path named twice, whether by two runs or by two
             * slots of one run, is still loaded once. */
            instances_.emplace_back(run, *field_for(run.solar_field), field_for(run.current_u_field),
                                    field_for(run.current_v_field), config_.out_directory);
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
    /* The loaded field at `path`, or nullptr for an empty path -- which is how
     * an optional field (the currents) says it was not configured.
     *
     * Looked up before loading so a repeated path loads once; map nodes are
     * stable, so the pointer handed to an instance survives later inserts. */
    boatforge::NpzField* field_for(const std::filesystem::path& path)
    {
        if (path.empty())
        {
            return nullptr;
        }

        auto field = fields_.find(path);
        if (field == fields_.end())
        {
            /* A directory is a field split along time, a file is the whole
             * cube. Asking the filesystem rather than the config keeps the two
             * from disagreeing about what is there. */
            boatforge::NpzField loaded = std::filesystem::is_directory(path)
                                             ? boatforge::NpzField::load_directory(path, config_.field_cache_parts)
                                             : boatforge::NpzField::load(path);
            field = fields_.emplace(path, std::move(loaded)).first;
        }

        return &field->second;
    }

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
        /* Empty when the run named no current fields: the model is absent
         * rather than present-and-zero, so nothing samples and nothing can
         * clear data_valid on a field that was never configured. */
        std::optional<OceanCurrentField> current_field_;
        Solver solver_;
        BoatState boat_;
        WorldPropogation world_;
        Info info_;
        boatforge::NpzRecorder recorder_;
        uint32_t steps_left_;

        sim_t(const run_t& run, boatforge::NpzField& solar_field, boatforge::NpzField* current_u,
              boatforge::NpzField* current_v, const std::filesystem::path& out_directory)
            : run_(run),
              output_path_(out_directory / (run.name + ".npz")),
              solar_field_(blackboard_, solar_field),
              solver_(blackboard_),
              boat_(blackboard_),
              world_(blackboard_),
              info_(blackboard_),
              steps_left_(run.max_steps)
        {
            /* Both or neither -- the config loader has already rejected a run
             * that named one without the other, so this only has to decide
             * whether the model exists at all. */
            if (current_u != nullptr && current_v != nullptr)
            {
                current_field_.emplace(blackboard_, *current_u, *current_v);
            }

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

            /* Recorded whether or not a current was configured, so every run in
             * a sweep writes the same columns and they can be compared without
             * checking which keys each file happens to have. A run with no
             * current field records the zeros that mean exactly that. */
            recorder_.record("ocean_current_heading_deg", blackboard_.ocean_current_heading);
            recorder_.record("ocean_current_velocity_ms", blackboard_.ocean_current_velocity);

            /* The three terms of WorldPropogation's vector sum, so a track can
             * be read back for where the speed over ground came from: what the
             * boat drove, what the world added, and what the two came to. They
             * do not sum arithmetically -- combined is the magnitude of the
             * vector sum, so a current on the nose makes it *smaller* than
             * powered alone, and a beam-on current makes it larger than either
             * while bending the track off the solver's heading. Reading the
             * three together is what makes that legible. */
            recorder_.record("powered_velocity_ms", blackboard_.powered_velocity);
            recorder_.record("environment_velocity_ms", blackboard_.environment_velocity);
            recorder_.record("combined_velocity_ms", blackboard_.combined_velocity);
        }

        bool step()
        {
            if (steps_left_ == 0)
            {
                return false;
            }

            // FIXME: Multi threading may cause issues here. Can add a field manager, safe to delete a field from memory
            // only if no one is using it
            // - To imporve paralellism we can then launch several processes with threads in each one, each process
            // loads at most N blocks, can tune based on data size
            solar_field_.sample();
            if (current_field_.has_value())
            {
                current_field_->sample();
            }
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
            if (blackboard_.distance_to_end < blackboard_.termination_distance)
            {
                return false;
            }
            return steps_left_ > 0;  // FIXME: For now only do a fixed number of steps
        }

        void end()
        {
            std::filesystem::create_directories(output_path_.parent_path());
            recorder_.save(output_path_);
        }
    };

    config_t config_;

    /* Every field of every kind, keyed by path so runs sharing one share a
     * single load. Node addresses are stable across inserts, which is what lets
     * sim_t hold a reference into it. */
    std::map<std::filesystem::path, boatforge::NpzField> fields_;

    /* deque, not vector: sim_t's members hold references to its own blackboard,
     * so an instance that got relocated by a growing vector would leave every
     * model pointing at the old one. */
    std::deque<sim_t> instances_;
};
