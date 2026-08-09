#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <format>
#include <print>
#include <set>
#include <sstream>
#include <string>

#include <yaml-cpp/yaml.h>

#include <boatforge/dynamics.h>
#include <boatforge/sim.h>

namespace
{
/* Paths in a config are relative to the config file, not to the working
 * directory, so a config and the npz it names can be moved together. */
std::filesystem::path resolve(const std::filesystem::path& base, const std::filesystem::path& path)
{
    return path.is_absolute() ? path : base / path;
}

/* Departure instant. Accepts seconds since the Unix epoch, or an ISO-8601 UTC
 * stamp ("2025-01-02T12:00:00Z"), which is what a hand-written config wants. */
std::chrono::seconds parse_start_time(const YAML::Node& node)
{
    std::int64_t epoch_seconds = 0;
    if (YAML::convert<std::int64_t>::decode(node, epoch_seconds))
    {
        return std::chrono::seconds(epoch_seconds);
    }

    const std::string text = node.as<std::string>();
    std::istringstream stream(text);
    std::chrono::sys_seconds stamp;
    stream >> std::chrono::parse("%FT%TZ", stamp);
    if (stream.fail())
    {
        throw std::runtime_error("start_time is neither epoch seconds nor an ISO stamp: " + text);
    }

    return stamp.time_since_epoch();
}

/* Spacing between the runs of a sweep. Accepts a suffixed duration ("2d",
 * "12h", "90m", "1800s") or plain seconds, so "a day apart" reads as `1d`
 * rather than as 86400. Fractions are allowed -- `0.5d` is twelve hours --
 * because the useful spacings are not all whole units of anything. */
std::chrono::seconds parse_duration(const YAML::Node& node)
{
    const std::string text = node.as<std::string>();
    if (text.empty())
    {
        throw std::runtime_error("empty duration");
    }

    double scale = 1.0;
    std::string number = text;
    switch (text.back())
    {
        case 'd': scale = 86400.0; break;
        case 'h': scale = 3600.0; break;
        case 'm': scale = 60.0; break;
        case 's': scale = 1.0; break;
        default: scale = 0.0; break;  // no suffix: the whole string is seconds
    }
    if (scale != 0.0)
    {
        number.pop_back();
    }
    else
    {
        scale = 1.0;
    }

    std::size_t consumed = 0;
    double value = 0.0;
    try
    {
        value = std::stod(number, &consumed);
    }
    catch (const std::exception&)
    {
        throw std::runtime_error("not a duration: " + text + " (try 1d, 12h, 90m, or seconds)");
    }
    if (consumed != number.size())
    {
        throw std::runtime_error("not a duration: " + text + " (try 1d, 12h, 90m, or seconds)");
    }

    return std::chrono::seconds{static_cast<std::int64_t>(std::llround(value * scale))};
}

Sim::lat_lon parse_lat_lon(const YAML::Node& node)
{
    return Sim::lat_lon{.lat = node["lat"].as<double>(), .lon = node["lon"].as<double>()};
}

/* Overlays one node onto `run`, leaving fields the node does not mention as
 * they were. That is what makes `defaults:` and a run entry the same shape: the
 * defaults block is overlaid onto the struct's own defaults, and each run entry
 * onto the result, so a run only writes what it changes. */
void overlay_run(Sim::run_t& run, const YAML::Node& node, const std::filesystem::path& base)
{
    if (!node || !node.IsMap())
    {
        throw std::runtime_error("expected a mapping of run fields");
    }

    if (node["name"])
    {
        run.name = node["name"].as<std::string>();
    }
    if (node["start_time"])
    {
        run.start_time = parse_start_time(node["start_time"]);
    }
    if (node["start"])
    {
        run.start = parse_lat_lon(node["start"]);
    }
    if (node["end"])
    {
        run.end = parse_lat_lon(node["end"]);
    }
    if (node["solar_field"])
    {
        run.solar_field = resolve(base, node["solar_field"].as<std::string>());
    }
    if (node["current_u_field"])
    {
        run.current_u_field = resolve(base, node["current_u_field"].as<std::string>());
    }
    if (node["current_v_field"])
    {
        run.current_v_field = resolve(base, node["current_v_field"].as<std::string>());
    }
    if (node["max_steps"])
    {
        run.max_steps = node["max_steps"].as<std::uint32_t>();
    }
}

/* The whole of an invocation. Runs share one output directory and are named by
 * hand, so a repeated name is rejected here rather than left to overwrite
 * another run's npz halfway through the sweep. */
Sim::config_t load_config(const std::filesystem::path& config_path)
{
    const YAML::Node document = YAML::LoadFile(config_path.string());
    const std::filesystem::path base = std::filesystem::absolute(config_path).parent_path();

    Sim::config_t config;
    if (document["out_directory"])
    {
        config.out_directory = resolve(base, document["out_directory"].as<std::string>());
    }
    if (document["field_cache_parts"])
    {
        config.field_cache_parts = document["field_cache_parts"].as<std::size_t>();
        if (config.field_cache_parts == 0)
        {
            throw std::runtime_error("field_cache_parts is 0; a field needs at least one part in memory");
        }
    }

    Sim::run_t defaults;
    if (document["defaults"])
    {
        overlay_run(defaults, document["defaults"], base);
    }

    const YAML::Node runs = document["runs"];
    if (!runs || !runs.IsSequence() || runs.size() == 0)
    {
        throw std::runtime_error("config has no `runs:` sequence");
    }

    std::set<std::string> names;
    for (std::size_t index = 0; index < runs.size(); index++)
    {
        Sim::run_t run = defaults;
        /* Positional fallback, so an unnamed run in a sweep still lands in its
         * own file instead of on top of the previous one. */
        run.name = std::format("run_{:02}", index);
        overlay_run(run, runs[index], base);

        /* `count` turns one entry into a sweep: the same run departing at a
         * fixed interval, which is the shape a start-day study wants and the
         * only thing that varies across it. Written as an expansion rather than
         * as a field on run_t so that everything downstream -- the duplicate
         * name check, Sim, the recorder -- keeps seeing a flat list of runs
         * that could equally have been typed out by hand. */
        std::size_t count = 1;
        std::chrono::seconds every{0};
        if (runs[index]["count"])
        {
            count = runs[index]["count"].as<std::size_t>();
            if (count == 0)
            {
                throw std::runtime_error(run.name + ": count is 0; a sweep needs at least one run");
            }
        }
        if (runs[index]["every"])
        {
            try
            {
                every = parse_duration(runs[index]["every"]);
            }
            catch (const std::exception& e)
            {
                throw std::runtime_error(run.name + ": " + e.what());
            }
        }
        if (count > 1 && every <= std::chrono::seconds{0})
        {
            throw std::runtime_error(run.name + ": count > 1 needs a positive `every`, or the runs would "
                                                "all depart at the same instant");
        }

        /* Wide enough for the last index, so the names sort in departure order
         * in a directory listing rather than putting run_10 before run_2. */
        const int width = static_cast<int>(std::to_string(count - 1).size());
        const std::string stem = run.name;

        for (std::size_t step = 0; step < count; step++)
        {
            Sim::run_t entry = run;
            if (count > 1)
            {
                entry.name = std::format("{}_{:0{}}", stem, step, width);
                entry.start_time = run.start_time + every * static_cast<std::int64_t>(step);
            }

            if (entry.solar_field.empty())
            {
                throw std::runtime_error(entry.name + ": no solar_field, in the run or in defaults");
            }
            /* Both or neither: one component is not a current, and a run that
             * named only half of one would otherwise sail through a field that
             * silently flows due east or due north. */
            if (entry.current_u_field.empty() != entry.current_v_field.empty())
            {
                throw std::runtime_error(entry.name + ": current_u_field and current_v_field go together; "
                                                      "one component on its own is not a current");
            }
            if (!names.insert(entry.name).second)
            {
                throw std::runtime_error("two runs are named " + entry.name + "; they would share an output file");
            }

            config.runs.push_back(std::move(entry));
        }
    }

    return config;
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::printf("usage: boatforge <config.yaml>\n");
        return 1;
    }

    const std::filesystem::path config_path = argv[1];

    try
    {
        const Sim::config_t config = load_config(config_path);
        std::println("boatforge — {} run(s) from {}", config.runs.size(), config_path.string());

        Sim simulator(config);

        while (simulator.run())
        {
        }

        simulator.end();
    }
    catch (const std::exception& error)
    {
        std::println(stderr, "boatforge: {}", error.what());
        return 1;
    }

    return 0;
}
