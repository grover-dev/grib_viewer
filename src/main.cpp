#include <chrono>
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

        if (run.solar_field.empty())
        {
            throw std::runtime_error(run.name + ": no solar_field, in the run or in defaults");
        }
        if (!names.insert(run.name).second)
        {
            throw std::runtime_error("two runs are named " + run.name + "; they would share an output file");
        }

        config.runs.push_back(std::move(run));
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
