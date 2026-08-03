#include <chrono>
#include <cstdio>
#include <print>
#include <string>

#include <boatforge/dynamics.h>
#include <boatforge/sim.h>

namespace
{
/* Stand-in for the yaml loader: the same document, written by hand. Replace
 * this with a parse of argv[1] and nothing else about main() has to change --
 * the whole of a run's input is the config_t it returns. */
Sim::config_t hard_coded_config(const std::filesystem::path& solar_field, const std::filesystem::path& out_directory)
{
    Sim::config_t config;
    config.out_directory = out_directory;

    Sim::run_t run;
    run.name = "run";
    run.start_time = std::chrono::seconds(1735776000) + std::chrono::hours(12);
    /* roughly off the coast of spain */
    run.start = {.lat = 40.0, .lon = -13.0};
    /* further off the coast of spaiun */
    run.end = {.lat = 36.0, .lon = -22.0};
    run.solar_field = solar_field;
    run.max_steps = 100;

    config.runs.push_back(run);

    return config;
}
}  // namespace

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::printf("usage: boatforge <field.npz> <out-directory>\n");
        return 1;
    }

    // FIXME: replace both arguments with a single config.yaml
    const std::filesystem::path path = argv[1];
    const std::filesystem::path out_directory = argv[2];

    std::println("boatforge — reading {}", path.string());

    Sim simulator(hard_coded_config(path, out_directory));

    while (simulator.run())
    {
    }

    simulator.end();
    return 0;
}
