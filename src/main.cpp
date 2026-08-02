#include <cstddef>
#include <exception>
#include <filesystem>
#include <print>
#include <string>

#include <boatforge/sim.h>

int main(int argc, char** argv)
{
    /**
     * How to start?
     * - Map has been created, lets architect the iteration loop
     *   - Will probably need to optimize, problem for later
     *
     *
     *
     */
    const std::string path = argc > 1 ? argv[1] : "input.npz";

    std::println("boatforge — reading {}", path);

    Sim::lat_lon start;
    // roughly off the coast of spain
    start.lat = 40.0;
    start.lon = -13.0;

    // further off the coast of spaiun
    Sim::lat_lon end;
    end.lat = 36.0;
    end.lon = -22.0;

    const std::string out_path = argc > 2 ? argv[2] : "track.npz";
    Sim simulator(std::chrono::seconds(1735776000), start, end, path, out_path);

    while (simulator.step())
    {
    }

    simulator.end();
    return 0;
}
