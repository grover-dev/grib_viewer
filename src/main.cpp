#include <cstddef>
#include <exception>
#include <print>
#include <string>
#include <filesystem>

#include <boatforge/sim.h>


int main(int argc, char** argv) {
    /**
     * How to start?
     * - Map has been created, lets architect the iteration loop
     *   - Will probably need to optimize, problem for later
     *
     *
     *
     */
    const std::string path = argc > 1 ? argv[1] : "track.npz";

    std::println("boatforge — reading {}", path);


    Sim::lat_lon start;
    // roughly off the coast of spain
   start.lat = 40.0;
    start.lon = 13.0;

    // further off the coast of spaiun
    Sim::lat_lon end;
    end.lat =36.0;
     end.lon = 22.0;


    Sim simulator(std::chrono::seconds(1735776000), start, end, path);

    while(simulator.step()){}

    return 0;
}
