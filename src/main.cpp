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

    Sim simulator(path);

    while(simulator.step()){}

    return 0;
}
