#include <cstddef>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <print>
#include <string>

#include <boatforge/dynamics.h>
#include <boatforge/sim.h>

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::printf("Need at least 3 arguments");
        return 1;
    }
    // FIXME: change this to take in larger blocks of data
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

    auto solar_field = boatforge::NpzField::load(path);

    const std::string out_directory = argv[2];

    // FIXME: take in a list of start + end conditions
    //
    // FIXME: build an array of these runs

    Sim simulator(std::chrono::seconds(1735776000) + std::chrono::hours(12), start, end, solar_field, out_directory);

    while (simulator.step())
    {
    }

    simulator.end();
    return 0;
}
