#pragma once
#include <boatforge/dynamics.h>

#include <filesystem>

class Sim
{
    public:
        // FIXME: add start lat/lon, end lat/lon
        Sim( const std::filesystem::path& path): solar_field_(blackboard_, path), solver_(blackboard_), boat_(blackboard_), world_(blackboard_), info_(blackboard_){}


        bool step()
        {
            solar_field_.sample();
            solver_.step();
            boat_.step();
            world_.step();

            info_.step();
            // FIXME: save output here -> to RAM, later to disk...
            count_--;
            return count_ == 0; // FIXME: For now only do a fixed number of steps
        }

    private:
        blackboard blackboard_;
        SolarIsolationField solar_field_;
        Solver solver_;
        BoatState boat_;
        WorldPropogation world_;
        Info info_;

        uint16_t count_ = 100;
};
