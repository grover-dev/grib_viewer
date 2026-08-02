#pragma once
#include "solar.h"
#include <chrono>

#include <boatforge/npz_field.h>


// FIXME: Start with a cost calculator with a point-to-point router based on great circle route
// - route planner is a tbd... can run different approaches for this, unclear what is optimal

// FIXME: Each environment model changes different parts of the sim...
// - how to structure...
// FIXME: Think about how to composite this properly (how horizon was meant to work)
struct blackboard
{
    // FIXME: easier to treat as chrono time? tbd, will need to convert to UTC for look ups at somepoint
    std::chrono::seconds time;
    const std::chrono::seconds time_step = std::chrono::hours(1);

    double lat;
    double long;
    double total_traversed_distance = 0.0;

    /* Boat outputs */
    double powered_heading;
    double powered_velocity;

    /* Combined effect from wind, current, etc... */
    double environment_heading;
    double environment_velocity;

    /* This creates our vector */
    double combined_heading;
    double combined_velocity;

    // FIXME: This will eventually be fed by a configuration
    const float surface_area_m = 1.0f;
    float solar_power_in_w;

    /* FIXME: assuming 1 hour steps? tbd... */
    float power_stored_wh; // FIXME: <- Include a starting value...

    float applied_motor_power_out_w;
    float avionics_power_out_w; // <- estimation based on state, this will be used with load shed algorithms, tbd
    // FIXME: power out?
};

class Solver
{
    public:
        Solver(blackboard & bb) : bb_(bb){}

        void step()
        {
            // FIXME: Eventually make this real
            bb_.applied_motor_power_out_w = 500.0;
            bb_.avionics_power_out_w = 100.0;

            // FIXME:
        }
    private:
        blackboard & bb_;
};

/**
 * @brief Boat state - boat motion and internal state
 */
class BoatState
{
    public:
        BoatState (blackboard & bb) : bb_(bb){}

        void step()
        {
            /* May need to split this out into its own class to run before the solver, solver will eventually take power into account */
            bb_.power_stored_wh = (bb_.surface_area_m * bb_.solar_power_in_w) - (bb_.applied_motor_power_out_w + bb_.avionics_power_out_w);

        }
    private:
        blackboard & bb_;
};

class WorldPropogation
{
    public:
        WorldPropogation(blackboard & bb): bb_(bb){}

        void step()
        {
            // FIXME: eventually combine the powered and environment vectors
            bb_.combined_heading = bb_.powered_heading;
            bb_.combined_velocity = bb_.powered_velocity;

            /* meters per second * seconds = meters */
            double step = bb_.combined_velocity * static_cast<double>(bb_.time_step.count());
            bb_.total_traversed_distance += step;

            // FIXME: Combine heading and velocity -> update lat and long
            // bb_.lat = bb_.combined_heading
            // bb_.long = bb_.combined_heading
        }

    private:
        blackboard & bb_;

};



/**
 * Different environment models have different effects
 * - ex: wind + currnet -> propulsion
 *       wave height + period -> risk
 *       sun -> power
 * -
 */
class SolarIsolationField // TODO: Generalize to environment models
{
public:
    // FIXME: This will need to load and sample data...
    SolarIsolationField(blackboard & bb, const std::filesystem::path& path) : bb_(bb), field_(boatforge::NpzField(path)){}

    void sample(){
        // FIXME: Add future optimization to cache data for parallel runs
        bb.solar_power_in_w= field_.sample(bb.time, bb.lat, bb.long);
    }
private:
    blackboard & bb_;
    boatforge::NpzField field_;
};

class EnvironmentField
{
    public:
        EnvironmentField(blackboard & bb) : bb_(bb){}

        void sample();// FIXME: Make this virtual or something...
    private:
        blackboard & bb_;
}
