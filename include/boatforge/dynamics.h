#pragma once
#include <chrono>

// FIXME: Start with a cost calculator with a point-to-point router based on great circle route
// - route planner is a tbd... can run different approaches for this, unclear what is optimal

// FIXME: Each environment model changes different parts of the sim...
// - how to structure...
// FIXME: Think about how to composite this properly (how horizon was meant to work)
struct blackboard
{
    // FIXME: easier to treat as chrono time? tbd, will need to convert to UTC for look ups at somepoint
    std::chrono::seconds time;

    float current_lat;
    float current_long;

    float current_heading;
    float current_velocity;

    float solar_power_in;

};

class Solver
{

};

/**
 * @brief Boat dynamics covers
 */
class BoatDynamics
{

};

class WorldDynamics
{

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
    SolarIsolationField(blackboard & bb) : bb_(bb){}

    void sample(){
        // FIXME: Add future optimization to cache data for parallel runs
        // FIXME: likely will want to interpolate data here
        // read_file(bb_.time, bb_.current_lat, bb_.current_long);
    }
private:
    blackboard & bb_;
};

class EnvironmentField
{
    public:
        EnvironmentField(blackboard & bb) : bb_(bb){}

        void sample();// FIXME: Make this virtual or something...
    private:
        blackboard & bb_;
}
