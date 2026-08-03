#pragma once
#include <chrono>
#include <cmath>
#include <print>

#include <npy_tools/npz_field.h>

namespace
{
/* Mean earth radius (IUGG), the sphere the great-circle propogation assumes */
constexpr double earth_radius_m = 6371008.8;
constexpr double pi = 3.14159265358979323846;

constexpr double deg_to_rad(double deg)
{
    return deg * pi / 180.0;
}
constexpr double rad_to_deg(double rad)
{
    return rad * 180.0 / pi;
}

/* Initial great-circle bearing from (lat1, lon1) to (lat2, lon2), degrees
 * clockwise from true north in [0, 360). Note this is the bearing at the start
 * of the leg only -- it changes continuously along a great circle. */
inline double initial_bearing_deg(double lat1, double lon1, double lat2, double lon2)
{
    const double phi1 = deg_to_rad(lat1);
    const double phi2 = deg_to_rad(lat2);
    /* No need to wrap the delta, sin and cos handle an antimeridian crossing */
    const double delta_lambda = deg_to_rad(lon2 - lon1);

    const double y = std::sin(delta_lambda) * std::cos(phi2);
    const double x = std::cos(phi1) * std::sin(phi2) - std::sin(phi1) * std::cos(phi2) * std::cos(delta_lambda);

    return std::fmod(rad_to_deg(std::atan2(y, x)) + 360.0, 360.0);
}

/* Great-circle distance in meters, haversine on the same sphere the propogation
 * steps along. The atan2 form holds precision for near-antipodal pairs where
 * the asin form does not. */
inline double great_circle_distance_m(double lat1, double lon1, double lat2, double lon2)
{
    const double phi1 = deg_to_rad(lat1);
    const double phi2 = deg_to_rad(lat2);
    const double half_delta_phi = deg_to_rad(lat2 - lat1) / 2.0;
    const double half_delta_lambda = deg_to_rad(lon2 - lon1) / 2.0;

    const double a = std::sin(half_delta_phi) * std::sin(half_delta_phi) +
                     std::cos(phi1) * std::cos(phi2) * std::sin(half_delta_lambda) * std::sin(half_delta_lambda);

    return 2.0 * earth_radius_m * std::atan2(std::sqrt(a), std::sqrt(1.0 - a));
}
}  // namespace

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

    /* Solver may eventually run way points, for now this will be hard coded at init time */
    double end_lat;
    double end_lon;
    /* Meters, great-circle, refreshed each Solver::step() */
    double distance_to_end;

    /* Degrees, WGS84. current_lon is normalized to [-180, 180) after every step */
    double current_lat;
    double current_lon;
    double total_traversed_distance = 0.0;

    /* Boat outputs, headings in degrees clockwise from true north, velocity in m/s */
    double powered_heading;
    double powered_velocity;

    /* Combined effect from wind, current, etc... */
    double environment_heading;
    double environment_velocity;

    /* This creates our vector */
    double combined_heading;
    double combined_velocity;

    // FIXME: This will eventually be fed by a configuration
    const float surface_area_m = 5.0f;
    float solar_power_in_w;

    /* FIXME: assuming 1 hour steps? tbd... */
    float power_stored_wh;  // FIXME: <- Include a starting value...

    float applied_motor_power_out_w;
    float avionics_power_out_w;  // <- estimation based on state, this will be used with load shed algorithms, tbd
    // FIXME: power out?
    uint32_t steps;
    std::chrono::seconds total_time;
};

/**
 * Different environment models have different effects
 * - ex: wind + currnet -> propulsion
 *       wave height + period -> risk
 *       sun -> power
 * -
 */
class SolarIsolationField  // TODO: Generalize to environment models
{
public:
    // FIXME: This will need to load and sample data...
    SolarIsolationField(blackboard& bb, const std::filesystem::path& path)
        : bb_(bb), field_(boatforge::NpzField::load(path))
    {
    }

    void sample()
    {
        // FIXME: Add future optimization to cache data for parallel runs
        bb_.solar_power_in_w = field_.sample(bb_.time, bb_.current_lat, bb_.current_lon);
    }

private:
    blackboard& bb_;
    boatforge::NpzField field_;
};

class EnvironmentField
{
public:
    EnvironmentField(blackboard& bb) : bb_(bb)
    {
    }

    void sample();  // FIXME: Make this virtual or something...
private:
    blackboard& bb_;
};

class Solver
{
public:
    Solver(blackboard& bb) : bb_(bb)
    {
    }

    void step()
    {
        /* Re-aim at the destination every step: on a sphere the great-circle
         * bearing changes as we move, so a heading fixed at t0 would sail a
         * rhumb line instead. */
        bb_.powered_heading = initial_bearing_deg(bb_.current_lat, bb_.current_lon, bb_.end_lat, bb_.end_lon);
        bb_.powered_velocity = 2.0;  // FIXME: Make this real F(power), 2ms ~4 knots
        bb_.distance_to_end = great_circle_distance_m(bb_.current_lat, bb_.current_lon, bb_.end_lat, bb_.end_lon);

        // FIXME: Eventually make this real
        bb_.applied_motor_power_out_w = 500.0;
        bb_.avionics_power_out_w = 100.0;

        // FIXME:
    }

private:
    blackboard& bb_;
};

/**
 * @brief Boat state - boat motion and internal state
 */
class BoatState
{
public:
    BoatState(blackboard& bb) : bb_(bb)
    {
    }

    void step()
    {
        /* May need to split this out into its own class to run before the solver, solver will eventually take power
         * into account */
        bb_.solar_power_in_w = bb_.surface_area_m * bb_.solar_power_in_w;
        bb_.power_stored_wh += (bb_.solar_power_in_w) - (bb_.applied_motor_power_out_w + bb_.avionics_power_out_w);
    }

private:
    blackboard& bb_;
};

class WorldPropogation
{
public:
    WorldPropogation(blackboard& bb) : bb_(bb)
    {
    }

    void step()
    {
        // FIXME: eventually combine the powered and environment vectors
        bb_.combined_heading = bb_.powered_heading;
        bb_.combined_velocity = bb_.powered_velocity;

        /* meters per second * seconds = meters */
        double step = bb_.combined_velocity * static_cast<double>(bb_.time_step.count());
        bb_.total_traversed_distance += step;

        /* Great-circle destination point: walk `step` meters from the current
         * position along the combined heading, held constant over the step. */
        const double angular_step = step / earth_radius_m;
        const double lat_rad = deg_to_rad(bb_.current_lat);
        const double lon_rad = deg_to_rad(bb_.current_lon);
        const double bearing_rad = deg_to_rad(bb_.combined_heading);

        const double sin_lat_next = std::sin(lat_rad) * std::cos(angular_step) +
                                    std::cos(lat_rad) * std::sin(angular_step) * std::cos(bearing_rad);
        const double lat_next = std::asin(sin_lat_next);
        const double lon_next = lon_rad + std::atan2(std::sin(bearing_rad) * std::sin(angular_step) * std::cos(lat_rad),
                                                     std::cos(angular_step) - std::sin(lat_rad) * sin_lat_next);

        bb_.current_lat = rad_to_deg(lat_next);
        /* remainder wraps into [-180, 180], keeping the field lookups in range
         * when a track crosses the antimeridian */
        bb_.current_lon = std::remainder(rad_to_deg(lon_next), 360.0);
    }

private:
    blackboard& bb_;
};

class Info
{
public:
    Info(blackboard& bb) : bb_(bb)
    {
    }

    void step()
    {
        /* One line per step, the sim's running commentary: where we are, when
         * we are, how far we have come and how far is left. */
        // std::println(
        //     "[{:%F %T}] lat {:9.4f}  lon {:9.4f}  travelled {:10.1f} km  remaining {:10.1f} km, step = {}, total
        //     hours "
        //     "{}, total charge {5.5f}",
        //     std::chrono::sys_seconds{bb_.time},  // TBD what to do with time...
        //     bb_.current_lat, bb_.current_lon, bb_.total_traversed_distance / 1000.0, bb_.distance_to_end / 1000.0,
        //     bb_.steps, std::chrono::duration_cast<std::chrono::hours>(bb_.total_time).count(), bb_.power_stored_wh);
    }

private:
    /* A reference, not a copy: a copy would freeze the values taken at
     * construction and log the same line every step. */
    blackboard& bb_;
};
