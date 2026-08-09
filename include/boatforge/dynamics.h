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
// FIXME: load off of yaml instead of recompiling shit
struct blackboard
{
    // FIXME: easier to treat as chrono time? tbd, will need to convert to UTC for look ups at somepoint
    std::chrono::seconds time{};
    const std::chrono::seconds time_step = std::chrono::hours(1);

    /* Solver may eventually run way points, for now this will be hard coded at init time */
    double end_lat{};
    double end_lon{};
    /* Meters, great-circle, refreshed each Solver::step() */
    double distance_to_end{};
    /* Terminate the run if we are less than 10 km from the end */
    const double termination_distance = 10000.0f;

    /* Degrees, WGS84. current_lon is normalized to [-180, 180) after every step */
    double current_lat{};
    double current_lon{};
    double total_traversed_distance = 0.0;

    /* Boat outputs, headings in degrees clockwise from true north, velocity in m/s */
    double powered_heading{};
    double powered_velocity{};

    /* Combined effect from wind, current, etc... */
    double environment_heading{};
    double environment_velocity{};

    double ocean_current_heading{};
    double ocean_current_velocity{};

    /* This creates our vector */
    double combined_heading{};
    double combined_velocity{};

    /* Cleared by any model that finds it has nothing to step on -- for now only
     * a field sample off the end of its coverage. The sim ends that run where
     * the flag drops, rather than carrying a NaN through the rest of the
     * track. */
    bool data_valid = true;

    // FIXME: This will eventually be fed by a configuration
    const float surface_area_m = 5.0f;
    float solar_power_in_w{};

    /* FIXME: assuming 1 hour steps? tbd... */
    const float max_power_storage_wh = 2000.0f;
    float power_stored_wh = 1000.0;  // FIXME: <- Include a starting value...

    float applied_motor_power_out_w{};
    float avionics_power_out_w{};  // <- estimation based on state, this will be used with load shed algorithms, tbd
    // FIXME: power out?
    uint32_t steps{};
    std::chrono::seconds total_time{};
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
    SolarIsolationField(blackboard& bb, boatforge::NpzField& field) : bb_(bb), field_(field)
    {
    }

    void sample()
    {
        // FIXME: Add future optimization to cache data for parallel runs
        /* On a miss solar_power_in_w keeps its last value, which nothing reads:
         * the sim ends the run on the cleared flag before the step that would
         * have used it. */
        bb_.data_valid &= field_.sample(bb_.time, bb_.current_lat, bb_.current_lon, bb_.solar_power_in_w);
    }

private:
    blackboard& bb_;
    boatforge::NpzField& field_;
};

class OceanCurrentField  // TODO: Generalize to environment models
{
public:
    // FIXME: This will need to load and sample data...
    OceanCurrentField(blackboard& bb, boatforge::NpzField& uo_field, boatforge::NpzField& vo_field)
        : bb_(bb), uo_field_(uo_field), vo_field_(vo_field)
    {
    }

    void sample()
    {
        // FIXME: Add future optimization to cache data for parallel runs
        /* Both are sampled even if the first misses: they are separate files
         * and only agree on coverage by construction, so asking each is the
         * only way to know the pair is usable. On a miss the heading and
         * velocity keep their last values, which nothing reads -- the sim ends
         * the run on the cleared flag before the step that would have used
         * them. */
        float uo{};
        float vo{};
        const bool have_uo = uo_field_.sample(bb_.time, bb_.current_lat, bb_.current_lon, uo);
        const bool have_vo = vo_field_.sample(bb_.time, bb_.current_lat, bb_.current_lon, vo);

        bb_.data_valid &= have_uo && have_vo;
        if (!have_uo || !have_vo)
        {
            return;
        }

        /* Components first, polar second -- never the other way round. Each
         * field interpolates its own component, and the pair is turned into a
         * heading and a speed here, once, from the result.
         *
         * Interpolating a stored speed and bearing instead would be wrong at
         * every point the flow turns. Across a shear where u runs +1 -> -1 the
         * true midpoint is slack water, but interpolated speed stays 1 m/s and
         * the bearing has two equally short arcs to choose between; a stored
         * bearing also has a seam at 0/360 that linear interpolation puts 180
         * degrees out. The components have no seam and pass through zero
         * correctly, so the vector below is the real one. */
        const double u = static_cast<double>(uo);
        const double v = static_cast<double>(vo);

        /* hypot rather than sqrt(u*u + v*v): it holds precision at the ends of
         * the range instead of squaring its way into an overflow or a
         * denormal. */
        bb_.ocean_current_velocity = std::hypot(u, v);

        /* The set: where the current flows *to*, which is the direction that
         * carries the boat. (Wind is named the opposite way, by where it comes
         * from -- do not carry this convention across to a wind field.)
         *
         * atan2(u, v), east over north, rather than the usual atan2(y, x):
         * that puts 0 at true north and turns clockwise through east, which is
         * the convention powered_heading and initial_bearing_deg already use.
         * Slack water leaves this at 0 -- a heading is meaningless with no
         * speed behind it, and every consumer weights it by the velocity. */
        bb_.ocean_current_heading = std::fmod(rad_to_deg(std::atan2(u, v)) + 360.0, 360.0);

        // TODO: Add other effects
        bb_.environment_heading = bb_.ocean_current_heading;
        bb_.environment_velocity = bb_.ocean_current_velocity;
    }

private:
    blackboard& bb_;
    boatforge::NpzField& uo_field_;
    boatforge::NpzField& vo_field_;
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
        if (bb_.power_stored_wh <= 0.0f)
        {
            bb_.applied_motor_power_out_w = 0.0;
            bb_.avionics_power_out_w = 0.0;
        }
        else if (bb_.power_stored_wh <= 600.0f)
        {
            bb_.applied_motor_power_out_w = 0.0;
            bb_.avionics_power_out_w = 25.0;
        }
        else
        {
            bb_.applied_motor_power_out_w = 500.0;
            bb_.avionics_power_out_w = 100.0;
        }

        // FIXME: eventually this should be based on sensor data
        bb_.powered_heading = initial_bearing_deg(bb_.current_lat, bb_.current_lon, bb_.end_lat, bb_.end_lon);

        bb_.powered_velocity =
            3.0 * (bb_.applied_motor_power_out_w / 500.0);  // FIXME: Make this real F(power), 2ms ~4 knots
        bb_.distance_to_end = great_circle_distance_m(bb_.current_lat, bb_.current_lon, bb_.end_lat, bb_.end_lon);
        // FIXME: Eventually make this real

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

        if (bb_.power_stored_wh >= bb_.max_power_storage_wh)
        {
            bb_.power_stored_wh = bb_.max_power_storage_wh;
        }
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
        /* What the boat drives and what the world carries it through are two
         * velocities over the ground, so they add as vectors -- through their
         * components, never by averaging the two headings. An average ignores
         * which of the two is stronger, and it carries the 0/360 seam: a boat
         * on 350 in a current setting 010 would come out on 180, pointing back
         * the way it came.
         *
         * The decomposition is the same one OceanCurrentField uses coming the
         * other way. A compass bearing puts east on sin and north on cos, so
         * the pair below is (east, north) in m/s. */
        const double powered_rad = deg_to_rad(bb_.powered_heading);
        const double environment_rad = deg_to_rad(bb_.environment_heading);

        const double east =
            bb_.powered_velocity * std::sin(powered_rad) + bb_.environment_velocity * std::sin(environment_rad);
        const double north =
            bb_.powered_velocity * std::cos(powered_rad) + bb_.environment_velocity * std::cos(environment_rad);

        bb_.combined_velocity = std::hypot(east, north);
        /* atan2(east, north) rather than the usual atan2(y, x): 0 at true
         * north, turning clockwise, which is the convention every heading on
         * the blackboard carries. Dead in the water leaves this at 0 and
         * nothing minds -- the step below is scaled by the velocity, so a
         * heading with no speed behind it moves the boat nowhere.
         *
         * Note this can be *slower* than powered_velocity, and should be: a
         * current on the nose subtracts, and a beam-on current turns the track
         * off the heading the solver asked for. Both fall straight out of the
         * addition. */
        bb_.combined_heading = std::fmod(rad_to_deg(std::atan2(east, north)) + 360.0, 360.0);

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
