#pragma once

class BoatModel
{

    // field.sample(t, lat, lon)            — environment (radiation, later currents/wind)
    // solve(state, env, value_fn)          — the decision
    // - this is the control algorithm
    // advance(state, request, env)         — dynamics (the boat)
    // - this is the boat control estimator
    // propagate(state, velocity, env, dt)  — kinematics (drift and currents)
    // - this is the world effect estimator (mostly motion related? - though currents should be factored in... tbd)
    //
};
