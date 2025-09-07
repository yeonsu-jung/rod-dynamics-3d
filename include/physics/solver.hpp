#pragma once
#include "rigid_body.hpp"
#include "types.hpp"

void applyImpulse(RigidBody& A, RigidBody& B, const Contact& c);
void positionalCorrection(RigidBody& A, RigidBody& B, const Contact& c, const SolverConfig& cfg);
