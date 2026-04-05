#pragma once

#include "physics/contact_geometry.hpp"
#include "physics/mujoco_contact.hpp"
#include "physics/soft_contact.hpp"

CommonContactGeometry toCommonContactGeometry(const ContactPrimitive& contact);

CommonContactGeometry toCommonContactGeometry(
    const MujocoContactSolver::Contact& contact);
