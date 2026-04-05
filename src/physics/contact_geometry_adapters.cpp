#include "physics/contact_geometry_adapters.hpp"

CommonContactGeometry toCommonContactGeometry(const ContactPrimitive& contact) {
  CommonContactGeometry out;
  out.bodyA = contact.body_a;
  out.bodyB = contact.body_b;
  out.pointA = contact.point_a;
  out.pointB = contact.point_b;
  out.normal = contact.normal;
  out.distance = contact.distance;
  out.surfaceLimit = contact.surface_limit;
  out.signedGap = contact.distance - contact.surface_limit;
  out.shiftB = contact.shift_b;
  return out;
}

CommonContactGeometry toCommonContactGeometry(
    const MujocoContactSolver::Contact& contact) {
  CommonContactGeometry out;
  out.bodyA = contact.a;
  out.bodyB = contact.b;
  out.pointA = contact.pA;
  out.pointB = contact.pB;
  out.normal = contact.n;
  out.distance = contact.dist;
  out.surfaceLimit = contact.surface_limit;
  out.signedGap = contact.dist - contact.surface_limit;
  return out;
}
