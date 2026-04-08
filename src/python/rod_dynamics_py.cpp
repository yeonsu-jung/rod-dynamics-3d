#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <stdexcept>
#include <string>

#include "app/python_api.hpp"

namespace py = pybind11;

namespace {

std::vector<double> vec3ToList(const glm::vec3 &value) {
  return {static_cast<double>(value.x), static_cast<double>(value.y),
          static_cast<double>(value.z)};
}

std::vector<double> quatToList(const glm::quat &value) {
  return {static_cast<double>(value.w), static_cast<double>(value.x),
          static_cast<double>(value.y), static_cast<double>(value.z)};
}

py::dict rigidBodyToDict(const RigidBody &body) {
  py::dict result;
  result["position"] = vec3ToList(body.x);
  result["orientation_wxyz"] = quatToList(body.q);
  result["linear_velocity"] = vec3ToList(body.v);
  result["angular_velocity"] = vec3ToList(body.w);
  result["force"] = vec3ToList(body.f);
  result["torque"] = vec3ToList(body.tau);
  result["mass"] = body.mass;
  result["inverse_mass"] = body.invMass;
  result["friction"] = body.friction;
  result["restitution"] = body.restitution;
  result["shape"] = body.type == ShapeType::Sphere ? "sphere" : "capsule";
  if (body.type == ShapeType::Sphere) {
    result["radius"] = body.sphere.r;
  } else {
    result["radius"] = body.cap.r;
    result["half_height"] = body.cap.h;
    glm::vec3 endpoint_a(0.0f);
    glm::vec3 endpoint_b(0.0f);
    body.capsuleEndpoints(endpoint_a, endpoint_b);
    result["endpoint_a"] = vec3ToList(endpoint_a);
    result["endpoint_b"] = vec3ToList(endpoint_b);
  }
  return result;
}

class PythonSimulator {
public:
  PythonSimulator(const std::string &scenePath,
                  const std::string &initCsvPath = std::string(),
                  bool quiet = true)
      : app_(app::createPythonApp(scenePath, initCsvPath, quiet),
             &app::destroyPythonApp) {}

  void step(int steps = 1) { app::stepPythonSession(app_.get(), steps); }

  py::dict diagnostics() const {
    py::dict result;
    result["frame_index"] = py::int_(app::pythonFrameIndex(app_.get()));
    result["dt"] = app::pythonDt(app_.get());
    result["last_ke"] = app::pythonLastKE(app_.get());
    result["last_hit_count"] = py::int_(app::pythonLastHitCount(app_.get()));
    result["last_island_count"] =
        py::int_(app::pythonLastIslandCount(app_.get()));
    result["rod_count"] = py::int_(app::pythonRods(app_.get()).size());
    return result;
  }

  std::vector<py::dict> rods() const {
    std::vector<py::dict> result;
    const auto &rods = app::pythonRods(app_.get());
    result.reserve(rods.size());
    for (const auto &rod : rods) {
      result.push_back(rigidBodyToDict(rod));
    }
    return result;
  }

private:
  std::unique_ptr<App, void (*)(App *)> app_;
};

} // namespace

PYBIND11_MODULE(rod_dynamics_py, m) {
  m.doc() = "Python bindings for one-step rod dynamics evolution";

  py::class_<PythonSimulator>(m, "Simulator")
      .def(py::init<const std::string &, const std::string &, bool>(),
           py::arg("scene_path"), py::arg("init_csv_path") = std::string(),
           py::arg("quiet") = true)
      .def("step", &PythonSimulator::step, py::arg("steps") = 1)
      .def("diagnostics", &PythonSimulator::diagnostics)
      .def("rods", &PythonSimulator::rods);
}