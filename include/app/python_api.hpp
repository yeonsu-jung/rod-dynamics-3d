#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "physics/rigid_body.hpp"

class App;

namespace app {

App *createPythonApp(const std::string &scenePath,
                     const std::string &initCsvPath = std::string(),
                     bool quiet = true);

void destroyPythonApp(App *appInstance);
void stepPythonSession(App *appInstance, int steps = 1);

const std::vector<RigidBody> &pythonRods(const App *appInstance);
uint64_t pythonFrameIndex(const App *appInstance);
double pythonLastKE(const App *appInstance);
size_t pythonLastHitCount(const App *appInstance);
size_t pythonLastIslandCount(const App *appInstance);
float pythonDt(const App *appInstance);

} // namespace app