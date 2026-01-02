
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// Helper to split string by delimiter
std::vector<std::string> split(const std::string &s, char delimiter) {
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream tokenStream(s);
  while (std::getline(tokenStream, token, delimiter)) {
    tokens.push_back(token);
  }
  return tokens;
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " <input_perrod.csv> <output_endpoints.csv> [rod_length=1.0]"
              << std::endl;
    return 1;
  }

  std::string inputPath = argv[1];
  std::string outputPath = argv[2];
  float rodLength = 1.0f;
  if (argc >= 4) {
    rodLength = std::stof(argv[3]);
  }

  std::ifstream inFile(inputPath);
  if (!inFile.is_open()) {
    std::cerr << "Error: Could not open input file " << inputPath << std::endl;
    return 1;
  }

  std::ofstream outFile(outputPath);
  if (!outFile.is_open()) {
    std::cerr << "Error: Could not open output file " << outputPath
              << std::endl;
    return 1;
  }

  // Write output header
  outFile << "frame,rod,x1,y1,z1,x2,y2,z2"
          << "\n";

  std::string line;
  int lineNum = 0;
  while (std::getline(inFile, line)) {
    lineNum++;
    // Skip comments and empty lines
    if (line.empty() || line[0] == '#')
      continue;

    std::vector<std::string> tokens = split(line, ',');

    // Check for header line (simple heuristic: first token contains alpha)
    if (!tokens.empty() && !tokens[0].empty() && !isdigit(tokens[0][0]) &&
        tokens[0][0] != '-') {
      // likely header, skip
      continue;
    }

    if (tokens.size() < 15) {
      // std::cerr << "Warning: Skipping line " << lineNum << " (not enough
      // columns)" << std::endl;
      continue;
    }

    try {
      int frame = std::stoi(tokens[0]);
      int rod = std::stoi(tokens[1]);

      float px = std::stof(tokens[2]);
      float py = std::stof(tokens[3]);
      float pz = std::stof(tokens[4]);

      // tokens[11] is qw, [12] qx, [13] qy, [14] qz
      float qw = std::stof(tokens[11]);
      float qx = std::stof(tokens[12]);
      float qy = std::stof(tokens[13]);
      float qz = std::stof(tokens[14]);

      glm::vec3 pos(px, py, pz);
      glm::quat q(qw, qx, qy, qz);

      // Normalize quaternion just in case
      q = glm::normalize(q);

      // Rod axis is Local Y (0, 1, 0) based on analysis
      glm::vec3 axis = q * glm::vec3(0.0f, 1.0f, 0.0f);

      glm::vec3 p1 = pos - (axis * (rodLength * 0.5f));
      glm::vec3 p2 = pos + (axis * (rodLength * 0.5f));

      outFile << frame << "," << rod << "," << p1.x << "," << p1.y << ","
              << p1.z << "," << p2.x << "," << p2.y << "," << p2.z << "\n";

    } catch (const std::exception &e) {
      std::cerr << "Error parsing line " << lineNum << ": " << e.what()
                << std::endl;
    }
  }

  std::cout << "Successfully converted " << inputPath << " to " << outputPath
            << std::endl;

  inFile.close();
  outFile.close();
  return 0;
}
