#include <nlohmann/json.hpp>
#include "tools/plotter.hpp"
#include "tools/exiter.hpp"

int main()
{
  tools::Plotter plotter;
  tools::Exiter exiter;
  nlohmann::json data;
  int num = 5;

  while (!exiter.exit()) {
    data["x"] = num;
    data["y"] = 2 * num;
    num++;
    plotter.plot(data);
  }

  return 0;
}