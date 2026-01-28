#include <opencv2/opencv.hpp>
#include <fmt/core.h>

#include "ecu/camera.hpp"
#include "function/auto_aim/detector.hpp"
#include "function/auto_aim/solver.hpp"
#include "function/auto_aim/tracker.hpp"
#include "function/auto_aim/DecisionMaker.hpp"

#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/plotter.hpp"

const std::string keys = "{help h usage ? |     | 输出命令行参数说明 }"
                         "{@config-path c |     | yaml配置文件的路径}";

int main(int argc, char* argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  auto config_path = cli.get<std::string>(0);

  tools::Exiter exiter;
  ecu::Camera camera(config_path);
  xz_vision::Detector detector(config_path);
  xz_vision::Solver solver(config_path);
  xz_vision::Tracker tracker(config_path, solver);
  xz_vision::DecisionMaker decision_maker(config_path);

  tools::Plotter plotter;

  auto timestamp = std::chrono::steady_clock::now();
  cv::Mat img;
  while (!exiter.exit()) {
    camera.read(img, timestamp);

    auto yolo_start = std::chrono::steady_clock::now();
    auto armors = detector.detect(img);

    auto tracker_start = std::chrono::steady_clock::now();
    auto targets = tracker.track(armors, timestamp);

    auto aimer_start = std::chrono::steady_clock::now();
  }
}