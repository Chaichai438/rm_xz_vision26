#include <opencv2/opencv.hpp>
#include <fmt/core.h>

#include "ecu/camera.hpp"
#include "ecu/gimbal/gimbal.hpp"

#include "function/auto_aim/detector.hpp"
#include "function/auto_aim/solver.hpp"
#include "function/auto_aim/tracker.hpp"
#include "function/auto_aim/aimer.hpp"

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

  ecu::Gimbal gimbal(config_path);

  xz_vision::Detector detector(config_path);
  xz_vision::Solver solver(config_path);
  xz_vision::Tracker tracker(config_path, solver);
  xz_vision::Aimer aimer(config_path);

  tools::Plotter plotter;

  auto timestamp = std::chrono::steady_clock::now();
  cv::Mat img;
  while (!exiter.exit()) {
    camera.read(img, timestamp);
    if (img.empty())
      continue;

    auto detect_start = std::chrono::steady_clock::now();
    auto armors = detector.detect(img);

    auto tracker_start = std::chrono::steady_clock::now();
    auto targets = tracker.track(armors, timestamp);

    auto aimer_start = std::chrono::steady_clock::now();
    auto command = aimer.aim(targets, timestamp, 27, false);

    // 构造发送给云台的数据包
    ecu::VisionToGimbal tx_data;
    tx_data.pitch = command.pitch;
    tx_data.yaw = command.yaw;
    tx_data.mode = static_cast<uint8_t>(gimbal.mode());
    gimbal.send(tx_data);
  }

  return 0;
}