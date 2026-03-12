#include "ecu/gimbal/gimbal.hpp"
#include "tools/plotter.hpp"

#include <iostream>
#include <chrono>
#include <thread>
#include <iomanip>
#include <opencv2/opencv.hpp>

// 统计频率的辅助函数
double get_fps()
{
  static auto last_t = std::chrono::steady_clock::now();
  auto now = std::chrono::steady_clock::now();
  double dt = std::chrono::duration<double>(now - last_t).count();
  last_t = now;
  return 1.0 / dt;
}

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

  if (argc > 1)
    config_path = argv[1];

  ecu::Gimbal gimbal(config_path);
  tools::Plotter plotter;
  nlohmann::json data;

  std::cout << "\n--- Gimbal Communication Latency Test ---" << std::endl;
  std::cout << "Target Frequency: 100Hz | Reading mode: Real-time Update" << std::endl;

  auto start_time = std::chrono::steady_clock::now();

  ecu::VisionToGimbal vtg;
  vtg.pitch = 0.0f;
  vtg.yaw = 0.1f;

  while (true) {
    gimbal.send(vtg);
    auto state = gimbal.state();
    auto mode = gimbal.mode();

    // 统计实际循环频率 (如果这个值远低于 100，说明 plotter 或 cout 太慢)
    double current_fps = get_fps();

    data["yaw_gimbal"] = state.yaw;
    data["pitch_gimbal"] = state.pitch;
    data["mode"] = static_cast<int>(mode);
    plotter.plot(data);

    static float last_yaw = 0;
    std::string status = (std::abs(state.yaw - last_yaw) < 0.0001f) ? " [STALE]" : " [LIVE]";
    last_yaw = state.yaw;

    std::cout << "\r"
              << "Hz: " << std::setw(3) << static_cast<int>(current_fps) << " | P: " << std::fixed
              << std::setprecision(2) << std::setw(6) << state.pitch << " | Y: " << std::fixed
              << std::setprecision(2) << std::setw(6) << state.yaw
              << " | Speed: " << state.bullet_speed << status << "    " << std::flush;

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    if (cv::waitKey(1) == 27)
      break; // 按 ESC 退出
  }

  return 0;
}
