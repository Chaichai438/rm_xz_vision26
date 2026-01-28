#include "ecu/gimbal/gimbal.hpp"
#include "tools/plotter.hpp"

#include <iostream>
#include <chrono>
#include <thread>
#include <iomanip>
#include <opencv2/opencv.hpp>

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

  std::cout << "--- Gimbal Communication Test Start ---" << std::endl;
  std::cout << "Testing Protocol: Tx (14 bytes), Rx (17 bytes)" << std::endl;

  // 2. 构造模拟的视觉目标数据
  ecu::VisionToGimbal test_data;
  test_data.mode = 1;       // 进入自瞄模式 (对应发送 0xFC)
  test_data.pitch = 12.34f; // 模拟 Pitch 目标角度
  test_data.yaw = -45.67f;  // 模拟 Yaw 目标角度

  int count = 0;
  while (true) {
    // A. 发送数据给 STM32 (频率约 100Hz)
    gimbal.send(test_data);

    // B. 获取并打印 STM32 反馈的数据 (来自 read_thread)
    auto state = gimbal.state();
    auto mode = gimbal.mode();

    // 使用 fixed 和 setprecision 让输出更整齐
    std::cout << "\r[Count: " << std::setw(5) << count++ << "] "
              << "Mode: " << gimbal.str(mode) << " | "
              << "MCU_Pitch: " << std::fixed << std::setprecision(2) << std::setw(7) << state.pitch
              << " | "
              << "MCU_Yaw: " << std::fixed << std::setprecision(2) << std::setw(7) << state.yaw
              << " | "
              << "BulletSpeed: " << state.bullet_speed << " m/s" << std::flush;

    // C. 动态改变一些测试数据，观察 STM32 是否有反应
    test_data.yaw += 0.01f;
    if (test_data.yaw > 180.0f)
      test_data.yaw = -180.0f;

    // 延时 10ms (100Hz)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  return 0;
}