#include <opencv2/opencv.hpp>
#include <thread>
#include "ecu/camera.hpp"
#include "ecu/gimbal/gimbal.hpp"
#include "function/auto_aim/detector.hpp"
#include "function/auto_aim/tracker.hpp"
#include "function/auto_aim/aimer.hpp"
#include "tools/thread_safe_queue.hpp"
#include "tools/exiter.hpp"

const std::string keys = "{help h usage ? |     | 输出命令行参数说明 }"
                         "{@config-path c |     | yaml配置文件的路径}";

// 数据包定义
struct FrameData {
  cv::Mat img;
  std::chrono::steady_clock::time_point timestamp;
};

struct DetectionData {
  std::list<xz_vision::Armor> armors;
  std::chrono::steady_clock::time_point timestamp;
};

int main(int argc, char* argv[])
{
  // 1. 初始化所有组件
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);

  tools::Exiter exiter;
  ecu::Camera camera(config_path);
  ecu::Gimbal gimbal(config_path); // 内部已启动接收线程

  xz_vision::Detector detector(config_path);
  xz_vision::Solver solver(config_path);
  xz_vision::Tracker tracker(config_path, solver);
  xz_vision::Aimer aimer(config_path);

  // 2. 初始化队列：PopWhenFull=true 保证实时性
  tools::ThreadSafeQueue<FrameData, true> img_queue(2);
  tools::ThreadSafeQueue<DetectionData, true> detect_queue(2);

  // --- 线程 A: 图像采集 (Producer) ---
  std::thread capture_thread([&]() {
    while (!exiter.exit()) {
      FrameData data;
      camera.read(data.img, data.timestamp);

      if (!data.img.empty()) {
        img_queue.push(data);
      }
    }
  });

  // --- 线程 B: 视觉检测 (Processor) ---
  std::thread detect_thread([&]() {
    while (!exiter.exit()) {
      FrameData frame = img_queue.pop(); // 阻塞等待新图像

      DetectionData det;
      det.armors = detector.detect(frame.img);
      det.timestamp = frame.timestamp;

      detect_queue.push(det);
    }
  });

  // --- 线程 C (主线程): 逻辑解算与串口发送 (Consumer) ---
  while (!exiter.exit()) {
    // 获取检测结果
    DetectionData det = detect_queue.pop();

    // 从 Gimbal 获取当前云台模式，决定是否需要处理
    if (gimbal.mode() == ecu::GimbalMode::IDLE) {
      continue;
    }

    // 跟踪与瞄准
    auto targets = tracker.track(det.armors, det.timestamp);

    // 获取当前弹速并计算指令
    float bullet_speed = gimbal.state().bullet_speed;
    auto command = aimer.aim(targets, det.timestamp, bullet_speed, false);

    // 构造发送给云台的数据包
    ecu::VisionToGimbal tx_data;
    tx_data.pitch = command.pitch;
    tx_data.yaw = command.yaw;
    tx_data.mode = static_cast<uint8_t>(gimbal.mode());

    // 通过串口发送
    gimbal.send(tx_data);
  }

  // 资源回收
  if (capture_thread.joinable())
    capture_thread.join();
  if (detect_thread.joinable())
    detect_thread.join();

  return 0;
}