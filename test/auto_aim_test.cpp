#include <opencv2/opencv.hpp>
#include <fmt/core.h>

#include "ecu/camera.hpp"
#include "function/auto_aim/detector.hpp"
#include "function/auto_aim/solver.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"
using namespace std;
using namespace xz_vision;

const std::string keys =
    "{help h usage ? |     | 输出命令行参数说明 }"
    "{@config-path c | /home/chaichai/project/rm_xz_vision26/configs/how_to_set_params.yaml | "
    "yaml配置文件的路径}";

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

  // --- 定义 FPS 统计变量 ---
  int frame_count = 0;            // 帧计数器
  double total_time = 0;          // 累积耗时
  const int stats_interval = 100; // 统计间隔（100帧）

  while (!exiter.exit()) {
    // 1. 计时开始
    auto start = std::chrono::steady_clock::now();

    cv::Mat raw_img;
    std::chrono::steady_clock::time_point timestamp;

    camera.read(raw_img, timestamp);

    auto t_now = std::chrono::steady_clock::now();

    auto armors = detector.detect(raw_img); // 直接调用 detect 方法

    for (auto& armor : armors) {
      solver.solve(armor);

      if (armor.xyz_in_world.norm() > 0.1 && armor.xyz_in_world.norm() < 10.0) {
        // 1. 格式化字符串
        std::string text =
            fmt::format("Pos:({:.2f},{:.2f},{:.2f})m, Dist:{:.2f}m", armor.xyz_in_world[0],
                        armor.xyz_in_world[1], armor.xyz_in_world[2], armor.xyz_in_world.norm());

        // 2. 打印到日志
        tools::logger()->info("{}", text);

        // 3. 绘制到图像
        cv::Point text_pos(10, 30);

        // 绘制文字 (参数：图像, 内容, 位置, 字体, 缩放, 颜色, 粗细)
        cv::putText(raw_img, text, text_pos, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0),
                    2);
      }
    }

    // 别忘了在循环最后显示图像，否则你看不到结果
    cv::imshow("Vision Debug", raw_img);
    auto key = cv::waitKey(1);
    if (key == 'q')
      break;
  }
  return 0;
}