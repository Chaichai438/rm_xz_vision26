#include "function/auto_aim/solver.hpp"
#include "function/auto_aim/tracker.hpp"
#include "ecu/camera.hpp"
#include "function/auto_aim/detector.hpp"
#include "tools/exiter.hpp"
#include "tools/plotter.hpp"

#include <iostream>
#include <unistd.h>
#include <chrono>

using namespace xz_vision;

void draw_armor_points(cv::Mat& frame, const std::vector<cv::Point2f>& points,
                       const std::string& label);

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

  // ecu::Camera camera(config_path);

  cv::VideoCapture cap("/home/chaichai/project/rm_xz_vision26/assets/armor.avi");

  xz_vision::Detector detector(config_path);
  xz_vision::Solver solver(config_path);
  // xz_vision::Tracker tracker(config_path);

  tools::Exiter exiter;
  tools::Plotter plotter;

  // --- 定义 FPS 统计变量 ---
  int frame_count = 0;            // 帧计数器
  double total_time = 0;          // 累积耗时
  const int stats_interval = 100; // 统计间隔（100帧）

  while (!exiter.exit()) {
    cv::Mat raw_img;
    std::chrono::steady_clock::time_point timestamp;

    cap >> raw_img;
    // camera.read(raw_img, timestamp);
    if (raw_img.empty())
      continue;

    // 1. 计时开始
    auto start = std::chrono::steady_clock::now();

    // 2. 算法处理
    auto armors = detector.detect_onnx(raw_img);

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

        nlohmann::json data;
        const auto& armor = armors.front();
        data["armor_x"] = armor.xyz_in_world[0];
        data["armor_y"] = armor.xyz_in_world[1];
        data["armor_yaw"] = armor.ypr_in_world[0] * 57.3;
        data["armor_yaw_raw"] = armor.yaw_raw * 57.3;
        data["armor_center_x"] = armor.center_norm.x;
        data["armor_center_y"] = armor.center_norm.y;
      }
    }

    // 3. 计时结束
    auto end = std::chrono::steady_clock::now();

    // 4. 统计逻辑
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    total_time += duration.count(); // 累加微秒
    frame_count++;

    // 5. 每隔 100 帧计算并输出一次
    if (frame_count >= stats_interval) {
      double avg_ms = (total_time / stats_interval) / 1000.0; // 100帧平均耗时（毫秒）
      double avg_fps = 1000.0 / avg_ms;                       // 平均FPS

      std::cout << "[INFO] Avg Time (last 100 frames): " << std::fixed << std::setprecision(2)
                << avg_ms << " ms | Avg FPS: " << avg_fps << std::endl;

      // 重置计数器
      frame_count = 0;
      total_time = 0;
    }

    auto key = cv::waitKey(0);
    if (key == 'q')
      break;

    //   // ========== 简洁绘制逻辑 ==========

    //   if () {
    //     const auto& armor = solver.target;

    //     // 1. 绘制装甲板 4 个角点 (核心需求)
    //     draw_armor_points(frame, armor.points, "ARMOR");

    //     // 2. 绘制边界框
    //     cv::rectangle(frame, armor.box, cv::Scalar(0, 255, 0), 1);

    //     // 3. 绘制准心
    //     cv::drawMarker(frame, cv::Point(cfg.screen_center_x, cfg.screen_center_y),
    //                    cv::Scalar(0, 255, 255), cv::MARKER_CROSS, 15, 2);

    //     // 4. 绘制信息文本
    //     std::string info = ARMOR_NAME[armor.name] + " " + COLOR[armor.color] + " (" +
    //                        std::to_string((int)(armor.confidence * 100)) + "%)";
    //     cv::putText(frame, info, armor.box.tl() + cv::Point(0, -5), cv::FONT_HERSHEY_SIMPLEX,
    //     0.6,
    //                 cv::Scalar(0, 255, 0), 2);
    //   } else {
    //     // 未检测到目标时，可选：绘制全图灯条调试 (如果需要)
    //     // const auto& lbs = solver.get_current_lightbars();
    //     // for(const auto& lb : lbs) cv::circle(frame, lb.center, 2, cv::Scalar(0,0,255), -1);
    //   }

    //   cv::imshow("Vision", frame);
    //   if (cv::waitKey(1) == 'q')
    //     break;
    // }

    return 0;
  }
}
// ✅ 简洁版绘制函数：直接绘制 vector<Point2f>
void draw_armor_points(cv::Mat& frame, const std::vector<cv::Point2f>& points,
                       const std::string& label = "")
{
  if (points.size() < 4)
    return;

  cv::Scalar color(0, 255, 0); // 绿色
  int thickness = 2;

  // 1. 绘制轮廓连线
  for (int i = 0; i < 4; i++) {
    cv::line(frame, points[i], points[(i + 1) % 4], color, thickness);
  }

  // 2. 绘制角点和标签 (一次性循环)
  const char* suffixes[4] = {"TL", "TR", "BR", "BL"};
  for (int i = 0; i < 4; i++) {
    // 画实心圆点
    cv::circle(frame, points[i], 4, color, -1);

    // 画序号标签 (可选)
    if (!label.empty()) {
      std::string txt = label + "-" + suffixes[i];
      cv::putText(frame, txt, points[i] + cv::Point2f(8, -8), cv::FONT_HERSHEY_SIMPLEX, 0.4, color,
                  1);
    }
  }

  // 3. 绘制中心点
  cv::Point2f center = (points[0] + points[2]) / 2.0f;
  cv::circle(frame, center, 3, cv::Scalar(0, 0, 255), -1); // 红色中心
}