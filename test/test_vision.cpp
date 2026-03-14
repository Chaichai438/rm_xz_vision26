#include "function/auto_aim/detector.hpp"
#include "function/auto_aim/solver.hpp"
#include "function/auto_aim/target.hpp"
#include "function/auto_aim/tracker.hpp"
#include "function/auto_aim/aimer.hpp"

#include "ecu/camera.hpp"
#include "ecu/gimbal/gimbal.hpp"

#include "tools/logger.hpp"
#include "tools/plotter.hpp"
#include "tools/exiter.hpp"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <fmt/core.h>
#include <nlohmann/json.hpp>

using namespace xz_vision;

// 辅助函数：绘制检测结果
void draw_debug_info(cv::Mat& frame, const std::list<Armor>& armors, double fps);
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

  xz_vision::Detector detector(config_path);
  xz_vision::Solver solver(config_path);
  xz_vision::Tracker tracker(config_path, solver);
  xz_vision::Aimer aimer(config_path);

  tools::Exiter exiter;
  ecu::Camera camera(config_path);

  tools::Plotter plotter;
  // cv::VideoCapture cap("/home/xinzhu/rm_xz_vision26/assets/armor.avi");

  cv::Mat raw_img;
  double avg_fps = 0;

  while (!exiter.exit()) {
    // std::cout << "Programing5..." << std::endl;
    auto start = std::chrono::steady_clock::now();

    std::chrono::steady_clock::time_point timestamp;
    camera.read(raw_img, timestamp);
    // cap >> raw_img;
    if (raw_img.empty())
      break;
    // std::cout << "Programing6..." << std::endl;
    //  --- 核心测试部分 ---
    //  调用你编写的深度学习+传统视觉结合函数
    //  这里会自动执行：YOLO推理 -> 灯条检测 -> 区域匹配 -> 赋予属性

    auto armors = detector.detect_onnx(raw_img);
    // std::cout << "Programing7..." << std::endl;
    //  ------------------
    //  for (auto& armor : armors) {
    //    solver.solve(armor);

    //   if (armor.xyz_in_world.norm() > 0.1 && armor.xyz_in_world.norm() < 10.0) {
    //     // 1. 格式化字符串
    //     std::string text =
    //         fmt::format("Pos:({:.2f},{:.2f},{:.2f})m, Dist:{:.2f}m", armor.xyz_in_world[0],
    //                     armor.xyz_in_world[1], armor.xyz_in_world[2], armor.xyz_in_world.norm());

    //     // 2. 打印到日志
    //     tools::logger()->info("{}", text);

    //     // 3. 绘制到图像
    //     cv::Point text_pos(10, 30);

    //     // 绘制文字 (参数：图像, 内容, 位置, 字体, 缩放, 颜色, 粗细)
    //     cv::putText(raw_img, text, text_pos, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255,
    //     0),
    //                 2);

    //     nlohmann::json data;
    //     const auto& armor = armors.front();
    //     data["armor_x"] = armor.xyz_in_world[0];
    //     data["armor_y"] = armor.xyz_in_world[1];
    //     data["armor_yaw"] = armor.ypr_in_world[0] * 57.3;
    //     data["armor_yaw_raw"] = armor.yaw_raw * 57.3;
    //     data["armor_center_x"] = armor.center_norm.x;
    //     data["armor_center_y"] = armor.center_norm.y;
    //     plotter.plot(data);
    //   }
    // }

    auto targets = tracker.track(armors, timestamp);

    // std::cout << "Programing1..." << std::endl;

    auto command = aimer.aim(targets, timestamp, 0);

    // std::cout << "Programing2..." << std::endl;

    ecu::VisionToGimbal vtg;
    vtg.yaw = command.yaw;
    vtg.pitch = command.pitch;
    // gimbal.send(vtg);

    if (armors.empty())
      continue;

    auto armor = armors.front();
    // std::cout << "Programing3..." << std::endl;
    nlohmann::json data;
    data["armor_x"] = armor.xyz_in_world.x();
    data["armor_y"] = armor.xyz_in_world.y();
    data["armor_z"] = armor.xyz_in_world.z();
    data["cmd_yaw"] = command.yaw;
    data["cmd_pitch"] = command.pitch;
    data["gimbal_yaw"] = command.yaw;
    data["gimbal_pitch"] = command.pitch;
    plotter.plot(data);

    auto end = std::chrono::steady_clock::now();
    double duration = std::chrono::duration<double, std::milli>(end - start).count();
    avg_fps = 1000.0 / duration;
    // std::cout << "Programing4..." << std::endl;

    // 绘制调试信息
    draw_debug_info(raw_img, armors, avg_fps);

    cv::imshow("XZ-Vision Test", raw_img);

    cv::waitKey(1);
    // if (key == 'q')
    //   break;
  }

  return 0;
}

/**
 * @brief 绘制调试信息
 * 绿色框/点：代表匹配成功的装甲板（灯条+数字均 OK）
 * 文本：显示类别 ID、置信度以及颜色信息
 */
void draw_debug_info(cv::Mat& frame, const std::list<Armor>& armors, double fps)
{
  // 1. 绘制左上角 FPS
  std::string fps_label = cv::format("FPS: %.1f", fps);
  cv::putText(frame, fps_label, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8,
              cv::Scalar(0, 255, 255), 2);

  for (const auto& armor : armors) {
    // 2. 绘制传统视觉识别出的精确 4 个角点
    if (armor.confidence < 0.6) {
      continue;
    }
    for (int i = 0; i < 4; i++) {
      cv::line(frame, armor.points[i], armor.points[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
      cv::circle(frame, armor.points[i], 3, cv::Scalar(255, 255, 0), -1);
    }
    // 3. 绘制 YOLO 输出的映射框 (确认匹配范围是否正确)
    if (armor.box.area() > 0) {
      cv::rectangle(frame, armor.box, cv::Scalar(255, 0, 255), 1,
                    cv::LINE_8); // 紫色虚框代表模型原始建议区
    }
    // 4. 绘制分类信息
    // cls_id 和 confidence 是从 19 维输出张量中解析出来的
    std::string info =
        cv::format("ID_NAME:%s Conf:%.2f", armor.get_id_name().c_str(), armor.confidence);
    cv::putText(frame, info, armor.points[0] + cv::Point2f(0, -10), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 255, 0), 1);

    // 5. 绘制中心点
    cv::circle(frame, armor.center, 4, cv::Scalar(0, 0, 255), -1);
  }
}