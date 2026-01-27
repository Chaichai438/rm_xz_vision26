#include "camera.hpp"

#include <stdexcept>

#include "hikrobot/hikrobot.hpp"
// #include "daheng/daheng.hpp"
// 大恒相机库
#include "tools/yaml.hpp"

namespace ecu
{
  Camera::Camera(const std::string& config_path)
  {
    auto yaml = tools::load(config_path);
    auto camera_name = tools::read<std::string>(yaml, "camera_name");
    auto exposure_ms = tools::read<double>(yaml, "exposure_ms");

    // if (camera_name == "daheng") {
    //   auto gamma = tools::read<double>(yaml, "gamma");
    //   auto vid_pid = tools::read<std::string>(yaml, "vid_pid");
    //   camera_ = std::make_unique<Daheng>(exposure_ms, gamma, vid_pid);
    // }

    if (camera_name == "hikrobot") {
      auto gain = tools::read<double>(yaml, "gain");
      auto vid_pid = tools::read<std::string>(yaml, "vid_pid");
      camera_ = std::make_unique<io::HikRobot>(exposure_ms, gain, vid_pid);
    }

    else {
      throw std::runtime_error("Unknow camera_name: " + camera_name + "!");
    }
  }

  void Camera::read(cv::Mat& img, std::chrono::steady_clock::time_point& timestamp)
  {
    camera_->read(img, timestamp);
  }

} // namespace ecu