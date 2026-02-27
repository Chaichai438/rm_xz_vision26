#ifndef AUTO_AIM__AIMER_HPP
#define AUTO_AIM__AIMER_HPP

#include <Eigen/Dense>
#include <chrono>
#include <list>

// #include "io/cboard.hpp"
#include "ecu/command.hpp"
#include "target.hpp"

namespace xz_vision
{

  struct AimPoint {
    bool valid;           // 是否有效的
    Eigen::Vector4d xyza; //[x,y,z,armor_yaw] 四维向量
  };

  class Aimer
  {
  public:
    AimPoint debug_aim_point;
    explicit Aimer(const std::string& config_path);
    ecu::Command aim(std::list<Target> targets, std::chrono::steady_clock::time_point timestamp,
                     double bullet_speed, bool to_now = true);

    // ecu::Command aim(std::list<Target> targets, std::chrono::steady_clock::time_point timestamp,
    //                  double bullet_speed, ecu::ShootMode shoot_mode, bool to_now = true);

  private:
    double yaw_offset_;
    std::optional<double> left_yaw_offset_, right_yaw_offset_;
    double pitch_offset_;
    double comming_angle_;
    double leaving_angle_;
    double lock_id_ = -1;
    double high_speed_delay_time_;
    double low_speed_delay_time_;
    double decision_speed_;

    AimPoint choose_aim_point(const Target& target);
  };

} // namespace xz_vision

#endif // AUTO_AIM__AIMER_HPP