#ifndef AUTO_AIM__SHOOTER_HPP
#define AUTO_AIM__SHOOTER_HPP

#include <string>

#include "ecu/command.hpp"
#include "aimer.hpp"

namespace xz_vision
{
  class Shooter
  {
  public:
    Shooter(const std::string& config_path);

    bool shoot(const ecu::Command& command, const xz_vision::Aimer& aimer,
               const std::list<xz_vision::Target>& targets, const Eigen::Vector3d& gimbal_pos);

  private:
    ecu::Command last_command_;
    double judge_distance_;
    double first_tolerance_;
    double second_tolerance_;
    bool auto_fire_;
  };
} // namespace xz_vision

#endif // AUTO_AIM__SHOOTER_HPP