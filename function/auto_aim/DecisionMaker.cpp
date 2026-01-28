#include "DecisionMaker.hpp"

#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <algorithm>

// 定义M_PI（某些编译器可能需要）
#ifndef M_PI
  #define M_PI 3.14159265358979323846
#endif

namespace xz_vision
{

  // 简化版的YAML解析
  struct SimpleYAML {
    std::map<std::string, double> doubles;
    std::map<std::string, bool> bools;
    std::map<std::string, int> ints;
    std::map<std::string, std::string> strings;
  };

  SimpleYAML loadSimpleYAML(const std::string& path)
  {
    SimpleYAML yaml;
    std::ifstream file(path);
    std::string line;

    while (std::getline(file, line)) {
      if (line.empty() || line[0] == '#')
        continue;

      size_t colon_pos = line.find(':');
      if (colon_pos == std::string::npos)
        continue;

      std::string key = line.substr(0, colon_pos);
      std::string value = line.substr(colon_pos + 1);

      // 去除空格
      key.erase(0, key.find_first_not_of(" \t"));
      key.erase(key.find_last_not_of(" \t") + 1);
      value.erase(0, value.find_first_not_of(" \t"));
      value.erase(value.find_last_not_of(" \t") + 1);

      // 解析值
      if (value == "true") {
        yaml.bools[key] = true;
      } else if (value == "false") {
        yaml.bools[key] = false;
      } else if (value.find('.') != std::string::npos) {
        try {
          yaml.doubles[key] = std::stod(value);
        } catch (...) {
          // 忽略转换错误
        }
      } else {
        try {
          yaml.ints[key] = std::stoi(value);
        } catch (...) {
          yaml.strings[key] = value; // 如果是字符串
        }
      }
    }

    return yaml;
  }

  DecisionMaker::DecisionMaker(const std::string& config_path)
      : current_target_(nullptr)
  {
    if (!config_path.empty()) {
      loadConfig(config_path);
    }
    reset();
  }

  bool DecisionMaker::loadConfig(const std::string& config_path)
  {
    try {
      auto yaml = loadSimpleYAML(config_path);

      // 加载基本参数
      if (yaml.doubles.find("yaw_offset") != yaml.doubles.end())
        config_.yaw_offset = yaml.doubles["yaw_offset"] * M_PI / 180.0;
      if (yaml.doubles.find("pitch_offset") != yaml.doubles.end())
        config_.pitch_offset = yaml.doubles["pitch_offset"] * M_PI / 180.0;
      // if (yaml.doubles.find("comming_angle") != yaml.doubles.end())
      //   config_.comming_angle = yaml.doubles["comming_angle"] * M_PI / 180.0;
      // if (yaml.doubles.find("leaving_angle") != yaml.doubles.end())
      //   config_.leaving_angle = yaml.doubles["leaving_angle"] * M_PI / 180.0;
      // if (yaml.doubles.find("fire_thresh") != yaml.doubles.end())
      //   config_.fire_thresh = yaml.doubles["fire_thresh"];
      // if (yaml.doubles.find("decision_speed") != yaml.doubles.end())
      //   config_.decision_speed = yaml.doubles["decision_speed"];
      // if (yaml.doubles.find("high_speed_delay_time") != yaml.doubles.end())
      //   config_.high_speed_delay_time = yaml.doubles["high_speed_delay_time"];
      // if (yaml.doubles.find("low_speed_delay_time") != yaml.doubles.end())
      //   config_.low_speed_delay_time = yaml.doubles["low_speed_delay_time"];

      // 射击参数
      if (yaml.doubles.find("first_tolerance") != yaml.doubles.end())
        config_.first_tolerance = yaml.doubles["first_tolerance"] * M_PI / 180.0;
      if (yaml.doubles.find("second_tolerance") != yaml.doubles.end())
        config_.second_tolerance = yaml.doubles["second_tolerance"] * M_PI / 180.0;
      if (yaml.doubles.find("judge_distance") != yaml.doubles.end())
        config_.judge_distance = yaml.doubles["judge_distance"];
      if (yaml.bools.find("auto_fire") != yaml.bools.end())
        config_.auto_fire = yaml.bools["auto_fire"];

      // 决策参数
      if (yaml.doubles.find("min_hit_prob") != yaml.doubles.end())
        config_.min_hit_prob = yaml.doubles["min_hit_prob"];
      if (yaml.doubles.find("max_switch_interval") != yaml.doubles.end())
        config_.max_switch_interval = yaml.doubles["max_switch_interval"];
      if (yaml.doubles.find("stability_threshold") != yaml.doubles.end())
        config_.stability_threshold = yaml.doubles["stability_threshold"];
      if (yaml.ints.find("min_track_frames") != yaml.ints.end())
        config_.min_track_frames = yaml.ints["min_track_frames"];
      if (yaml.bools.find("enable_smooth") != yaml.bools.end())
        config_.enable_smooth = yaml.bools["enable_smooth"];
      if (yaml.bools.find("enable_mpc") != yaml.bools.end())
        config_.enable_mpc = yaml.bools["enable_mpc"];

      return true;
    } catch (const std::exception& e) {
      std::cerr << "[DecisionMaker] Failed to load config: " << e.what() << std::endl;
      return false;
    }
  }

  DecisionCommand DecisionMaker::makeDecision(const std::list<Armor>& detected_armors,
                                              double bullet_speed,
                                              const Eigen::Vector3d& gimbal_pos,
                                              std::chrono::steady_clock::time_point timestamp)
  {
    DecisionCommand cmd;
    cmd.timestamp = timestamp;
    cmd.valid = false;
    cmd.fire = false;

    // 1. 检查输入
    if (detected_armors.empty()) {
      lost_frames_++;
      if (lost_frames_ > 30) {
        reset();
      }
      return cmd;
    }

    lost_frames_ = 0;

    // 2. 选择目标
    auto new_target = selectBestTarget(detected_armors, timestamp);
    if (!new_target || !new_target->valid) {
      return cmd;
    }

    // 3. 目标切换判断
    if (!current_target_ || !current_target_->valid || shouldSwitchTarget(new_target, timestamp)) {
      current_target_ = new_target;
      last_switch_time_ = timestamp;
      stable_frames_ = 0;
      lock_id_ = -1;
    } else {
      // 更新目标状态
      updateTargetState(detected_armors, timestamp);
    }

    // 4. 计算控制指令
    cmd = calculateCommand(current_target_, bullet_speed, timestamp);
    if (!cmd.valid) {
      return cmd;
    }

    // 5. 评估命中概率
    hit_probability_ = evaluateHitProbability(current_target_, cmd);
    target_threat_ = evaluateTargetThreat(current_target_);

    // 6. 判断是否射击
    if (config_.auto_fire && hit_probability_ >= config_.min_hit_prob) {
      cmd.fire = shouldFire(cmd, gimbal_pos);
    }

    // 7. 更新历史记录
    last_command_ = cmd;
    yaw_history_.push_back(cmd.yaw);
    pitch_history_.push_back(cmd.pitch);
    if (yaw_history_.size() > 10)
      yaw_history_.pop_front();
    if (pitch_history_.size() > 10)
      pitch_history_.pop_front();

    // 8. 更新稳定帧数
    if (cmd.fire) {
      stable_frames_++;
    } else {
      stable_frames_ = std::max(0, stable_frames_ - 1);
    }

    return cmd;
  }

  void DecisionMaker::reset()
  {
    current_target_.reset();
    last_command_ = DecisionCommand();
    debug_aim_point_ = AimPoint();
    hit_probability_ = 0.0;
    target_threat_ = 0.0;
    stable_frames_ = 0;
    lost_frames_ = 0;
    lock_id_ = -1;
    yaw_history_.clear();
    pitch_history_.clear();
    last_switch_time_ = std::chrono::steady_clock::now();
  }

  std::shared_ptr<TargetState>
  DecisionMaker::selectBestTarget(const std::list<Armor>& armors,
                                  std::chrono::steady_clock::time_point timestamp)
  {
    if (armors.empty()) {
      return nullptr;
    }

    // 1. 按优先级排序
    std::vector<Armor> sorted_armors(armors.begin(), armors.end());
    std::sort(sorted_armors.begin(), sorted_armors.end(),
              [](const Armor& a, const Armor& b) { return a.priority < b.priority; });

    // 2. 选择优先级最高的装甲板
    const auto& best_armor = sorted_armors.front();

    // 3. 创建目标状态
    auto target = std::make_shared<TargetState>();
    target->valid = true;
    target->name = best_armor.name;
    target->type = best_armor.type;
    target->color = best_armor.color;
    target->priority = best_armor.priority;
    target->xyz_in_world = best_armor.xyz_in_world;
    target->center = best_armor.xyz_in_world; // 简化处理
    target->last_update_time = deltaTime(timestamp, std::chrono::steady_clock::time_point());
    target->track_duration = 0.0;
    target->jumped = false;
    target->converged = false;

    // 4. 初始化EKF状态
    target->ekf_x = Eigen::VectorXd::Zero(11);
    target->ekf_x[0] = best_armor.xyz_in_world.x();                                          // x
    target->ekf_x[2] = best_armor.xyz_in_world.y();                                          // y
    target->ekf_x[4] = best_armor.xyz_in_world.z();                                          // z
    target->ekf_x[6] = std::atan2(best_armor.xyz_in_world.y(), best_armor.xyz_in_world.x()); // yaw
    target->ekf_x[8] = 0.53; // 半径 (简化)

    // 5. 初始化装甲板列表
    target->armor_xyza_list.clear();
    Eigen::Vector4d xyza;
    xyza << best_armor.xyz_in_world.x(), best_armor.xyz_in_world.y(), best_armor.xyz_in_world.z(),
        target->ekf_x[6];
    target->armor_xyza_list.push_back(xyza);

    return target;
  }

  bool DecisionMaker::shouldSwitchTarget(const std::shared_ptr<TargetState>& new_target,
                                         std::chrono::steady_clock::time_point timestamp)
  {
    if (!current_target_ || !current_target_->valid) {
      return true;
    }

    const auto& current = current_target_;

    // 1. 检查切换间隔
    double dt = deltaTime(timestamp, last_switch_time_);
    if (dt < config_.max_switch_interval) {
      return false;
    }

    // 2. 优先级比较
    if (new_target->priority < current->priority) {
      return true;
    }

    // 3. 威胁度比较
    double new_threat = evaluateTargetThreat(new_target);
    double current_threat = evaluateTargetThreat(current);
    if (new_threat > current_threat * 1.5) {
      return true;
    }

    return false;
  }

  void DecisionMaker::updateTargetState(const std::list<Armor>& armors,
                                        std::chrono::steady_clock::time_point timestamp)
  {
    if (!current_target_ || !current_target_->valid) {
      return;
    }

    auto& target = *current_target_;

    // 查找匹配的装甲板
    std::vector<const Armor*> matched_armors;
    for (const auto& armor : armors) {
      if (armor.name == target.name && armor.type == target.type) {
        matched_armors.push_back(&armor);
      }
    }

    if (matched_armors.empty()) {
      target.valid = false;
      return;
    }

    // 更新位置
    const auto& armor = *matched_armors.front();
    // 明确转换
    // 使用 duration_cast 明确转换
    auto last_update = std::chrono::steady_clock::time_point() +
                       std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                           std::chrono::duration<double>(target.last_update_time));

    double dt = deltaTime(timestamp, last_update);

    if (dt < 0.001)
      dt = 0.001; // 避免除零

    // 计算速度
    Eigen::Vector3d new_pos = armor.xyz_in_world;
    Eigen::Vector3d old_pos = target.xyz_in_world;
    Eigen::Vector3d velocity = (new_pos - old_pos) / dt;

    // 更新目标状态
    target.xyz_in_world = new_pos;
    target.center = new_pos; // 简化
    target.last_update_time = deltaTime(timestamp, std::chrono::steady_clock::time_point());
    target.track_duration += dt;
    target.velocity = velocity;

    // 更新EKF状态
    target.ekf_x[0] = new_pos.x();
    target.ekf_x[2] = new_pos.y();
    target.ekf_x[4] = new_pos.z();
    target.ekf_x[1] = velocity.x(); // vx
    target.ekf_x[3] = velocity.y(); // vy
    target.ekf_x[5] = velocity.z(); // vz
    target.ekf_x[6] = std::atan2(new_pos.y(), new_pos.x());
    target.ekf_x[7] = target.angular_velocity.norm(); // 旋转速度

    // 收敛判断
    if (target.track_duration > 0.5) {
      target.converged = true;
    }
  }

  AimPoint DecisionMaker::chooseAimPoint(const std::shared_ptr<TargetState>& target)
  {
    AimPoint aim_point;
    aim_point.valid = false;

    if (!target || !target->valid || target->armor_xyza_list.empty()) {
      return aim_point;
    }

    // 简化版瞄准点选择
    if (target->armor_xyza_list.size() == 1) {
      aim_point.xyza = target->armor_xyza_list[0];
      aim_point.valid = true;
      return aim_point;
    }

    // 如果有多个装甲板，选择最靠近中心的
    double min_angle = std::numeric_limits<double>::max();
    int best_idx = 0;

    for (size_t i = 0; i < target->armor_xyza_list.size(); ++i) {
      double angle = std::fabs(target->armor_xyza_list[i][3]);
      if (angle < min_angle) {
        min_angle = angle;
        best_idx = i;
      }
    }

    aim_point.xyza = target->armor_xyza_list[best_idx];
    aim_point.valid = true;
    return aim_point;
  }

  DecisionCommand DecisionMaker::calculateCommand(const std::shared_ptr<TargetState>& target,
                                                  double bullet_speed,
                                                  std::chrono::steady_clock::time_point timestamp)
  {
    DecisionCommand cmd;
    cmd.timestamp = timestamp;
    cmd.valid = false;

    if (!target || !target->valid || bullet_speed < 10) {
      return cmd;
    }

    // 1. 选择瞄准点
    debug_aim_point_ = chooseAimPoint(target);
    if (!debug_aim_point_.valid) {
      return cmd;
    }

    // 2. 计算弹道
    Eigen::Vector3d target_pos = debug_aim_point_.xyza.head<3>();
    auto trajectory = calculateTrajectory(bullet_speed, target_pos);
    if (trajectory.unsolvable) {
      debug_aim_point_.valid = false;
      return cmd;
    }

    // 3. 计算角度
    double yaw = std::atan2(target_pos.y(), target_pos.x()) + config_.yaw_offset;
    double pitch = -(trajectory.pitch + config_.pitch_offset);

    // 4. 平滑处理
    if (config_.enable_smooth && !yaw_history_.empty()) {
      double yaw_smooth = 0.0;
      double pitch_smooth = 0.0;
      double total_weight = 0.0;

      for (size_t i = 0; i < yaw_history_.size(); ++i) {
        double weight = 1.0 / (i + 1);
        yaw_smooth += yaw_history_[i] * weight;
        total_weight += weight;

        if (i < pitch_history_.size()) {
          pitch_smooth += pitch_history_[i] * weight;
        }
      }

      yaw_smooth /= total_weight;
      pitch_smooth /= total_weight;

      yaw = 0.7 * yaw + 0.3 * yaw_smooth;
      pitch = 0.7 * pitch + 0.3 * pitch_smooth;
    }

    // 5. 计算角速度
    double yaw_vel = 0.0;
    double pitch_vel = 0.0;

    if (yaw_history_.size() >= 2) {
      yaw_vel = (yaw - yaw_history_.back()) / 0.01; // 假设10ms周期
    }
    if (pitch_history_.size() >= 2) {
      pitch_vel = (pitch - pitch_history_.back()) / 0.01;
    }

    cmd.valid = true;
    cmd.yaw = yaw;
    cmd.pitch = pitch;
    cmd.yaw_vel = yaw_vel;
    cmd.pitch_vel = pitch_vel;
    cmd.yaw_acc = 0.0; // 简化处理
    cmd.pitch_acc = 0.0;

    return cmd;
  }

  bool DecisionMaker::shouldFire(const DecisionCommand& command, const Eigen::Vector3d& gimbal_pos)
  {
    if (!command.valid || !config_.auto_fire) {
      return false;
    }

    // 1. 检查瞄准点有效性
    if (!debug_aim_point_.valid) {
      return false;
    }

    // 2. 检查指令突变
    if (last_command_.valid) {
      double yaw_diff = std::fabs(last_command_.yaw - command.yaw);
      if (yaw_diff > config_.second_tolerance * 2) {
        return false;
      }
    }

    // 3. 检查云台到位
    double tolerance = std::sqrt(command.yaw * command.yaw + command.pitch * command.pitch) >
                               config_.judge_distance
                           ? config_.second_tolerance
                           : config_.first_tolerance;

    double yaw_error = std::fabs(gimbal_pos[0] - command.yaw);
    if (yaw_error > tolerance) {
      return false;
    }

    // 4. 检查稳定性
    if (stable_frames_ < config_.min_track_frames) {
      return false;
    }

    return true;
  }

  Trajectory DecisionMaker::calculateTrajectory(double bullet_speed,
                                                const Eigen::Vector3d& target_pos)
  {
    Trajectory trajectory;

    if (bullet_speed < 1.0) {
      trajectory.unsolvable = true;
      return trajectory;
    }

    // 简化弹道计算
    double dx = std::sqrt(target_pos.x() * target_pos.x() + target_pos.y() * target_pos.y());
    double dy = target_pos.z();
    double g = 9.8; // 重力加速度

    if (dx < 0.1) {
      trajectory.unsolvable = true;
      return trajectory;
    }

    // 计算俯仰角
    double v = bullet_speed;
    double v2 = v * v;
    double v4 = v2 * v2;
    double root = v4 - g * (g * dx * dx + 2 * dy * v2);

    if (root < 0) {
      trajectory.unsolvable = true;
      return trajectory;
    }

    double tan_theta = (v2 + std::sqrt(root)) / (g * dx);
    trajectory.pitch = std::atan(tan_theta);

    // 计算飞行时间
    trajectory.fly_time = dx / (v * std::cos(trajectory.pitch));

    return trajectory;
  }

  Eigen::Vector3d DecisionMaker::predictTargetPosition(const std::shared_ptr<TargetState>& target,
                                                       double delta_time)
  {
    Eigen::Vector3d predicted_pos(0.0, 0.0, 0.0);

    if (target && target->valid) {
      // 简单线性预测
      predicted_pos = target->xyz_in_world + target->velocity * delta_time;

      // 考虑旋转
      if (std::fabs(target->ekf_x[7]) > 0.1) { // 如果有旋转
        double radius = target->ekf_x[8];
        double delta_yaw = target->ekf_x[7] * delta_time;

        // 2D旋转
        double cos_a = std::cos(delta_yaw);
        double sin_a = std::sin(delta_yaw);
        double new_x = predicted_pos.x() * cos_a - predicted_pos.y() * sin_a;
        double new_y = predicted_pos.x() * sin_a + predicted_pos.y() * cos_a;

        predicted_pos.x() = new_x;
        predicted_pos.y() = new_y;
      }
    }

    return predicted_pos;
  }

  double DecisionMaker::evaluateHitProbability(const std::shared_ptr<TargetState>& target,
                                               const DecisionCommand& command)
  {
    if (!target || !target->valid || !command.valid) {
      return 0.0;
    }

    // 计算命中概率的因素：
    double probability = 1.0;

    // 1. 距离因素
    double distance = target->xyz_in_world.norm();
    probability *= std::exp(-distance / 5.0);

    // 2. 速度因素
    double speed = target->velocity.norm();
    probability *= std::exp(-speed / 2.0);

    // 3. 跟踪稳定性
    probability *= std::min(static_cast<double>(stable_frames_) / 20.0, 1.0);

    // 4. 目标类型
    if (target->name == ArmorName::outpost) {
      probability *= 0.8;
    } else if (target->name == ArmorName::base) {
      probability *= 0.9;
    }

    // 限制在0-1之间
    if (probability < 0.0)
      probability = 0.0;
    if (probability > 1.0)
      probability = 1.0;

    return probability;
  }

  double DecisionMaker::evaluateTargetThreat(const std::shared_ptr<TargetState>& target)
  {
    if (!target || !target->valid) {
      return 0.0;
    }

    double threat = 0.0;

    // 1. 距离威胁
    double distance = target->xyz_in_world.norm();
    double distance_threat = 1.0 / (1.0 + distance);

    // 2. 目标类型威胁
    double type_threat = 1.0;
    switch (target->name) {
    case ArmorName::outpost: type_threat = 0.8; break;
    case ArmorName::base: type_threat = 1.5; break;
    case ArmorName::one: type_threat = 1.2; break;
    default: type_threat = 1.0; break;
    }

    // 3. 运动状态威胁
    double speed = target->velocity.norm();
    double motion_threat = 1.0 + speed / 5.0;

    threat = distance_threat * 0.4 + type_threat * 0.4 + motion_threat * 0.2;

    return threat;
  }

  double DecisionMaker::limitRad(double angle) const
  {
    angle = std::fmod(angle, 2.0 * M_PI);
    if (angle > M_PI) {
      angle -= 2.0 * M_PI;
    } else if (angle < -M_PI) {
      angle += 2.0 * M_PI;
    }
    return angle;
  }

  double DecisionMaker::deltaTime(std::chrono::steady_clock::time_point t2,
                                  std::chrono::steady_clock::time_point t1) const
  {
    return std::chrono::duration<double>(t2 - t1).count();
  }

  Eigen::Vector3d DecisionMaker::calculateVelocity(const std::shared_ptr<TargetState>& target) const
  {
    if (target && target->valid) {
      return target->velocity;
    }
    return Eigen::Vector3d::Zero();
  }

} // namespace xz_vision