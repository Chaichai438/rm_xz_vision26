#ifndef AUTO_AIM__DECISION_MAKER_HPP
#define AUTO_AIM__DECISION_MAKER_HPP

#include "armor.hpp"
#include <Eigen/Dense>

#include <chrono>
#include <cmath>
#include <deque>
#include <list>
#include <memory>
#include <string>
#include <vector>

namespace xz_vision
{
  // 决策指令
  struct DecisionCommand {
    bool valid{false};                               // 指令是否有效
    bool fire{false};                                // 是否射击
    double yaw{0.0};                                 // 偏航角 (rad)
    double pitch{0.0};                               // 俯仰角 (rad)
    double yaw_vel{0.0};                             // 偏航角速度 (rad/s)
    double pitch_vel{0.0};                           // 俯仰角速度 (rad/s)
    double yaw_acc{0.0};                             // 偏航加速度 (rad/s²)
    double pitch_acc{0.0};                           // 俯仰加速度 (rad/s²)
    std::chrono::steady_clock::time_point timestamp; // 时间戳
  };

  // 瞄准点
  struct AimPoint {
    bool valid{false};
    Eigen::Vector4d xyza{Eigen::Vector4d::Zero()}; // x, y, z, yaw
  };

  // 目标状态
  struct TargetState {
    bool valid{false};
    ArmorName name{ArmorName::not_armor};
    ArmorType type{ArmorType::small};
    Color color{Color::blue};
    int priority{0};
    Eigen::Vector3d xyz_in_world{Eigen::Vector3d::Zero()};     // 世界坐标位置
    Eigen::Vector3d velocity{Eigen::Vector3d::Zero()};         // 速度
    Eigen::Vector3d angular_velocity{Eigen::Vector3d::Zero()}; // 角速度
    Eigen::Vector3d center{Eigen::Vector3d::Zero()};           // 图像中心
    double last_update_time{0.0};                              // 上次更新时间
    double track_duration{0.0};                                // 跟踪时长
    bool jumped{false};                                        // 是否发生跳变
    bool converged{false};                                     // 是否收敛
    std::vector<Eigen::Vector4d> armor_xyza_list;              // 装甲板位置列表

    // 简化EKF状态向量 [x, vx, y, vy, z, vz, yaw, vyaw, radius, 0, 0]
    Eigen::VectorXd ekf_x{Eigen::VectorXd::Zero(11)};

    // 构造函数
    TargetState() = default;

    // 复制构造函数
    TargetState(const TargetState& other) = default;

    // 赋值运算符
    TargetState& operator=(const TargetState& other) = default;
  };

  // 决策器配置
  struct DecisionConfig {
    // 基本参数
    double yaw_offset{0.0};                    // 偏航偏移 (rad)
    double pitch_offset{0.0};                  // 俯仰偏移 (rad)
    double comming_angle{60.0 * M_PI / 180.0}; // 进入角度 (rad)
    double leaving_angle{30.0 * M_PI / 180.0}; // 离开角度 (rad)
    double fire_thresh{0.05};                  // 射击阈值 (rad)
    double decision_speed{1.5};                // 决策速度阈值 (m/s)
    double high_speed_delay_time{0.1};         // 高速延迟时间 (s)
    double low_speed_delay_time{0.05};         // 低速延迟时间 (s)

    // 射击参数
    double first_tolerance{1.0 * M_PI / 180.0};  // 近距离容差 (rad)
    double second_tolerance{2.0 * M_PI / 180.0}; // 远距离容差 (rad)
    double judge_distance{3.0};                  // 距离判断阈值 (m)
    bool auto_fire{true};                        // 自动射击

    // 决策参数
    double min_hit_prob{0.6};         // 最小命中概率
    double max_switch_interval{0.5};  // 最大切换间隔 (s)
    double stability_threshold{0.05}; // 稳定性阈值 (rad)
    int min_track_frames{5};          // 最小跟踪帧数
    bool enable_smooth{true};         // 启用平滑
    bool enable_mpc{false};           // 启用MPC控制
  };

  // 弹道轨迹
  struct Trajectory {
    double fly_time{0.0};   // 飞行时间
    double pitch{0.0};      // 俯仰角
    bool unsolvable{false}; // 是否可解
  };

  class DecisionMaker
  {
  public:
    // 构造函数
    explicit DecisionMaker(const std::string& config_path = "");

    // 主决策函数
    DecisionCommand makeDecision(const std::list<Armor>& detected_armors, double bullet_speed,
                                 const Eigen::Vector3d& gimbal_pos,
                                 std::chrono::steady_clock::time_point timestamp);

    // 设置配置
    void setConfig(const DecisionConfig& config) { config_ = config; }

    // 获取当前配置
    DecisionConfig getConfig() const { return config_; }

    // 获取当前目标（返回shared_ptr）
    std::shared_ptr<TargetState> getCurrentTarget() const { return current_target_; }

    // 获取命中概率
    double getHitProbability() const { return hit_probability_; }

    // 获取调试信息
    AimPoint getDebugAimPoint() const { return debug_aim_point_; }

    // 重置状态
    void reset();

  private:
    // 配置
    DecisionConfig config_;

    // 状态（使用shared_ptr替代optional）
    std::shared_ptr<TargetState> current_target_;
    DecisionCommand last_command_;
    AimPoint debug_aim_point_;
    double hit_probability_{0.0};
    double target_threat_{0.0};
    int stable_frames_{0};
    int lost_frames_{0};
    int lock_id_{-1};
    std::chrono::steady_clock::time_point last_switch_time_;
    std::deque<double> yaw_history_;
    std::deque<double> pitch_history_;

    // 私有方法
    bool loadConfig(const std::string& config_path);
    std::shared_ptr<TargetState> selectBestTarget(const std::list<Armor>& armors,
                                                  std::chrono::steady_clock::time_point timestamp);
    bool shouldSwitchTarget(const std::shared_ptr<TargetState>& new_target,
                            std::chrono::steady_clock::time_point timestamp);
    void updateTargetState(const std::list<Armor>& armors,
                           std::chrono::steady_clock::time_point timestamp);
    AimPoint chooseAimPoint(const std::shared_ptr<TargetState>& target);
    DecisionCommand calculateCommand(const std::shared_ptr<TargetState>& target,
                                     double bullet_speed,
                                     std::chrono::steady_clock::time_point timestamp);
    bool shouldFire(const DecisionCommand& command, const Eigen::Vector3d& gimbal_pos);
    Trajectory calculateTrajectory(double bullet_speed, const Eigen::Vector3d& target_pos);
    Eigen::Vector3d predictTargetPosition(const std::shared_ptr<TargetState>& target,
                                          double delta_time);
    double evaluateHitProbability(const std::shared_ptr<TargetState>& target,
                                  const DecisionCommand& command);
    double evaluateTargetThreat(const std::shared_ptr<TargetState>& target);
    double limitRad(double angle) const;
    double deltaTime(std::chrono::steady_clock::time_point t2,
                     std::chrono::steady_clock::time_point t1) const;
    Eigen::Vector3d calculateVelocity(const std::shared_ptr<TargetState>& target) const;

    // 辅助函数：检查目标是否有效
    bool isTargetValid(const std::shared_ptr<TargetState>& target) const
    {
      return target && target->valid;
    }
  };

} // namespace xz_vision

#endif // AUTO_AIM__DECISION_MAKER_HPPexi