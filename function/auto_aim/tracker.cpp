#include "tracker.hpp"

#include <yaml-cpp/yaml.h>

#include <tuple>

#include "tools/logger.hpp"
#include "tools/rotary_tool.hpp"

namespace xz_vision
{
  Tracker::Tracker(const std::string& config_path, Solver& solver)
      : solver_{solver}
      , detect_count_(0)
      , temp_lost_count_(0)
      , state_{"lost"}
      , pre_state_{"lost"}
      , last_timestamp_(std::chrono::steady_clock::now())
  {
    auto yaml = YAML::LoadFile(config_path);

    enemy_color_ = (yaml["enemy_color"].as<std::string>() == "red") ? Color::red : Color::blue;
    min_detect_count_ = yaml["min_detect_count"].as<int>();
    max_temp_lost_count_ = yaml["max_temp_lost_count"].as<int>();
    outpost_max_temp_lost_count_ = yaml["outpost_max_temp_lost_count"].as<int>();
    normal_temp_lost_count_ = max_temp_lost_count_;
  }

  // 返回跟踪状态
  std::string Tracker::state() const { return state_; }

  // 跟踪主函数，返回目标
  std::list<Target> Tracker::track(std::list<Armor>& armors,
                                   std::chrono::steady_clock::time_point t, bool use_enemy_color)
  {
    auto dt = tools::delta_time(t, last_timestamp_);
    last_timestamp_ = t;

    // 时间间隔过长，说明可能发生了相机离线
    if (state_ != "lost" && dt > 0.1) {
      tools::logger()->warn("[Tracker] Large dt: {:.3f}s", dt);
      state_ = "lost";
    }

    // 过滤掉非我方装甲板
    armors.remove_if([&](const Armor& a) { return a.color != enemy_color_; });

    // 过滤前哨站顶部装甲板
    // armors.remove_if([this](const auto_aim::Armor & a) {
    //   return a.name == ArmorName::outpost &&
    //          solver_.oupost_reprojection_error(a, 27.5 * CV_PI / 180.0) <
    //            solver_.oupost_reprojection_error(a, -15 * CV_PI / 180.0);
    // });

    // 优先选择靠近图像中心的装甲板
    armors.sort([](const Armor& a, const Armor& b) {
      cv::Point2f img_center(1440 / 2, 1080 / 2); // TODO: 应该从配置读取
      auto distance_1 = cv::norm(a.center - img_center);
      auto distance_2 = cv::norm(b.center - img_center);
      return distance_1 < distance_2;
    });

    // 按优先级排序，优先级最高在首位(优先级越高数字越小，1的优先级最高)
    armors.sort([](const Armor& a, const Armor& b) { return a.priority < b.priority; });

    bool found;
    if (state_ == "lost") {
      found = set_target(armors, t);
    } else {
      // 如果主相机画面中出现了优先级更高的装甲板，切换目标
      if (!armors.empty() && armors.front().priority < target_.priority) {
        tools::logger()->debug("Switch target to {}", ARMOR_NAME[armors.front().name]);
        found = set_target(armors, t);
      } else {
        found = update_target(armors, t);
      }
    }

    pre_state_ = state_;
    state_machine(found);

    // 发散检测
    if (state_ != "lost" && target_.diverged()) {
      tools::logger()->debug("[Tracker] Target diverged!");
      state_ = "lost";
      return {};
    }

    // 收敛效果检测：
    if (std::accumulate(target_.ekf().recent_nis_failures.begin(),
                        target_.ekf().recent_nis_failures.end(),
                        0) >= (0.4 * target_.ekf().window_size)) {
      tools::logger()->debug("[Target] Bad Converge Found!");
      state_ = "lost";
      return {};
    }

    if (state_ == "lost")
      return {};

    std::list<Target> targets = {target_};
    return targets;
  }

  void Tracker::state_machine(bool found)
  {
    if (state_ == "lost") {
      if (!found)
        return;

      state_ = "detecting";
      detect_count_ = 1;
    }

    else if (state_ == "detecting") {
      if (found) {
        detect_count_++;
        if (detect_count_ >= min_detect_count_)
          state_ = "tracking";
      } else {
        detect_count_ = 0;
        state_ = "lost";
      }
    }

    else if (state_ == "tracking") {
      if (found)
        return;

      temp_lost_count_ = 1;
      state_ = "temp_lost";
    }

    else if (state_ == "temp_lost") {
      if (found) {
        state_ = "tracking";
        temp_lost_count_ = 0;
      } else {
        temp_lost_count_++;
        if (target_.name == ArmorName::outpost)
          // 前哨站的temp_lost_count需要设置的大一些
          max_temp_lost_count_ = outpost_max_temp_lost_count_;
        else
          max_temp_lost_count_ = normal_temp_lost_count_;

        if (temp_lost_count_ > max_temp_lost_count_) {
          state_ = "lost";
          temp_lost_count_ = 0;
        }
      }
    }
  }

  // 为不同类型的装甲板设置基础参数
  bool Tracker::set_target(std::list<Armor>& armors, std::chrono::steady_clock::time_point t)
  {
    if (armors.empty())
      return false;

    auto& armor = armors.front();
    solver_.solve(armor);

    // 根据兵种优化初始化参数
    auto is_balance = (armor.type == ArmorType::small) &&
                      (armor.name == ArmorName::three || armor.name == ArmorName::four);

    if (is_balance) {
      // clang-format off
      /* 初始协方差矩阵（11维度）
         索引：0,2,4   x,y,z (位置)         初始位置相对确定（由 PnP 视觉解算得到）
         索引：1,3,5   vx​,vy​,vz​ (速度)      不确定初始速度。因为第一帧无法知晓速度，给大值让滤波器快点收敛。 
         索引: 6      θ (目标旋转角度)       对目标当前转到哪个角度的初步估计。 
         索引: 7      ω (角速度)            极不确定。目标转得多快？一开始完全不知道，所以给最高值。 
         索引: 8      r (旋转半径)           目标装甲板到中心的距离。 
         索引: 9, 10  其他/加速度            系统的辅助修正项。
      */
      // clang-format on
      Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 64, 0.4, 100, 1, 1, 1}};
      target_ = Target(armor, t, 0.2, 2, P0_dig);
    }

    else if (armor.name == ArmorName::outpost) {
      Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 81, 0.4, 100, 1e-4, 0, 0}};
      target_ = Target(armor, t, 0.2765, 3, P0_dig);
    }

    else if (armor.name == ArmorName::base) {
      Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 64, 0.4, 100, 1e-4, 0, 0}};
      target_ = Target(armor, t, 0.3205, 3, P0_dig);
    }

    else {
      Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 64, 0.4, 100, 1, 1, 1}};
      target_ = Target(armor, t, 0.2, 4, P0_dig);
    }

    return true;
  }

  // 在已知上一次目标的情况下，从当前帧识别到的一堆装甲板中，找到最匹配的那一个，并更新滤波器状态
  bool Tracker::update_target(std::list<Armor>& armors, std::chrono::steady_clock::time_point t)
  {
    target_.predict(t);

    int found_count = 0;
    double min_x = 1e10; // 画面最左侧
    for (const auto& armor : armors) {
      if (armor.name != target_.name || armor.type != target_.armor_type)
        continue;
      found_count++;
      min_x = armor.center.x < min_x ? armor.center.x : min_x;
    }

    if (found_count == 0)
      return false;

    for (auto& armor : armors) {
      if (armor.name != target_.name || armor.type != target_.armor_type
          //  || armor.center.x != min_x
      )
        continue;

      solver_.solve(armor);

      target_.update(armor);
    }

    return true;
  }

} // namespace xz_vision