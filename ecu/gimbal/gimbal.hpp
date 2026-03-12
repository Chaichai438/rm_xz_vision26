#ifndef IO__GIMBAL_HPP
#define IO__GIMBAL_HPP

#include <Eigen/Geometry>
#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>

#include "ecu/serial/include/serial/serial.h"
#include "tools/thread_safe_queue.hpp"

namespace ecu
{
  struct __attribute__((packed)) GimbalToVision {
    uint8_t head[2] = {'S', 'P'};
    uint8_t mode; // 0: 空闲, 1: 自瞄, 2: 小符, 3: 大符
    float q[4];   // wxyz顺序 四元数
    float pitch;
    float yaw;
    float bullet_speed;    // 弹速
    uint16_t bullet_count; // 子弹累计发送次数
    uint16_t crc16;
  };

  static_assert(sizeof(GimbalToVision) <= 64);

  struct __attribute__((packed)) VisionToGimbal {
    uint8_t head[2] = {'S', 'P'};
    uint8_t mode; // 0: 不控制 1: 控制云台且开火
    float yaw;
    float yaw_vel;
    float yaw_acc;
    float pitch;
    float pitch_vel;
    float pitch_acc;
    uint16_t crc16;
  };

  static_assert(sizeof(VisionToGimbal) <= 64);

  enum GimbalMode {
    IDLE,       // 空闲
    AUTO_AIM,   // 自瞄
    SMALL_BUFF, // 小符
    BIG_BUFF    // 大符
  };
  const std::vector<std::string> GIMBAL_MODE = {"IDLE", "AUTO_AIM", "SMALL_BUFF", "BIG_BUFF"};

  struct GimbalState {
    float yaw;
    float pitch;
    float bullet_speed;
    uint16_t bullet_count;
  };

  class Gimbal
  {
  public:
    Gimbal(const std::string& config_path);

    ~Gimbal();

    GimbalMode mode() const;   // 获取云台模式
    GimbalState state() const; // 获取云台位姿
    std::string str(GimbalMode mode) const;
    Eigen::Quaterniond q(std::chrono::steady_clock::time_point t);

    void send(ecu::VisionToGimbal VisionToGimbal);

  private:
    serial::Serial serial_;
    uint32_t baudrate;

    std::thread thread_;
    std::atomic<bool> quit_ = false;
    mutable std::mutex mutex_;

    GimbalToVision rx_data_;
    VisionToGimbal tx_data_;

    GimbalMode mode_ = GimbalMode::IDLE;
    GimbalState state_;
    tools::ThreadSafeQueue<std::tuple<Eigen::Quaterniond, std::chrono::steady_clock::time_point>>
        queue_{1000};

    bool read(uint8_t* buffer, size_t size);
    void read_thread();
    void reconnect(); // 重连函数
  };

} // namespace ecu

#endif // IO__GIMBAL_HPP