#include "gimbal.hpp"

#include <Eigen/Geometry>
#include <cstring>

#include "tools/rotary_tool.hpp"
#include "tools/logger.hpp"
#include "tools/yaml.hpp"

namespace ecu
{
  Gimbal::Gimbal(const std::string& config_path)
  {
    auto yaml = tools::load(config_path);
    auto com_port = tools::read<std::string>(yaml, "com_port");

    try {
      serial_.setPort(com_port);
      serial_.open();
    } catch (const std::exception& e) {
      tools::logger()->error("[Gimbal] Failed to open serial: {}", e.what());
      exit(1);
    }

    thread_ = std::thread(&Gimbal::read_thread, this);
    tools::logger()->info("[Gimbal] Visual-to-MCU communication link established.");
  }

  Gimbal::~Gimbal()
  {
    quit_ = true;
    if (thread_.joinable())
      thread_.join();
    serial_.close();
  }

  GimbalMode Gimbal::mode() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return mode_;
  }

  GimbalState Gimbal::state() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return state_;
  }

  std::string Gimbal::str(GimbalMode mode) const
  {
    switch (mode) {
    case GimbalMode::IDLE: return "IDLE";
    case GimbalMode::AUTO_AIM: return "AUTO_AIM";
    case GimbalMode::SMALL_BUFF: return "SMALL_BUFF";
    case GimbalMode::BIG_BUFF: return "BIG_BUFF";
    default: return "INVALID";
    }
  }

  Eigen::Quaterniond Gimbal::q(std::chrono::steady_clock::time_point t)
  {
    while (true) {
      auto [q_a, t_a] = queue_.pop();
      auto [q_b, t_b] = queue_.front();
      auto t_ab = tools::delta_time(t_a, t_b);
      auto t_ac = tools::delta_time(t_a, t);
      auto k = t_ac / t_ab;
      Eigen::Quaterniond q_c = q_a.slerp(k, q_b).normalized();
      if (t < t_a)
        return q_c;
      if (!(t_a < t && t <= t_b))
        continue;

      return q_c;
    }
  }

  /**
   * @brief 【发送协议】视觉发送给 STM32
   * 匹配 STM32 接收函数: vision_data_UARTget (图片 fb736...)
   */
  void Gimbal::send(ecu::VisionToGimbal data)
  {
    // STM32 接收缓冲区 vision_buffer 长度为 14 (0-13)
    uint8_t tx_buf[14];
    memset(tx_buf, 0, sizeof(tx_buf));

    // Byte 0: 帧头
    tx_buf[0] = 0xED;

    // Byte 1: 视觉开关/模式 (SHIJUE_KAIGUAN)
    // STM32 判断 if(SHIJUE_KAIGUAN == 0xFC) 开启视觉
    tx_buf[1] = (data.mode > 0) ? 0xFC : 0x00;

    // 辅助函数：将 float 转为 STM32 期望的大端序 int32 字节流
    auto pack_big_endian = [](uint8_t* buf, float val) {
      int32_t int_val = static_cast<int32_t>(val * 100.0f);
      buf[0] = (int_val >> 24) & 0xFF; // b[2]<<24
      buf[1] = (int_val >> 16) & 0xFF; // b[3]<<16
      buf[2] = (int_val >> 8) & 0xFF;  // b[4]<<8
      buf[3] = (int_val >> 0) & 0xFF;  // b[5]
    };

    // Byte 2-5: Pitch_angle
    pack_big_endian(&tx_buf[2], data.pitch);
    // Byte 6-9: Yaw_angle
    pack_big_endian(&tx_buf[6], data.yaw);

    // Byte 10: Fire_Ctrl_Flag
    tx_buf[10] = (data.mode == 2) ? 0x01 : 0x00; // 假设 mode 2 为开火

    // Byte 11-12: 状态与瞄准 ID (对应 STM32 的 State_Now_V 和 Aiming_ID)
    tx_buf[11] = 0x01;
    tx_buf[12] = 0x01;

    // Byte 13: 帧尾
    tx_buf[13] = 0xEC;

    try {
      serial_.write(tx_buf, 14);
    } catch (const std::exception& e) {
      tools::logger()->warn("[Gimbal] Send to STM32 failed: {}", e.what());
    }
  }

  /**
   * @brief 【接收协议】读取从 STM32 发来的状态
   * 匹配 STM32 发送函数: Version_Data_Send (图片 2530b...)
   */
  void Gimbal::read_thread()
  {
    int error_count = 0;
    const size_t RX_SIZE = 17; // HAL_UART_Transmit(..., 17)
    uint8_t rx_buf[RX_SIZE];

    while (!quit_) {
      if (error_count > 500) {
        error_count = 0;
        reconnect();
        continue;
      }

      // 1. 寻找帧头 0xED
      uint8_t head;
      if (!read(&head, 1) || head != 0xED) {
        error_count++;
        continue;
      }
      rx_buf[0] = head;

      // 2. 读取后续 16 字节
      if (!read(rx_buf + 1, RX_SIZE - 1)) {
        error_count++;
        continue;
      }

      error_count = 0;
      auto t = std::chrono::steady_clock::now();

      // 3. 解析 Pitch, Yaw, Roll (小端序，对应 STM32 union b[0]-b[3])
      auto parse_union_float = [](uint8_t* b) {
        int32_t raw;
        memcpy(&raw, b, 4); // STM32 小端 union 直接拷贝
        return static_cast<float>(raw) / 100.0f;
      };

      float pitch_from_mcu = parse_union_float(&rx_buf[2]); // Pitch
      float yaw_from_mcu = parse_union_float(&rx_buf[6]);   // Yaw
      float roll_from_mcu = parse_union_float(&rx_buf[11]); // Roll

      // 4. 更新视觉内部状态
      std::lock_guard<std::mutex> lock(mutex_);
      state_.pitch = pitch_from_mcu;
      state_.yaw = yaw_from_mcu;
      state_.bullet_speed = static_cast<float>(rx_buf[16]); // Byte 16: bullet_speed

      // 5. 模式反馈 (Byte 1: vision_mode_select)
      if (rx_buf[1] == 0x07) {
        mode_ = GimbalMode::AUTO_AIM;
      }

      // 6. 构造四元数供预测算法使用
      Eigen::Quaterniond q =
          Eigen::AngleAxisd(yaw_from_mcu * M_PI / 180.0, Eigen::Vector3d::UnitZ()) *
          Eigen::AngleAxisd(pitch_from_mcu * M_PI / 180.0, Eigen::Vector3d::UnitY());
      queue_.push({q, t});
    }
  }

  // 辅助函数保持原样
  bool Gimbal::read(uint8_t* buffer, size_t size)
  {
    try {
      return serial_.read(buffer, size) == size;
    } catch (...) {
      return false;
    }
  }

  void Gimbal::reconnect()
  {
    serial_.close();
    std::this_thread::sleep_for(std::chrono::seconds(1));
    try {
      serial_.open();
      queue_.clear();
    } catch (...) {
    }
  }

} // namespace ecu