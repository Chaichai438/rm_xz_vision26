// 串腿步兵usb虚拟串口通信
#include "gimbal.hpp"
#include "tools/crc.hpp"

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
    auto baudrate = tools::read<std::uint32_t>(yaml, "baudrate");

    try {
      serial_.setPort(com_port);
      serial_.setBaudrate(baudrate);
      serial_.open();
    } catch (const std::exception& e) {
      tools::logger()->error("[Gimbal][Init] Failed to open serial: {}", e.what());
      exit(1);
    }

    thread_ = std::thread(&Gimbal::read_thread, this);
    tools::logger()->info("[Gimbal][Init] Visual-to-MCU communication link established.");
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

  /*
   * @brief [Command]Command格式转成VisionTOGimbal
   */

  /**
   * @brief 【发送协议】视觉发送给 STM32
   */
  void Gimbal::send(ecu::VisionToGimbal vtg)
  {
    // 1. 检查串口是否打开
    if (!serial_.isOpen()) {
      tools::logger()->warn("[Gimbal][send] Serial port not open, attempting to reopen...");
      try {
        serial_.open();
      } catch (const std::exception& e) {
        tools::logger()->error("[Gimbal][send] Failed to open serial port: {}", e.what());
        return; // 直接返回，避免进一步错误
      }
    }

    const size_t Tx_size = 57; // 数据长度
    uint8_t tx_buf[Tx_size];
    memset(tx_buf, 0, sizeof(tx_buf));

    // Byte 0: 帧头
    tx_buf[0] = 0xA5;

    // Byte 1: 视觉开关/模式
    tx_buf[1] = 0x00;

    auto pack_to_buf = [](uint8_t* buf, float val) {
      memcpy(buf, &val, sizeof(float)); // 直接拷贝 4 字节内存
    };

    // Yaw
    pack_to_buf(&tx_buf[5], vtg.yaw);
    // Pitch
    pack_to_buf(&tx_buf[9], vtg.pitch);

    // 开火标志
    tx_buf[4] = (vtg.mode == 1) ? 0x01 : 0x00;

    // 计算并添加 CRC16 (核心改动)
    uint16_t crc_val = tools::get_crc16(tx_buf, 55);

    // 按照小端序放入最后两字节：[55]为低位，[56]为高位
    tx_buf[55] = static_cast<uint8_t>(crc_val & 0xFF);
    tx_buf[56] = static_cast<uint8_t>((crc_val >> 8) & 0xFF);

    // 4. 发送逻辑
    const int MAX_RETRIES = 3;
    bool send_success = false;

    for (int retry = 0; retry < MAX_RETRIES && !send_success; ++retry) {
      try {
        if (!serial_.isOpen())
          serial_.open();

        // 注意：发送长度改为 Tx_size (14)
        serial_.write(tx_buf, Tx_size);
        send_success = true;
      } catch (const std::exception& e) {
        tools::logger()->warn("[Gimbal] Send retry {}/{}", retry + 1, MAX_RETRIES);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    }
  }

  /**
   * @brief 【接收协议】读取从 STM32 发来的状态
   */
  void Gimbal::read_thread()
  {
    int error_count = 0;
    const size_t RX_SIZE = 39;
    uint8_t rx_buf[RX_SIZE];

    while (!quit_) {
      // 核心优化】检查缓冲区积压，确保零延迟
      size_t available = serial_.available();
      if (available > RX_SIZE * 3) {
        serial_.flushInput();
        // 注意：flush之后当前循环没必要往下走了，因为缓冲区空了
        continue;
      }

      // 寻找帧头 0x5A
      uint8_t head = 0;
      if (!read(&head, 1))
        continue;
      if (head != 0x5A) {
        error_count++;
        continue;
      }
      rx_buf[0] = head; // 存入缓冲区供 CRC 校验

      // 先读满数据包，再校验
      // 建议在这里设置一个较短的超时，防止 read 永远阻塞
      if (serial_.read(rx_buf + 1, RX_SIZE - 1) != RX_SIZE - 1) {
        error_count++;
        continue;
      }

      // 【CRC 校验】数据读取完成后再校验
      if (!tools::check_crc16(rx_buf, RX_SIZE)) {
        // 如果 CRC 失败，说明这一帧数据受干扰错位了
        // tools::logger()->warn("[Gimbal] CRC Check Failed!");
        error_count++;
        // 校验失败建议 flush，防止下一帧从错误的位置开始读
        serial_.flushInput();
        continue;
      }

      error_count = 0;
      auto t = std::chrono::steady_clock::now();

      // 解析数据
      float pitch_from_mcu, yaw_from_mcu;
      memcpy(&pitch_from_mcu, &rx_buf[11], 4);
      memcpy(&yaw_from_mcu, &rx_buf[15], 4);

      // 模式更新 (rx_buf[1] 是协议定义的模式位)
      uint8_t mode_byte = rx_buf[1];
      GimbalMode temp_mode;
      if (mode_byte == 0x00)
        temp_mode = GimbalMode::IDLE;
      else if (mode_byte == 0xAA)
        temp_mode = GimbalMode::AUTO_AIM;
      else if (mode_byte == 0xAB)
        temp_mode = GimbalMode::SMALL_BUFF;
      else if (mode_byte == 0xAC)
        temp_mode = GimbalMode::BIG_BUFF;
      else
        temp_mode = GimbalMode::IDLE;

      // 更新状态
      {
        std::lock_guard<std::mutex> lock(mutex_);
        mode_ = temp_mode; // 建议在锁内更新 mode
        state_.pitch = pitch_from_mcu * 180.0 / M_PI;
        state_.yaw = yaw_from_mcu * 180.0 / M_PI;

        Eigen::Quaterniond q = Eigen::AngleAxisd(yaw_from_mcu, Eigen::Vector3d::UnitZ()) *
                               Eigen::AngleAxisd(pitch_from_mcu, Eigen::Vector3d::UnitY());
        queue_.push({q, t});
      }
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

  // 重连函数
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

// #include "gimbal.hpp"

// #include <Eigen/Geometry>
// #include <cstring>

// #include "tools/rotary_tool.hpp"
// #include "tools/logger.hpp"
// #include "tools/yaml.hpp"

// namespace ecu
// {
//   Gimbal::Gimbal(const std::string& config_path)
//   {
//     auto yaml = tools::load(config_path);
//     auto com_port = tools::read<std::string>(yaml, "com_port");

//     try {
//       serial_.setPort(com_port);
//       serial_.open();
//     } catch (const std::exception& e) {
//       tools::logger()->error("[Gimbal] Failed to open serial: {}", e.what());
//       exit(1);
//     }

//     thread_ = std::thread(&Gimbal::read_thread, this);
//     tools::logger()->info("[Gimbal] Visual-to-MCU communication link established.");
//   }

//   Gimbal::~Gimbal()
//   {
//     quit_ = true;
//     if (thread_.joinable())
//       thread_.join();
//     serial_.close();
//   }

//   GimbalMode Gimbal::mode() const
//   {
//     std::lock_guard<std::mutex> lock(mutex_);
//     return mode_;
//   }

//   GimbalState Gimbal::state() const
//   {
//     std::lock_guard<std::mutex> lock(mutex_);
//     return state_;
//   }

//   std::string Gimbal::str(GimbalMode mode) const
//   {
//     switch (mode) {
//     case GimbalMode::IDLE: return "IDLE";
//     case GimbalMode::AUTO_AIM: return "AUTO_AIM";
//     case GimbalMode::SMALL_BUFF: return "SMALL_BUFF";
//     case GimbalMode::BIG_BUFF: return "BIG_BUFF";
//     default: return "INVALID";
//     }
//   }

//   Eigen::Quaterniond Gimbal::q(std::chrono::steady_clock::time_point t)
//   {
//     while (true) {
//       auto [q_a, t_a] = queue_.pop();
//       auto [q_b, t_b] = queue_.front();
//       auto t_ab = tools::delta_time(t_a, t_b);
//       auto t_ac = tools::delta_time(t_a, t);
//       auto k = t_ac / t_ab;
//       Eigen::Quaterniond q_c = q_a.slerp(k, q_b).normalized();
//       if (t < t_a)
//         return q_c;
//       if (!(t_a < t && t <= t_b))
//         continue;

//       return q_c;
//     }
//   }

//   /**
//    * @brief 【发送协议】视觉发送给 STM32
//    * 匹配 STM32 接收函数: vision_data_UARTget (图片 fb736...)
//    */
//   void Gimbal::send(ecu::VisionToGimbal data)
//   {
//     // STM32 接收缓冲区 vision_buffer 长度为 14 (0-13)
//     uint8_t tx_buf[14];
//     memset(tx_buf, 0, sizeof(tx_buf));

//     // Byte 0: 帧头
//     tx_buf[0] = 0xED;

//     // Byte 1: 视觉开关/模式 (SHIJUE_KAIGUAN)
//     // STM32 判断 if(SHIJUE_KAIGUAN == 0xFC) 开启视觉
//     tx_buf[1] = (data.mode > 0) ? 0xFC : 0x00;

//     // 辅助函数：将 float 转为 STM32 期望的大端序 int32 字节流
//     auto pack_big_endian = [](uint8_t* buf, float val) {
//       int32_t int_val = static_cast<int32_t>(val * 100.0f);
//       buf[0] = (int_val >> 24) & 0xFF; // b[2]<<24
//       buf[1] = (int_val >> 16) & 0xFF; // b[3]<<16
//       buf[2] = (int_val >> 8) & 0xFF;  // b[4]<<8
//       buf[3] = (int_val >> 0) & 0xFF;  // b[5]
//     };

//     // Byte 2-5: Pitch_angle
//     pack_big_endian(&tx_buf[2], data.pitch);
//     // Byte 6-9: Yaw_angle
//     pack_big_endian(&tx_buf[6], data.yaw);

//     // Byte 10: Fire_Ctrl_Flag
//     tx_buf[10] = (data.mode == 2) ? 0x01 : 0x00; // 假设 mode 2 为开火

//     // Byte 11-12: 状态与瞄准 ID (对应 STM32 的 State_Now_V 和 Aiming_ID)
//     tx_buf[11] = 0x01;
//     tx_buf[12] = 0x01;

//     // Byte 13: 帧尾
//     tx_buf[13] = 0xEC;

//     try {
//       serial_.write(tx_buf, 14);
//     } catch (const std::exception& e) {
//       tools::logger()->warn("[Gimbal] Send to STM32 failed: {}", e.what());
//     }
//   }

//   /**
//    * @brief 【接收协议】读取从 STM32 发来的状态
//    * 匹配 STM32 发送函数: Version_Data_Send (图片 2530b...)
//    */
//   void Gimbal::read_thread()
//   {
//     int error_count = 0;
//     const size_t RX_SIZE = 17; // HAL_UART_Transmit(..., 17)
//     uint8_t rx_buf[RX_SIZE];

//     while (!quit_) {
//       if (error_count > 500) {
//         error_count = 0;
//         reconnect();
//         continue;
//       }

//       // 1. 寻找帧头 0xED
//       uint8_t head;
//       if (!read(&head, 1) || head != 0xED) {
//         error_count++;
//         continue;
//       }
//       rx_buf[0] = head;

//       // 2. 读取后续 16 字节
//       if (!read(rx_buf + 1, RX_SIZE - 1)) {
//         error_count++;
//         continue;
//       }

//       error_count = 0;
//       auto t = std::chrono::steady_clock::now();

//       // 3. 解析 Pitch, Yaw, Roll (小端序，对应 STM32 union b[0]-b[3])
//       auto parse_union_float = [](uint8_t* b) {
//         int32_t raw;
//         memcpy(&raw, b, 4); // STM32 小端 union 直接拷贝
//         return static_cast<float>(raw) / 100.0f;
//       };

//       float pitch_from_mcu = parse_union_float(&rx_buf[2]); // Pitch
//       float yaw_from_mcu = parse_union_float(&rx_buf[6]);   // Yaw
//       float roll_from_mcu = parse_union_float(&rx_buf[11]); // Roll

//       // 4. 更新视觉内部状态
//       std::lock_guard<std::mutex> lock(mutex_);
//       state_.pitch = pitch_from_mcu;
//       state_.yaw = yaw_from_mcu;
//       state_.bullet_speed = static_cast<float>(rx_buf[16]); // Byte 16: bullet_speed

//       // 5. 模式反馈 (Byte 1: vision_mode_select)
//       if (rx_buf[1] == 0x07) {
//         mode_ = GimbalMode::AUTO_AIM;
//       }

//       // 6. 构造四元数供预测算法使用
//       Eigen::Quaterniond q =
//           Eigen::AngleAxisd(yaw_from_mcu * M_PI / 180.0, Eigen::Vector3d::UnitZ()) *
//           Eigen::AngleAxisd(pitch_from_mcu * M_PI / 180.0, Eigen::Vector3d::UnitY());
//       queue_.push({q, t});
//     }
//   }

//   // 辅助函数保持原样
//   bool Gimbal::read(uint8_t* buffer, size_t size)
//   {
//     try {
//       return serial_.read(buffer, size) == size;
//     } catch (...) {
//       return false;
//     }
//   }

//   void Gimbal::reconnect()
//   {
//     serial_.close();
//     std::this_thread::sleep_for(std::chrono::seconds(1));
//     try {
//       serial_.open();
//       queue_.clear();
//     } catch (...) {
//     }
//   }

// } // namespace ecu

// 新英雄串口协议

// #include "gimbal.hpp"

// #include <Eigen/Geometry>
// #include <cstring>

// #include "tools/rotary_tool.hpp"
// #include "tools/logger.hpp"
// #include "tools/yaml.hpp"

// namespace ecu
// {
//   Gimbal::Gimbal(const std::string& config_path)
//   {
//     auto yaml = tools::load(config_path);
//     auto com_port = tools::read<std::string>(yaml, "com_port");

//     try {
//       serial_.setPort(com_port);
//       serial_.open();
//     } catch (const std::exception& e) {
//       tools::logger()->error("[Gimbal] Failed to open serial: {}", e.what());
//       exit(1);
//     }

//     thread_ = std::thread(&Gimbal::read_thread, this);
//     tools::logger()->info("[Gimbal] Visual-to-MCU communication link established.");
//   }

//   Gimbal::~Gimbal()
//   {
//     quit_ = true;
//     if (thread_.joinable())
//       thread_.join();
//     serial_.close();
//   }

//   GimbalMode Gimbal::mode() const
//   {
//     std::lock_guard<std::mutex> lock(mutex_);
//     return mode_;
//   }

//   GimbalState Gimbal::state() const
//   {
//     std::lock_guard<std::mutex> lock(mutex_);
//     return state_;
//   }

//   std::string Gimbal::str(GimbalMode mode) const
//   {
//     switch (mode) {
//     case GimbalMode::IDLE: return "IDLE";         //
//     case GimbalMode::AUTO_AIM: return "AUTO_AIM"; // 0x07
//     case GimbalMode::SMALL_BUFF: return "SMALL_BUFF";
//     case GimbalMode::BIG_BUFF: return "BIG_BUFF";
//     default: return "INVALID";
//     }
//   }

//   Eigen::Quaterniond Gimbal::q(std::chrono::steady_clock::time_point t)
//   {
//     while (true) {
//       auto [q_a, t_a] = queue_.pop();
//       auto [q_b, t_b] = queue_.front();
//       auto t_ab = tools::delta_time(t_a, t_b);
//       auto t_ac = tools::delta_time(t_a, t);
//       auto k = t_ac / t_ab;
//       Eigen::Quaterniond q_c = q_a.slerp(k, q_b).normalized();
//       if (t < t_a)
//         return q_c;
//       if (!(t_a < t && t <= t_b))
//         continue;

//       return q_c;
//     }
//   }

//   /**
//    * @brief 【发送协议】视觉发送给 STM32
//    */
//   void Gimbal::send(ecu::VisionToGimbal data)
//   {
//     // 1. 检查串口是否打开
//     if (!serial_.isOpen()) {
//       tools::logger()->warn("[Gimbal] Serial port not open, attempting to reopen...");
//       try {
//         serial_.open();
//       } catch (const std::exception& e) {
//         tools::logger()->error("[Gimbal] Failed to open serial port: {}", e.what());
//         return; // 直接返回，避免进一步错误
//       }
//     }

//     //  STM32 接收缓冲区 vision_buffer 长度为12
//     const size_t Tx_size = 13;
//     uint8_t tx_buf[Tx_size];
//     memset(tx_buf, 0, sizeof(tx_buf));

//     // Byte 0: 帧头
//     tx_buf[0] = 0xED;

//     // Byte 1: 视觉开关/模式
//     if (data.mode > 0) {
//       tx_buf[1] = 0xFC; // 开启视觉
//     } else {
//       tx_buf[1] = 0x00; // 关闭视觉
//     }

//     // 改进的辅助函数：添加范围检查
//     auto pack_big_endian = [](uint8_t* buf, float val) -> bool {
//       // 限制角度范围（根据实际需求调整）
//       const float MAX_ANGLE = 180.0f; // 最大±180度
//       if (val < -MAX_ANGLE || val > MAX_ANGLE) {
//         tools::logger()->warn("[Gimbal] Angle value out of range: {}", val);
//         val = std::max(-MAX_ANGLE, std::min(MAX_ANGLE, val));
//       }

//       // 检查是否转换后会溢出
//       double scaled_val = static_cast<double>(val) * 100.0;
//       if (scaled_val > 2147483647.0 || scaled_val < -2147483648.0) {
//         tools::logger()->error("[Gimbal] Angle value would overflow int32: {}", val);
//         return false;
//       }

//       int32_t int_val = static_cast<int32_t>(scaled_val);
//       buf[0] = static_cast<uint8_t>((int_val >> 24) & 0xFF);
//       buf[1] = static_cast<uint8_t>((int_val >> 16) & 0xFF);
//       buf[2] = static_cast<uint8_t>((int_val >> 8) & 0xFF);
//       buf[3] = static_cast<uint8_t>((int_val >> 0) & 0xFF);

//       //         auto pack_little_endian = [](uint8_t* buf, float val) {
//       //         int32_t int_val = static_cast<int32_t>(val * 100.0f);
//       //         buf[0] = static_cast<uint8_t>(int_val & 0xFF);
//       //         buf[1] = static_cast<uint8_t>((int_val >> 8) & 0xFF);
//       //         buf[2] = static_cast<uint8_t>((int_val >> 16) & 0xFF);
//       //         buf[3] = static_cast<uint8_t>((int_val >> 24) & 0xFF);
//       // };
//       return true;
//     };

//     // Pitch_angle (Byte 3-6)
//     if (!pack_big_endian(&tx_buf[2], data.pitch)) {
//       tools::logger()->error("[Gimbal] Failed to pack pitch angle");
//       return;
//     }

//     // Yaw_angle (Byte 7-10)
//     if (!pack_big_endian(&tx_buf[6], data.yaw)) {
//       tools::logger()->error("[Gimbal] Failed to pack yaw angle");
//       return;
//     }

//     // Fire_Ctrl_Flag (Byte 11)
//     tx_buf[10] = (data.mode == 2) ? 0x01 : 0x00;

//     // 状态与瞄准 ID (Byte 11-12)
//     // tx_buf[11] = 0x01;  // State_Now_V
//     // tx_buf[12] = 0x01;  // Aiming_ID

//     // 帧尾
//     tx_buf[11] = 0xEC;

//     // 尝试发送，包含重试机制
//     const int MAX_RETRIES = 3;
//     bool send_success = false;

//     for (int retry = 0; retry < MAX_RETRIES && !send_success; ++retry) {
//       try {
//         // 检查串口是否仍然打开
//         if (!serial_.isOpen()) {
//           tools::logger()->warn("[Gimbal] Serial closed, reopening... (retry {}/{})", retry + 1,
//                                 MAX_RETRIES);
//           serial_.open();
//         }

//         // 发送数据
//         serial_.write(tx_buf, Tx_size);
//         send_success = true;

//       } catch (const serial::PortNotOpenedException& e) {
//         tools::logger()->warn("[Gimbal] Port not open: {} (retry {}/{})", e.what(), retry + 1,
//                               MAX_RETRIES);
//         std::this_thread::sleep_for(std::chrono::milliseconds(50));

//       } catch (const serial::IOException& e) {
//         tools::logger()->warn("[Gimbal] IO Error: {} (retry {}/{})", e.what(), retry + 1,
//                               MAX_RETRIES);
//         std::this_thread::sleep_for(std::chrono::milliseconds(50));

//       } catch (const std::exception& e) {
//         tools::logger()->error("[Gimbal] Send failed: {}", e.what());
//         break; // 其他异常不重试
//       }
//     }

//     if (!send_success) {
//       tools::logger()->error("[Gimbal] Failed to send after {} retries", MAX_RETRIES);
//     }
//   }
//   /**
//    * @brief 【接收协议】读取从 STM32 发来的状态
//    * 匹配 STM32 发送函数: Version_Data_Send (图片 2530b...)
//    */
//   void Gimbal::read_thread()
//   {
//     int error_count = 0;
//     const size_t RX_SIZE = 12; // HAL_UART_Transmit(..., 17)
//     uint8_t rx_buf[RX_SIZE];

//     while (!quit_) {
//       if (error_count > 500) {
//         error_count = 0;
//         reconnect();
//         continue;
//       }

//       uint8_t head;
//       if (!read(&head, 1) || head != 0xED) {
//         error_count++;
//         continue;
//       }
//       rx_buf[0] = head;

//       if (!read(rx_buf + 1, RX_SIZE - 1)) {
//         error_count++;
//         continue;
//       }

//       error_count = 0;
//       auto t = std::chrono::steady_clock::now();

//       float pitch_from_mcu = 0.0f, yaw_from_mcu = 0.0f, roll_from_mcu = 0.0f;
//       int16_t bullet_speed = 0;
//       uint8_t color = 0;

//       memcpy(&pitch_from_mcu, &rx_buf[2], sizeof(float)); // bytes 3-6
//       memcpy(&yaw_from_mcu, &rx_buf[6], sizeof(float));   // bytes 7-10
//       // memcpy(&roll_from_mcu, &rx_buf[10], sizeof(int16_t)); // bytes 9-10
//       color = rx_buf[10];

//       // 4. 更新视觉内部状态
//       std::lock_guard<std::mutex> lock(mutex_);
//       state_.pitch = pitch_from_mcu / 100.0f;
//       state_.yaw = yaw_from_mcu / 100.0f;
//       state_.bullet_speed = static_cast<float>(rx_buf[11]);
//       // memcpy(&state_.bullet_speed, &rx_buf[10], sizeof(float));

//       // 5. 模式反馈 (Byte 1: vision_mode_select)
//       if (rx_buf[1] == 0x07) {
//         mode_ = GimbalMode::AUTO_AIM;
//       }

//       // 构造四元数供预测算法使用
//       Eigen::Quaterniond q =
//           Eigen::AngleAxisd(yaw_from_mcu / 100.0f * M_PI / 180.0, Eigen::Vector3d::UnitZ()) *
//           Eigen::AngleAxisd(pitch_from_mcu / 100.0f * M_PI / 180.0, Eigen::Vector3d::UnitY());
//       queue_.push({q, t});
//     }
//   }

//   // 辅助函数保持原样
//   bool Gimbal::read(uint8_t* buffer, size_t size)
//   {
//     try {
//       return serial_.read(buffer, size) == size;
//     } catch (...) {
//       return false;
//     }
//   }

//   void Gimbal::reconnect()
//   {
//     serial_.close();
//     std::this_thread::sleep_for(std::chrono::seconds(1));
//     try {
//       serial_.open();
//       queue_.clear();
//     } catch (...) {
//     }
//   }

// } // namespace ecu

// 黑铁兽步兵串口协议
// #include "gimbal.hpp"

// #include <opencv2/opencv.hpp>
// #include <Eigen/Geometry>
// #include <cstring>

// #include "tools/rotary_tool.hpp"
// #include "tools/logger.hpp"
// #include "tools/yaml.hpp"

// namespace ecu
// {
//   Gimbal::Gimbal(const std::string& config_path)
//   {
//     auto yaml = tools::load(config_path);
//     auto com_port = tools::read<std::string>(yaml, "com_port");

//     try {
//       serial_.setPort(com_port);
//       serial_.open();
//     } catch (const std::exception& e) {
//       tools::logger()->error("[Gimbal] Failed to open serial: {}", e.what());
//       exit(1);
//     }

//     thread_ = std::thread(&Gimbal::read_thread, this);
//     tools::logger()->info("[Gimbal] Visual-to-MCU communication link established.");
//   }

//   Gimbal::~Gimbal()
//   {
//     quit_ = true;
//     if (thread_.joinable())
//       thread_.join();
//     serial_.close();
//   }

//   GimbalMode Gimbal::mode() const
//   {
//     std::lock_guard<std::mutex> lock(mutex_);
//     return mode_;
//   }

//   GimbalState Gimbal::state() const
//   {
//     std::lock_guard<std::mutex> lock(mutex_);
//     return state_;
//   }

//   std::string Gimbal::str(GimbalMode mode) const
//   {
//     switch (mode) {
//     case GimbalMode::IDLE: return "IDLE";         //
//     case GimbalMode::AUTO_AIM: return "AUTO_AIM"; // 0x07
//     case GimbalMode::SMALL_BUFF: return "SMALL_BUFF";
//     case GimbalMode::BIG_BUFF: return "BIG_BUFF";
//     default: return "INVALID";
//     }
//   }

//   Eigen::Quaterniond Gimbal::q(std::chrono::steady_clock::time_point t)
//   {
//     while (true) {
//       auto [q_a, t_a] = queue_.pop();
//       auto [q_b, t_b] = queue_.front();
//       auto t_ab = tools::delta_time(t_a, t_b);
//       auto t_ac = tools::delta_time(t_a, t);
//       auto k = t_ac / t_ab;
//       Eigen::Quaterniond q_c = q_a.slerp(k, q_b).normalized();
//       if (t < t_a)
//         return q_c;
//       if (!(t_a < t && t <= t_b))
//         continue;

//       return q_c;
//     }
//   }

//   /**
//    * @brief 【发送协议】视觉发送给 STM32
//    */
//   void Gimbal::send(ecu::VisionToGimbal data)
//   {
//     // 1. 检查串口是否打开
//     if (!serial_.isOpen()) {
//       tools::logger()->warn("[Gimbal] Serial port not open, attempting to reopen...");
//       try {
//         serial_.open();
//       } catch (const std::exception& e) {
//         tools::logger()->error("[Gimbal] Failed to open serial port: {}", e.what());
//         return; // 直接返回，避免进一步错误
//       }
//     }

//     //  STM32 接收缓冲区 vision_buffer 长度为12
//     const size_t Tx_size = 12;
//     uint8_t tx_buf[Tx_size];
//     memset(tx_buf, 0, sizeof(tx_buf));

//     // Byte 1: 帧头
//     tx_buf[0] = 0xED;

//     // Byte 2: 视觉开关/模式
//     if (data.mode > 0) {
//       tx_buf[1] = 0xFC; // 开启视觉
//     } else {
//       tx_buf[1] = 0x00; // 关闭视觉
//     }

//     // 改进的辅助函数：添加范围检查
//     auto pack_big_endian = [](uint8_t* buf, float val) -> bool {
//       // 限制角度范围（根据实际需求调整）
//       const float MAX_ANGLE = 180.0f; // 最大±180度
//       if (val < -MAX_ANGLE || val > MAX_ANGLE) {
//         tools::logger()->warn("[Gimbal] Angle value out of range: {}", val);
//         val = std::max(-MAX_ANGLE, std::min(MAX_ANGLE, val));
//       }

//       // 检查是否转换后会溢出
//       double scaled_val = static_cast<double>(val) * 100.0;
//       if (scaled_val > 2147483647.0 || scaled_val < -2147483648.0) {
//         tools::logger()->error("[Gimbal] Angle value would overflow int32: {}", val);
//         return false;
//       }

//       int32_t int_val = static_cast<int32_t>(scaled_val);
//       buf[0] = static_cast<uint8_t>((int_val >> 24) & 0xFF);
//       buf[1] = static_cast<uint8_t>((int_val >> 16) & 0xFF);
//       buf[2] = static_cast<uint8_t>((int_val >> 8) & 0xFF);
//       buf[3] = static_cast<uint8_t>((int_val >> 0) & 0xFF);

//       //         auto pack_little_endian = [](uint8_t* buf, float val) {
//       //         int32_t int_val = static_cast<int32_t>(val * 100.0f);
//       //         buf[0] = static_cast<uint8_t>(int_val & 0xFF);
//       //         buf[1] = static_cast<uint8_t>((int_val >> 8) & 0xFF);
//       //         buf[2] = static_cast<uint8_t>((int_val >> 16) & 0xFF);
//       //         buf[3] = static_cast<uint8_t>((int_val >> 24) & 0xFF);
//       // };
//       return true;
//     };

//     // Pitch_angle (Byte 3-6)
//     if (!pack_big_endian(&tx_buf[2], data.pitch)) {
//       tools::logger()->error("[Gimbal] Failed to pack pitch angle");
//       return;
//     }

//     // Yaw_angle (Byte 7-10)
//     if (!pack_big_endian(&tx_buf[6], data.yaw)) {
//       tools::logger()->error("[Gimbal] Failed to pack yaw angle");
//       return;
//     }

//     // Fire_Ctrl_Flag (Byte 11)
//     tx_buf[10] = (data.mode == 2) ? 0x01 : 0x00;

//     // 状态与瞄准 ID (Byte 11-12)
//     // tx_buf[11] = 0x01;  // State_Now_V
//     // tx_buf[12] = 0x01;  // Aiming_ID

//     // 帧尾
//     tx_buf[11] = 0xEC;

//     // 尝试发送，包含重试机制
//     const int MAX_RETRIES = 3;
//     bool send_success = false;

//     for (int retry = 0; retry < MAX_RETRIES && !send_success; ++retry) {
//       try {
//         // 检查串口是否仍然打开
//         if (!serial_.isOpen()) {
//           tools::logger()->warn("[Gimbal] Serial closed, reopening... (retry {}/{})", retry + 1,
//                                 MAX_RETRIES);
//           serial_.open();
//         }

//         // 发送数据
//         serial_.write(tx_buf, Tx_size);
//         send_success = true;

//       } catch (const serial::PortNotOpenedException& e) {
//         tools::logger()->warn("[Gimbal] Port not open: {} (retry {}/{})", e.what(), retry + 1,
//                               MAX_RETRIES);
//         std::this_thread::sleep_for(std::chrono::milliseconds(50));

//       } catch (const serial::IOException& e) {
//         tools::logger()->warn("[Gimbal] IO Error: {} (retry {}/{})", e.what(), retry + 1,
//                               MAX_RETRIES);
//         std::this_thread::sleep_for(std::chrono::milliseconds(50));

//       } catch (const std::exception& e) {
//         tools::logger()->error("[Gimbal] Send failed: {}", e.what());
//         break; // 其他异常不重试
//       }
//     }

//     if (!send_success) {
//       tools::logger()->error("[Gimbal] Failed to send after {} retries", MAX_RETRIES);
//     }
//   }
//   /**
//    * @brief 【接收协议】读取从 STM32 发来的状态
//    *
//    */
//   void Gimbal::read_thread()
//   {
//     int error_count = 0;
//     const size_t RX_SIZE = 13; // HAL_UART_Transmit(..., 17)
//     uint8_t rx_buf[RX_SIZE];

//     while (!quit_) {
//       if (error_count > 500) {
//         error_count = 0;
//         reconnect();
//         continue;
//       }

//       uint8_t head, tail;
//       head = 0x99;
//       tail = 0X98;

//       if (!read(&head, 1) || head != 0x99) {
//         error_count++;
//         continue;
//       }
//       rx_buf[0] = head;

//       if (!read(rx_buf + 1, RX_SIZE - 1)) {
//         error_count++;
//         continue;
//       }

//       error_count = 0;
//       auto t = std::chrono::steady_clock::now();

//       float pitch_from_mcu = 0.0f, yaw_from_mcu = 0.0f, roll_from_mcu = 0.0f;
//       int16_t bullet_speed = 0;
//       uint8_t color = 0; // 0是蓝色 1是红色

//       if (rx_buf[1] == 0x00) {
//         mode_ = GimbalMode::IDLE;
//       }

//       else if (rx_buf[1] == 0xAA) {
//         mode_ = GimbalMode::AUTO_AIM;
//       }

//       else if (rx_buf[1] == 0xAB) {
//         mode_ = GimbalMode::SMALL_BUFF;
//       }

//       else if (rx_buf[1] == 0xAC) {
//         mode_ = GimbalMode::BIG_BUFF;
//       }

//       memcpy(&pitch_from_mcu, &rx_buf[2], sizeof(float)); // bytes 3-6
//       memcpy(&yaw_from_mcu, &rx_buf[6], sizeof(float));   // bytes 7-10
//       // memcpy(&roll_from_mcu, &rx_buf[10], sizeof(int16_t)); // bytes 9-10

//       // 我方颜色
//       color = rx_buf[10];

//       // 更新视觉内部状态
//       std::lock_guard<std::mutex> lock(mutex_);

//       state_.pitch = pitch_from_mcu * 180 / cv::;
//       state_.yaw = yaw_from_mcu / 100.0f;

//       state_.bullet_speed = static_cast<float>(rx_buf[11]);
//       // memcpy(&state_.bullet_speed, &rx_buf[10], sizeof(float));

//       // 5. 模式反馈 (Byte 1: vision_mode_select)

//       // 构造四元数供预测算法使用
//       Eigen::Quaterniond q =
//           Eigen::AngleAxisd(yaw_from_mcu / 100.0f * M_PI / 180.0, Eigen::Vector3d::UnitZ()) *
//           Eigen::AngleAxisd(pitch_from_mcu / 100.0f * M_PI / 180.0, Eigen::Vector3d::UnitY());
//       queue_.push({q, t});
//     }
//   }

//   // 辅助函数保持原样
//   bool Gimbal::read(uint8_t* buffer, size_t size)
//   {
//     try {
//       return serial_.read(buffer, size) == size;
//     } catch (...) {
//       return false;
//     }
//   }

//   void Gimbal::reconnect()
//   {
//     serial_.close();
//     std::this_thread::sleep_for(std::chrono::seconds(1));
//     try {
//       serial_.open();
//       queue_.clear();
//     } catch (...) {
//     }
//   }

// } // namespace ecu

// #include "gimbal.hpp"

// #include <Eigen/Geometry>
// #include <cstring>

// #include "tools/rotary_tool.hpp"
// #include "tools/logger.hpp"
// #include "tools/yaml.hpp"

// namespace ecu
// {
//   Gimbal::Gimbal(const std::string& config_path)
//   {
//     auto yaml = tools::load(config_path);
//     auto com_port = tools::read<std::string>(yaml, "com_port");

//     try {
//       serial_.setPort(com_port);
//       serial_.open();
//     } catch (const std::exception& e) {
//       tools::logger()->error("[Gimbal] Failed to open serial: {}", e.what());
//       exit(1);
//     }

//     thread_ = std::thread(&Gimbal::read_thread, this);
//     tools::logger()->info("[Gimbal] Visual-to-MCU communication link established.");
//   }

//   Gimbal::~Gimbal()
//   {
//     quit_ = true;
//     if (thread_.joinable())
//       thread_.join();
//     serial_.close();
//   }

//   GimbalMode Gimbal::mode() const
//   {
//     std::lock_guard<std::mutex> lock(mutex_);
//     return mode_;
//   }

//   GimbalState Gimbal::state() const
//   {
//     std::lock_guard<std::mutex> lock(mutex_);
//     return state_;
//   }

//   std::string Gimbal::str(GimbalMode mode) const
//   {
//     switch (mode) {
//     case GimbalMode::IDLE: return "IDLE";         //
//     case GimbalMode::AUTO_AIM: return "AUTO_AIM"; // 0x07
//     case GimbalMode::SMALL_BUFF: return "SMALL_BUFF";
//     case GimbalMode::BIG_BUFF: return "BIG_BUFF";
//     default: return "INVALID";
//     }
//   }

//   Eigen::Quaterniond Gimbal::q(std::chrono::steady_clock::time_point t)
//   {
//     while (true) {
//       auto [q_a, t_a] = queue_.pop();
//       auto [q_b, t_b] = queue_.front();
//       auto t_ab = tools::delta_time(t_a, t_b);
//       auto t_ac = tools::delta_time(t_a, t);
//       auto k = t_ac / t_ab;
//       Eigen::Quaterniond q_c = q_a.slerp(k, q_b).normalized();
//       if (t < t_a)
//         return q_c;
//       if (!(t_a < t && t <= t_b))
//         continue;

//       return q_c;
//     }
//   }

//   /*
//    * @brief [Command]Command格式转成VisionTOGimbal
//    */

//   /**
//    * @brief 【发送协议】视觉发送给 STM32
//    */
//   void Gimbal::send(ecu::VisionToGimbal vtg)
//   {
//     // 检查串口是否打开
//     if (!serial_.isOpen()) {
//       tools::logger()->warn("[Gimbal] Serial port not open, attempting to reopen...");
//       try {
//         serial_.open();
//       } catch (const std::exception& e) {
//         tools::logger()->error("[Gimbal] Failed to open serial port: {}", e.what());
//         return; // 直接返回，避免进一步错误
//       }
//     }

//     //  STM32 接收缓冲区 vision_buffer 长度为12
//     const size_t Tx_size = 12;
//     uint8_t tx_buf[Tx_size];
//     memset(tx_buf, 0, sizeof(tx_buf));

//     // Byte 1: 帧头
//     tx_buf[0] = 0xED;

//     // Byte 2: 视觉开关/模式
//     if (vtg.mode > 0) {
//       tx_buf[1] = 0xFC; // 开启视觉
//     } else {
//       tx_buf[1] = 0x00; // 关闭视觉
//     }

//     // 改进的辅助函数：添加范围检查
//     auto pack_big_endian = [](uint8_t* buf, float val) -> bool {
//       // 限制角度范围（根据实际需求调整）
//       const float MAX_ANGLE = 180.0f; // 最大±180度
//       if (val < -MAX_ANGLE || val > MAX_ANGLE) {
//         tools::logger()->warn("[Gimbal] Angle value out of range: {}", val);
//         val = std::max(-MAX_ANGLE, std::min(MAX_ANGLE, val));
//       }

//       // 检查是否转换后会溢出
//       double scaled_val = static_cast<double>(val) * 100.0;
//       if (scaled_val > 2147483647.0 || scaled_val < -2147483648.0) {
//         tools::logger()->error("[Gimbal] Angle value would overflow int32: {}", val);
//         return false;
//       }

//       // 大端序
//       int32_t int_val = static_cast<int32_t>(scaled_val);
//       buf[0] = static_cast<uint8_t>((int_val >> 24) & 0xFF);
//       buf[1] = static_cast<uint8_t>((int_val >> 16) & 0xFF);
//       buf[2] = static_cast<uint8_t>((int_val >> 8) & 0xFF);
//       buf[3] = static_cast<uint8_t>((int_val >> 0) & 0xFF);

//       // 小端序
//       // auto pack_little_endian = [](uint8_t* buf, float val) {
//       //   int32_t int_val = static_cast<int32_t>(val * 100.0f);
//       //   buf[0] = static_cast<uint8_t>(int_val & 0xFF);
//       //   buf[1] = static_cast<uint8_t>((int_val >> 8) & 0xFF);
//       //   buf[2] = static_cast<uint8_t>((int_val >> 16) & 0xFF);
//       //   buf[3] = static_cast<uint8_t>((int_val >> 24) & 0xFF);
//       // };
//       return true;
//     };

//     // Pitch_angle (Byte 3-6)
//     if (!pack_big_endian(&tx_buf[2], vtg.pitch)) {
//       tools::logger()->error("[Gimbal] Failed to pack pitch angle");
//       return;
//     }

//     // Yaw_angle (Byte 7-10)
//     if (!pack_big_endian(&tx_buf[6], vtg.yaw)) {
//       tools::logger()->error("[Gimbal] Failed to pack yaw angle");
//       return;
//     }

//     // Fire_Ctrl_Flag (Byte 11)
//     tx_buf[10] = (vtg.mode == 2) ? 0x01 : 0x00;

//     // 帧尾
//     tx_buf[11] = 0xEC;

//     // 尝试发送，包含重试机制
//     const int MAX_RETRIES = 3;
//     bool send_success = false;

//     for (int retry = 0; retry < MAX_RETRIES && !send_success; ++retry) {
//       try {
//         // 检查串口是否仍然打开
//         if (!serial_.isOpen()) {
//           tools::logger()->warn("[Gimbal] Serial closed, reopening... (retry {}/{})", retry + 1,
//                                 MAX_RETRIES);
//           serial_.open();
//         }

//         // 发送数据
//         serial_.write(tx_buf, Tx_size);
//         send_success = true;

//       } catch (const serial::PortNotOpenedException& e) {
//         tools::logger()->warn("[Gimbal] Port not open: {} (retry {}/{})", e.what(), retry + 1,
//                               MAX_RETRIES);
//         std::this_thread::sleep_for(std::chrono::milliseconds(50));

//       } catch (const serial::IOException& e) {
//         tools::logger()->warn("[Gimbal] IO Error: {} (retry {}/{})", e.what(), retry + 1,
//                               MAX_RETRIES);
//         std::this_thread::sleep_for(std::chrono::milliseconds(50));

//       } catch (const std::exception& e) {
//         tools::logger()->error("[Gimbal] Send failed: {}", e.what());
//         break; // 其他异常不重试
//       }
//     }

//     if (!send_success) {
//       tools::logger()->error("[Gimbal] Failed to send after {} retries", MAX_RETRIES);
//     }
//   }

//   /**
//    * @brief 【接收协议】读取从 STM32 发来的状态
//    */
//   void Gimbal::read_thread()
//   {
//     int error_count = 0;
//     const size_t RX_SIZE = 13; // HAL_UART_Transmit(..., 17)
//     uint8_t rx_buf[RX_SIZE];

//     while (!quit_) {
//       if (error_count > 500) {
//         error_count = 0;
//         reconnect();
//         continue;
//       }

//       uint8_t head;

//       if (!read(&head, 1) || head != 0x99) {
//         error_count++;
//         continue;
//       }
//       rx_buf[0] = head;

//       if (!read(rx_buf + 1, RX_SIZE - 1)) {
//         error_count++;
//         continue;
//       }

//       if (rx_buf[12] != 0x98) {
//         error_count++;
//         serial_.flushInput();
//         continue;
//       }

//       error_count = 0;
//       auto t = std::chrono::steady_clock::now();

//       float pitch_from_mcu = 0.0f, yaw_from_mcu = 0.0f;
//       int16_t bullet_speed = 0;
//       uint8_t color = 0; // 0是蓝色 1是红色

//       if (rx_buf[1] == 0x00) {
//         mode_ = GimbalMode::IDLE;
//       }

//       else if (rx_buf[1] == 0xAA) {
//         mode_ = GimbalMode::AUTO_AIM;
//       }

//       else if (rx_buf[1] == 0xAB) {
//         mode_ = GimbalMode::SMALL_BUFF;
//       }

//       else if (rx_buf[1] == 0xAC) {
//         mode_ = GimbalMode::BIG_BUFF;
//       }

//       memcpy(&yaw_from_mcu, &rx_buf[2], sizeof(float));   // bytes 3-6
//       memcpy(&pitch_from_mcu, &rx_buf[6], sizeof(float)); // bytes 7-10

//       // 我方颜色
//       color = rx_buf[10];

//       // 更新视觉内部状态
//       std::lock_guard<std::mutex> lock(mutex_);

//       // STM传过来的值为弧度
//       state_.pitch = pitch_from_mcu * 180.0 / M_PI;
//       state_.yaw = yaw_from_mcu * 180.0 / M_PI;

//       state_.bullet_speed = static_cast<float>(rx_buf[11]);
//       // memcpy(&state_.bullet_speed, &rx_buf[10], sizeof(float));

//       // 构造四元数供预测算法使用
//       Eigen::Quaterniond q = Eigen::AngleAxisd(yaw_from_mcu, Eigen::Vector3d::UnitZ()) *
//                              Eigen::AngleAxisd(pitch_from_mcu, Eigen::Vector3d::UnitY());
//       queue_.push({q, t});
//     }
//   }

//   // 辅助函数保持原样
//   bool Gimbal::read(uint8_t* buffer, size_t size)
//   {
//     try {
//       return serial_.read(buffer, size) == size;
//     } catch (...) {
//       return false;
//     }
//   }

//   // 重连函数
//   void Gimbal::reconnect()
//   {
//     serial_.close();
//     std::this_thread::sleep_for(std::chrono::seconds(1));
//     try {
//       serial_.open();
//       queue_.clear();
//     } catch (...) {
//     }
//   }

// } // namespace ecu