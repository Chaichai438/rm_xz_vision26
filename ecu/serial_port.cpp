#include "ecu/serial_port.hpp"

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <yaml-cpp/yaml.h>

#include "tools/logger.hpp"
#include "tools/math_tool.hpp"
#include "tools/yaml.hpp"

namespace ecu
{
  SerialPort::SerialPort(const std::string& config_path)
      : mode(Mode::idle)
      , bullet_speed(0)
      , ft_angle(0)
      , queue_(5000)
      , fd_(-1)
      , running_(true)
  {
    tools::logger()->info("[SerialPort] 初始化 SerialPort ...");

    // 读取配置文件
    auto dev_path = read_yaml(config_path);

    // 打开串口
    fd_ = open(dev_path.c_str(), O_RDWR | O_NOCTTY | O_SYNC);
    if (fd_ < 0) {
      throw std::runtime_error("[SerialBoard] 无法打开串口: " + dev_path);
    }

    // 配置串口参数
    struct termios tty{};
    if (tcgetattr(fd_, &tty) != 0) {
      throw std::runtime_error("[SerialBoard] 无法获取串口属性");
    }

    read_yaml(config_path); // 确保 port_ 和 baudrate_ 已初始化

    if (baudrate_ == 115200) {
      cfsetospeed(&tty, B115200);
      cfsetispeed(&tty, B115200);
    } else {
      tools::logger()->warn("[SerialBoard]不支持的波特率 {}, 使用 115200 作为默认值", baudrate_);
      cfsetospeed(&tty, B115200);
      cfsetispeed(&tty, B115200);
    }

    tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8; // 8位数据位
    tty.c_iflag &= ~IGNBRK;
    tty.c_lflag = 0;
    tty.c_oflag = 0;
    tty.c_cc[VMIN] = 1;  // 至少读取1字节
    tty.c_cc[VTIME] = 1; // 读取超时 0.1s
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);
    tty.c_cflag |= (CLOCAL | CREAD);
    tty.c_cflag &= ~(PARENB | PARODD); // 无校验
    tty.c_cflag &= ~CSTOPB;            // 1个停止位
    tty.c_cflag &= ~CRTSCTS;           // 禁止硬件流控

    if (tcsetattr(fd_, TCSANOW, &tty) != 0) {
      throw std::runtime_error("[SerialBoard] 无法设置串口属性");
    }

    tools::logger()->info("[SerialBoard] 成功打开串口: {}", dev_path);

    // 启动读线程
    read_thread_ = std::thread(&SerialPort::readLoop, this);
  }

  SerialPort::~SerialPort()
  {
    running_ = false;
    if (read_thread_.joinable())
      read_thread_.join();
    if (fd_ >= 0)
      close(fd_);
  }

  void SerialPort::readLoop()
  {
    tools::logger()->info("启动串口读取线程...");

    std::vector<uint8_t> recv_buf;
    recv_buf.reserve(256);

    const uint8_t FRAME_HEAD = 0x78;
    const uint8_t FRAME_TAIL = 0x76;
    const size_t FRAME_LEN = 14;

    uint8_t byte = 0;

    while (running_) {
      int n = read(fd_, &byte, 1);
      if (n <= 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        continue;
      }

      recv_buf.push_back(byte);

      // 检查帧头
      if (recv_buf.size() == 1 && recv_buf[0] != FRAME_HEAD) {
        recv_buf.clear();
        continue;
      }

      // 检查帧长度
      if (recv_buf.size() == FRAME_LEN) {
        if (recv_buf.front() == FRAME_HEAD && recv_buf.back() == FRAME_TAIL) {
          parseFrame(recv_buf);
        } else {
          tools::logger()->warn("[SerialBoard] 异常帧（帧头/帧尾错误）");
        }
        recv_buf.clear();
      }
    }
  }

  void SerialPort::parseFrame(const std::vector<uint8_t>& recv_buf)
  {
    if (fd_ < 0) {
      tools::logger()->warn("[SerialBoard] 异常帧（帧头/帧尾错误）");
      return;
    }
    // ✅ 解析 pitch 和 yaw
    float pitch = 0.0f, yaw = 0.0f;
    int16_t bullet_speed = 0;
    uint8_t color = 0, buff = 0;

    memcpy(&pitch, &recv_buf[1], sizeof(float));          // bytes 1-4
    memcpy(&yaw, &recv_buf[5], sizeof(float));            // bytes 5-8
    memcpy(&bullet_speed, &recv_buf[9], sizeof(int16_t)); // bytes 9-10
    color = recv_buf[11];
    buff = recv_buf[12];

    /// ✅ 将 pitch / yaw 转换为四元数（roll = 0）
    auto deg2rad = [](float deg) { return deg * static_cast<float>(M_PI) / 180.0f; };

    float yaw_rad = deg2rad(yaw);
    float pitch_rad = deg2rad(pitch);

    double cy = cos(yaw_rad * 0.5);
    double sy = sin(yaw_rad * 0.5);
    double cp = cos(pitch_rad * 0.5);
    double sp = sin(pitch_rad * 0.5);

    // 使用 (w, x, y, z) 的常见构造，x 对应 pitch，z 对应 yaw
    Eigen::Quaterniond q;
    q.w() = cy * cp;
    q.x() = sp * cy; // pitch 分量
    q.y() = 0.0;
    q.z() = sy * cp; // yaw 分量
    q.normalize();

    // ✅ 打包入队（保留时间戳）
    queue_.push({q, std::chrono::steady_clock::now()});

    // ✅ 调试输出
    // tools::logger()->info(
    //   "[SerialBoard] 接收数据: pitch={:.3f}, yaw={:.3f}, speed={}, color={}, buff={}, "
    //   "四元数=({:.3f}, {:.3f}, {:.3f}, {:.3f})",
    //   pitch, yaw, bullet_speed, color, buff,
    //   q.x(), q.y(), q.z(), q.w()
    // );
  }

  Eigen::Quaterniond SerialPort::imu_at(std::chrono::steady_clock::time_point timestamp)
  {
    if (data_behind_.timestamp < timestamp)
      data_ahead_ = data_behind_;
    while (true) {
      queue_.pop(data_behind_);
      if (data_behind_.timestamp > timestamp)
        break;
      data_ahead_ = data_behind_;
    }

    // 提取前后帧姿态
    Eigen::Quaterniond q_a = data_ahead_.q.normalized();
    Eigen::Quaterniond q_b = data_behind_.q.normalized();

    // 计算时间插值比例
    auto t_a = data_ahead_.timestamp;
    auto t_b = data_behind_.timestamp;
    std::chrono::duration<double> t_ab = t_b - t_a;
    std::chrono::duration<double> t_ac = timestamp - t_a;
    double k = (t_ab.count() == 0) ? 0.0 : std::clamp(t_ac.count() / t_ab.count(), 0.0, 1.0);

    // 改进部分：对欧拉角逐轴插值，而不是四元数整体 slerp
    Eigen::Vector3d euler_a = q_a.toRotationMatrix().eulerAngles(2, 1, 0); // Yaw-Pitch-Roll
    Eigen::Vector3d euler_b = q_b.toRotationMatrix().eulerAngles(2, 1, 0);

    Eigen::Vector3d euler_interp = euler_a + k * (euler_b - euler_a);

    // 仅保留独立 yaw 或 pitch 插值（防止耦合）
    // 例如：若当前只控制 yaw（pitch 应保持稳定）
    // 可根据你的主程序 axis 决定，但这里我们保持 pitch 独立插值
    // 若你确定当前是 yaw 测试，可固定 pitch
    // euler_interp[1] = euler_a[1]; // <- 取消注释则 pitch 不随 yaw 动

    // 重建四元数
    Eigen::AngleAxisd yawAngle(euler_interp[0], Eigen::Vector3d::UnitZ());
    Eigen::AngleAxisd pitchAngle(euler_interp[1], Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd rollAngle(euler_interp[2], Eigen::Vector3d::UnitX());
    Eigen::Quaterniond q_interp = yawAngle * pitchAngle * rollAngle;

    return q_interp.normalized();
  }

  void SerialPort::send(Command command) const
  {
    if (fd_ < 0) {
      tools::logger()->warn("[SerialBoard] 串口未打开，无法发送数据");
      return;
    }

    const uint8_t FRAME_HEAD = 0xED;
    const uint8_t FRAME_TAIL = 0xEC;
    const size_t FRAME_LEN = 14;

    uint8_t frame[FRAME_LEN] = {0};

    // 帧头
    frame[0] = FRAME_HEAD;

    frame[1] = 0xFC;

    // yaw（float，小端）
    memcpy(&frame[2], &command.pitch, sizeof(float));

    // pitch（float，小端）
    memcpy(&frame[6], &command.yaw, sizeof(float));

    // 发射标志
    frame[10] = 0x00; // 0x77 单发, 0x88 连发, 其他禁止

    // 模式标志
    frame[11] = 0x00; // 0x55 能量机关, 0x00 装甲板

    // 瞄准id
    frame[12] = 0x00;

    // 帧尾
    frame[13] = FRAME_TAIL;

    // 发送数据
    ssize_t written = write(fd_, frame, FRAME_LEN);
    if (written != (ssize_t)FRAME_LEN) {
      perror("[SerialBoard] 串口发送失败");
    } else {
      // tools::logger()->info(
      //   "[SerialBoard] 发送成功 -> yaw={:.2f}, pitch={:.2f}, shoot=0x{:02X},
      //   mode=0x{:02X}", command.yaw, command.pitch, command.shoot, command.control);
    }
    // std::this_thread::sleep_for(std::chrono::seconds(1));

    // ✅ 调试打印
    // std::cout << "[Serial] Send frame: ";
    // for (auto a : frame) {
    //   printf("%02X ", a);
    //   std::this_thread::sleep_for(std::chrono::milliseconds(2));
    // }
    // std::cout << std::endl;
  }

  std::string SerialPort::read_yaml(const std::string& config_path)
  {
    tools::logger()->info("读取配置文件中...");
    auto yaml = tools::load(config_path);

    if (!yaml["serial_port"]) {
      throw std::runtime_error("缺少 'serial_port' 配置项");
    }
    if (!yaml["baudrate"]) {
      throw std::runtime_error("缺少 'baudrate' 配置项");
    }

    port_ = yaml["serial_port"].as<std::string>();
    baudrate_ = yaml["baudrate"].as<int>();

    return port_;
  }

} // namespace ecu
