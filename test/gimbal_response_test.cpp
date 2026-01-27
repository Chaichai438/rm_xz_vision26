#include "ecu/serial_port.hpp"
#include <opencv2/opencv.hpp>

const std::string keys =
    "{help h usage ? |     | 输出命令行参数说明 }"
    "{@config-path c | /home/chaichai/project/rm_xz_vision26/configs/how_to_set_params.yaml | "
    "yaml配置文件的路径}";

int main(int argc, char* argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  auto config_path = cli.get<std::string>(0);

  ecu::SerialPort serialport(config_path);
  ecu::Command cmd;
  cmd.yaw = 30.5f;    // 向右偏转 30.5 度
  cmd.pitch = -10.2f; // 向下俯仰 10.2 度
  cmd.shoot = 0x00;   // 单发射击
  cmd.statu = 0x00;   // 装甲板模式

  serialport.send(cmd); // 发送控制指令

  return 0;
}