#pragma once

namespace ecu
{
  struct Command {

    bool control;
    float pitch;           // 偏航角
    float yaw;             // 俯仰角
    bool shoot;            // 开火标志
    bool statu;            // 当前状态
    std::string aiming_id; // 瞄准ID
  };

} // namespace ecu