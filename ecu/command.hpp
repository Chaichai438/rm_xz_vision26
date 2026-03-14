#pragma once

namespace ecu
{
  struct Command {
    bool shoot; // 开火标志
    bool control;
    double yaw;
    double pitch;
  };

} // namespace ecu