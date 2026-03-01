#pragma once

#include <opencv2/opencv.hpp>
#include <chrono>
#include <string>
#include "ecu/daheng/include/GxIAPI.h"
#include "ecu/daheng/include/DxImageProc.h"

#include "ecu/camera.hpp"

namespace ecu
{
  class Daheng : public ecu::CameraBase
  {
  public:
    /**
     * @param exposure_ms 曝光时间(ms)
     * @param gamma 伽马值
     * @param vid_pid 相机标识符（此处逻辑可根据需要调整为SN或Index）
     */
    Daheng(double exposure_ms, double gamma, const std::string& vid_pid);
    ~Daheng();

    // 核心读取接口，匹配 camera.cpp
    void read(cv::Mat& img, std::chrono::steady_clock::time_point& timestamp);

  private:
    GX_DEV_HANDLE hDevice_ = nullptr;
    GX_FRAME_DATA frameData_;

    // 图像处理缓冲区
    unsigned char* pRawBuffer_ = nullptr;
    unsigned char* pRGBBuffer_ = nullptr;
    int64_t nWidth_ = 0;
    int64_t nHeight_ = 0;
    int64_t nPayloadSize_ = 0;

    void checkStatus(GX_STATUS status, const std::string& msg);
  };
} // namespace ecu