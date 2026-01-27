#include "daheng.hpp"
#include <stdexcept>
#include <iostream>

namespace ecu::io
{
  Daheng::Daheng(double exposure_ms, double gamma, const std::string& vid_pid)
  {
    // 1. 初始化库
    checkStatus(GXInitLib(), "Init GxIAPI");

    // 2. 打开相机 (这里简化为按索引打开，实际可根据 vid_pid 筛选)
    uint32_t deviceNum = 0;
    checkStatus(GXUpdateDeviceList(&deviceNum, 1000), "Update Device List");
    if (deviceNum <= 0)
      throw std::runtime_error("No Daheng camera found!");

    GX_OPEN_PARAM openParam;
    openParam.accessMode = GX_ACCESS_EXCLUSIVE;
    openParam.openMode = GX_OPEN_INDEX;
    openParam.pszContent = (char*)"1";
    checkStatus(GXOpenDevice(&openParam, &hDevice_), "Open Device");

    // 3. 设置参数
    // 设置曝光 (SDK单位为us)
    checkStatus(GXSetFloat(hDevice_, GX_FLOAT_EXPOSURE_TIME, exposure_ms * 1000.0), "Set Exposure");
    // 设置Gamma
    checkStatus(GXSetEnum(hDevice_, GX_ENUM_GAMMA_MODE, GX_GAMMA_SELECTOR_USER), "Enable Gamma");
    checkStatus(GXSetFloat(hDevice_, GX_FLOAT_GAMMA, gamma), "Set Gamma");

    // 4. 准备缓冲区
    checkStatus(GXGetInt(hDevice_, GX_INT_PAYLOAD_SIZE, &nPayloadSize_), "Get Payload Size");
    checkStatus(GXGetInt(hDevice_, GX_INT_WIDTH, &nWidth_), "Get Width");
    checkStatus(GXGetInt(hDevice_, GX_INT_HEIGHT, &nHeight_), "Get Height");

    pRawBuffer_ = new unsigned char[(size_t)nPayloadSize_];
    pRGBBuffer_ = new unsigned char[(size_t)(nWidth_ * nHeight_ * 3)];

    // 5. 开流
    checkStatus(GXStreamOn(hDevice_), "Stream On");
  }

  void Daheng::read(cv::Mat& img, std::chrono::steady_clock::time_point& timestamp)
  {
    // 捕获一帧
    GX_STATUS status = GXiGetImage(hDevice_, &frameData_, 500); // 500ms超时
    timestamp = std::chrono::steady_clock::now();

    if (status == GX_STATUS_SUCCESS && frameData_.nStatus == GX_FRAME_STATUS_SUCCESS) {
      // 将 Bayer 或 Raw 格式转换为 RGB (使用大恒 DxImageProc 库)
      VxInt32 dxStatus = DxRaw8toRGB24((void*)frameData_.pImgBuf, pRGBBuffer_, (VxUint32)nWidth_,
                                       (VxUint32)nHeight_, RAW2RGB_NEIGHBOUR,
                                       DX_PIXEL_COLOR_FILTER(BAYER_RG), // 需根据相机实际格式调整
                                       false);

      if (dxStatus == DX_OK) {
        // 创建 Mat 并拷贝（或封装）
        cv::Mat bgr(nHeight_, nWidth_, CV_8UC3, pRGBBuffer_);
        cv::cvtColor(bgr, img, cv::COLOR_RGB2BGR);
      }
    } else {
      throw std::runtime_error("Failed to capture image from Daheng");
    }
  }

  void Daheng::checkStatus(GX_STATUS status, const std::string& msg)
  {
    if (status != GX_STATUS_SUCCESS) {
      throw std::runtime_error("Daheng Error: " + msg + " (Code: " + std::to_string(status) + ")");
    }
  }

  Daheng::~Daheng()
  {
    if (hDevice_) {
      GXStreamOff(hDevice_);
      GXCloseDevice(hDevice_);
    }
    GXCloseLib();
    delete[] pRawBuffer_;
    delete[] pRGBBuffer_;
  }
} // namespace ecu::io