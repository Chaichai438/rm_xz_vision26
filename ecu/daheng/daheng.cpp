#include "daheng.hpp"
#include <stdexcept>
#include <iostream>

namespace ecu::io
{
  Daheng::Daheng(double exposure_ms, double gamma, const std::string& vid_pid)
  {
    // 1. 初始化库
    checkStatus(GXInitLib(), "Init GxIAPI");

    // 2. 更新并打开相机
    uint32_t deviceNum = 0;
    checkStatus(GXUpdateDeviceList(&deviceNum, 1000), "Update Device List");
    if (deviceNum <= 0)
      throw std::runtime_error("No Daheng camera found!");

    GX_OPEN_PARAM openParam;
    openParam.accessMode = GX_ACCESS_EXCLUSIVE; // 独占访问
    openParam.openMode = GX_OPEN_INDEX;         // 按索引打开
    openParam.pszContent = (char*)"1";
    checkStatus(GXOpenDevice(&openParam, &hDevice_), "Open Device");

    // 3. 设置参数 (SDK单位为us)
    checkStatus(GXSetFloat(hDevice_, GX_FLOAT_EXPOSURE_TIME, exposure_ms * 1000.0), "Set Exposure");

    // 设置Gamma
    checkStatus(GXSetEnum(hDevice_, GX_ENUM_GAMMA_MODE, GX_GAMMA_SELECTOR_USER), "Enable Gamma");
    checkStatus(GXSetFloat(hDevice_, GX_FLOAT_GAMMA, gamma), "Set Gamma");

    // 4. 获取图像属性并准备缓冲区
    checkStatus(GXGetInt(hDevice_, GX_INT_PAYLOAD_SIZE, &nPayloadSize_), "Get Payload Size");
    checkStatus(GXGetInt(hDevice_, GX_INT_WIDTH, &nWidth_), "Get Width");
    checkStatus(GXGetInt(hDevice_, GX_INT_HEIGHT, &nHeight_), "Get Height");

    pRawBuffer_ = new unsigned char[(size_t)nPayloadSize_];
    pRGBBuffer_ = new unsigned char[(size_t)(nWidth_ * nHeight_ * 3)];

    // 初始化 frameData 结构体
    frameData_.pImgBuf = pRawBuffer_;

    // 5. 开流
    checkStatus(GXStreamOn(hDevice_), "Stream On");
  }

  void Daheng::read(cv::Mat& img, std::chrono::steady_clock::time_point& timestamp)
  {
    // 捕获一帧：使用 GXGetImage
    GX_STATUS status = GXGetImage(hDevice_, &frameData_, 500); // 500ms超时
    timestamp = std::chrono::steady_clock::now();

    if (status == GX_STATUS_SUCCESS && frameData_.nStatus == GX_FRAME_STATUS_SUCCESS) {
      // 使用 DxImageProc 库将 Raw8 转换为 RGB24
      // 注意：BAYERRG 对应相机实际的 Bayer 布局，需根据设备调整
      VxInt32 dxStatus =
          DxRaw8toRGB24(frameData_.pImgBuf, pRGBBuffer_, (VxUint32)frameData_.nWidth,
                        (VxUint32)frameData_.nHeight, RAW2RGB_NEIGHBOUR, BAYERRG, false);

      if (dxStatus == DX_OK) {
        // 创建 Mat。由于 OpenCV 默认使用 BGR，此处进行颜色转换
        cv::Mat rgb(frameData_.nHeight, frameData_.nWidth, CV_8UC3, pRGBBuffer_);
        cv::cvtColor(rgb, img, cv::COLOR_RGB2BGR);
      } else {
        throw std::runtime_error("Image processing failed with status: " +
                                 std::to_string(dxStatus));
      }
    } else {
      throw std::runtime_error(
          "Failed to capture image from Daheng (Status: " + std::to_string(status) + ")");
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
      GXStreamOff(hDevice_);   // 停止流
      GXCloseDevice(hDevice_); // 关闭设备
    }
    GXCloseLib(); // 关闭库
    delete[] pRawBuffer_;
    delete[] pRGBBuffer_;
  }
} // namespace ecu::io