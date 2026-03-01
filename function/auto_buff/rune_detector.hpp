#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <tools/logger.hpp>

#include "tools/math_tool.hpp"
#include "tools/imgs_tool.hpp"

/**
 * 0: 不显示
 * 1: 显示箭头、装甲板、中心、预测点
 * 2: 在 1 的基础上显示灯条、roi
 * 3: 在 2 的基础上显示二值化图片
 */
#define SHOW_IMAGE 0
#define CONSOLE_OUTPUT 0

namespace auto_buff
{
  inline static float image_width, image_height;

  enum class Status {
    SUCCESS,
    ARROW_FAILURE,
    ARMOR_FAILURE,
    CENTER_FAILURE
  }; // 成功，箭头检测失败，装甲板检测失败，中心R检测失败

  // 大符模式和小符模式
  enum class Mode { SMALL, BIG };

  // 颜色
  inline static cv::Scalar DRAW_COLOR;
  const inline static cv::Scalar RED{0, 0, 255};
  const inline static cv::Scalar BLUE{255, 0, 0};
  const inline static cv::Scalar GREEN{0, 255, 0};
  const inline static cv::Scalar WHITE{255, 255, 255};
  const inline static cv::Scalar YELLOW{0, 255, 255};
  const inline static cv::Scalar PURPLE{128, 0, 128};

  enum class Color { RED, BLUE };

  // 旋转方向
  enum class Direction { UNKNOWN, STABLE, ANTI_CLOCKWISE, CLOCKWISE };
  // 拟合曲线凹凸性
  enum class Convexity { UNKNOWN, CONCAVE, CONVEX };

  /**
   * @brief 灯条
   */
  struct LightLine {
    LightLine() = default;
    LightLine(const std::vector<cv::Point>& contour, const cv::Rect2f& global_roi,
              const cv::Rect2f& local_roi = cv::Rect2f(0, 0, image_width, image_height));
    std::vector<cv::Point> m_contour; // 轮廓点集
    double m_contour_area;            // 轮廓面积
    double m_area;                    // 外接旋转矩形面积
    cv::RotatedRect m_rotated_rect;   // 外接旋转矩形
    cv::Point2f m_tl;                 // 左上角点
    cv::Point2f m_tr;                 // 右上角点
    cv::Point2f m_bl;                 // 左下角点
    cv::Point2f m_br;                 // 右下角点
    cv::Point2f m_center;             // 中心点
    double m_length;                  // 长度
    double m_width;                   // 宽度
    double m_x;                       // 中心点 x 坐标
    double m_y;                       // 中心点 y 坐标
    double m_angle;                   // 旋转矩形角度
    double m_aspect_ratio;            // 旋转矩形长宽比
  };

  /**
   * @brief 装甲板
   */
  struct Armor {
    Armor() = default;
    void set(const LightLine& l1, const LightLine& l2);
    inline std::vector<cv::Point2f> getCornerPoints() { return {m_tlIn, m_trIn, m_blOut, m_brOut}; }
    void setCornerPoints(const std::vector<cv::Point2f>& points);
    LightLine m_inside;   // 内部灯条
    LightLine m_outside;  // 外部灯条
    cv::Point2f m_tlIn;   // 内部灯条左上角点
    cv::Point2f m_trIn;   // 内部灯条右上角点
    cv::Point2f m_blIn;   // 内部灯条左下角点
    cv::Point2f m_brIn;   // 内部灯条右下角点
    cv::Point2f m_tlOut;  // 外部灯条左上角点
    cv::Point2f m_trOut;  // 外部灯条右上角点
    cv::Point2f m_blOut;  // 外部灯条左下角点
    cv::Point2f m_brOut;  // 外部灯条右下角点
    cv::Point2f m_center; // 装甲板中心点
    double m_x;           // 装甲板中心 x 坐标
    double m_y;           // 装甲板中心 y 坐标
  };

  /**
   * @brief 箭头
   */
  struct Arrow {
    Arrow() = default;
    void set(const std::vector<LightLine>& points, const cv::Point2f& roi);
    std::vector<cv::Point> m_contour; // 轮廓点集
    cv::RotatedRect m_rotated_rect;   // 外接旋转矩形
    double m_length;                  // 长度
    double m_width;                   // 宽度
    cv::Point2f m_center;             // 中心点
    double m_angle;                   // 角度
    double m_aspect_ratio;            // 长宽比
    double m_area;                    // 面积
    double m_fill_ratio;              // 填充比例
  };

  /**
   * @brief 中心 R
   */
  struct CenterR {
    CenterR() = default;
    void set(const LightLine& contour);
    LightLine m_lightline;    // 中心 R 灯条
    cv::Point2f m_center_R;   // 中心 R 点
    cv::Rect m_bounding_rect; // 中心 R 最小正矩形
    double m_x;               // 中心 R x 坐标
    double m_y;               // 中心 R y 坐标
  };

  // 封装一帧图像及其相关的姿态信息和时间戳
  struct Frame {
    Frame() = default;
    Frame(const cv::Mat& image, const std::chrono::steady_clock::time_point& time, double pitch,
          double yaw, double roll)
        : m_image{image}
        , m_time{time}
        , m_roll{roll}
        , m_pitch{pitch}
        , m_yaw{yaw}
    {
    }
    cv::Mat m_image;                              // 图像数据
    std::chrono::steady_clock::time_point m_time; // 图像捕获的时间戳（单调时钟，适合测时间间隔）
    double m_roll, m_pitch, m_yaw;                // 设备/相机的姿态（旋转角度，单位通常是弧度）
    void set(const cv::Mat& image, const std::chrono::steady_clock::time_point& time, double pitch,
             double yaw, double roll);
    void set(const cv::Mat& image, const std::chrono::steady_clock::time_point& time);
  };

  class BuffDetection
  {
  public:
    BuffDetection(const std::string& config);
    inline void drawTargetPoint(const cv::Point2f& point)
    {
      cv::circle(m_image_show, point, 4, DRAW_COLOR, 2);
    }
    bool detect(const Frame& frame);

    inline void visualize() { cv::imshow("visualized", m_image_show); }

    /**
     * @brief
     * 得到像素坐标系特征点，分别为装甲板内灯条的左上，右上，外灯条的中上，左下，右下，中心R。
     * @return std::vector<cv::Point2f>
     */
    inline std::vector<cv::Point2f> getCameraPoints()
    {
      return {m_armor.m_tlIn,  m_armor.m_trIn,  (m_armor.m_tlOut + m_armor.m_trOut) * 0.5,
              m_armor.m_blOut, m_armor.m_brOut, m_centerR.m_center_R};
    }
    //============ 状态参数 ============//
    // private:
    cv::Mat m_image_arrow; // 检测箭头用的二值化图片
    cv::Mat m_image_armor; // 检测装甲板边框用的二值化图片
    cv::Mat m_image_show;  // 可视化图片

    cv::Mat m_local_mask;    // 局部 roi 的掩码
    cv::Rect2f m_global_roi; // 全局 roi ，用来圈定识别的范围，加快处理速度
    cv::Rect2f m_armor_roi;  // 装甲板 roi
    cv::Rect2f m_center_roi; // 中心 R roi
    CenterR m_centerR;       // 中心 R

    cv::Mat m_image_center; // 检测中心 R 用的二值化图片

    Status m_status; // 检测标志，包括成功、箭头检测失败、装甲板检测失败、中心 R 检测失败
    Armor m_armor;   // 装甲板
    Arrow m_arrow;   // 箭头

    std::chrono::steady_clock::time_point m_startTime; // 检测开始的时间戳
    std::chrono::steady_clock::time_point m_frameTime; // 当前帧的时间戳
    int m_lightArmorNum;                               // 点亮的装甲板数目

    inline static Mode MODE;

    //============ 初始化参数 ============//
    inline static Color color;

    bool whether_use_debug_pre;
    bool whether_use_debug_arrow;
    bool whether_use_debug_local_roi;
    bool whether_use_debug_armor;
    bool whether_use_debug_centerR;

    double param_thresh;
    double param_maxval;
    int param_kernel_width;
    int param_kernel_height;

    inline static double min_arrow_lightline_area;
    inline static double max_arrow_lightline_area;
    inline static double max_arrow_lightline_aspect_ratio;

    inline static int min_arrow_lightline_num;
    inline static int max_arrow_lightline_num;

    inline static double min_arrow_aspect_ratio;
    inline static double max_arrow_aspect_ratio;

    inline static double max_arrow_area;

    inline static double max_same_arrow_area_ratio;

    inline static double local_roi_distance_ratio;
    inline static float local_roi_width;

    inline static double min_armor_lightline_aspect_ratio;
    inline static double max_armor_lightline_aspect_ratio;

    inline static double min_armor_lightline_contour_area;
    inline static double max_armor_lightline_contour_area;

    inline static double min_armor_lightline_area;
    inline static double max_armor_lightline_area;

    inline static double max_same_armor_area_ratio;

    inline static double min_same_armor_distance;
    inline static double max_same_armor_distance;

    inline static double min_center_area;
    inline static double max_center_area;

    inline static double max_center_aspect_ratio;

    inline static float armor_center_vertical_distance_threshold;

    inline static double global_roi_length_ratio;

    inline static const double power_rune_radius{700.0};

    inline static double armor_outside_width;
    inline static double armor_outside_y;
    inline static double armor_outside_height;

    inline static double armor_inside_width;
    inline static double armor_inside_y;

    //============ 函数 ============//
    void preprocess_imgs(const Frame& frame);
    bool detect_armor();
    bool check_arrow();
    bool find_armor(Armor& armor, const std::vector<LightLine>& frames, const Arrow& arrow);
    static bool is_same_armor(const LightLine& l1, const LightLine& l2);
    void find_arrow_lightlines(const cv::Mat& binary, std::vector<LightLine>& lightlines,
                               const cv::Rect2f& roi);
    bool find_armor_lightlines(const cv::Mat& image, std::vector<LightLine>& lightlines,
                               const cv::Rect2f& globalRoi, const cv::Rect2f& localRoi);
    bool match_arrow(Arrow& arrow, const std::vector<LightLine>& lightlines, const cv::Rect2f& roi);
    static bool is_same_arrow(const LightLine& l1, const LightLine& l2);
    void set_local_roi();
    void set_global_roi();
    bool detect_centerR();
    bool findCenterLightlines(const cv::Mat& image, std::vector<LightLine>& lightlines,
                              const cv::Rect2f& globalRoi, const cv::Rect2f& localRoi);
    void set_armor();
    bool find_centerR(CenterR& center, const std::vector<LightLine>& lightlines, const Arrow& arrow,
                      const Armor& armor);
    static double calAngleBetweenLightlines(const LightLine& l1, const LightLine& l2);
    void reset_on_failure(const cv::Mat& image);

    //==================== 绘图函数 ====================
    void draw(const LightLine& lightline, const cv::Scalar& color, const int thickness = 1,
              const cv::Rect2f& localRoi = cv::Rect2f(0, 0, image_width, image_height));

    void draw(const cv::RotatedRect& rotatedRect, const cv::Scalar& color, const int thickness = 1,
              const cv::Rect2f& localRoi = cv::Rect2f(0, 0, image_width, image_height));

    void draw(const cv::Rect2f& rect, const cv::Scalar& color, const int thickness = 1,
              const cv::Rect2f& localRoi = cv::Rect2f(0, 0, image_width, image_height));

    void draw(const std::vector<cv::Point2f>& points, const cv::Scalar& color,
              const int thickness = 1,
              const cv::Rect2f& localRoi = cv::Rect2f(0, 0, image_width, image_height));

    void draw(const cv::Point2f* points, const size_t size, const cv::Scalar& color,
              const int thickness = 1,
              const cv::Rect2f& localRoi = cv::Rect2f(0, 0, image_width, image_height));
  };

} // namespace auto_buff