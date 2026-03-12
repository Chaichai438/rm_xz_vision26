#pragma once

#include <list>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>
#include <fmt/chrono.h>
#include <openvino/openvino.hpp>

#include "armor.hpp"
#include "classifier.hpp"
#include "./tools/draw_tool.hpp"
#include "./tools/logger.hpp"

namespace xz_vision
{
  // 装甲板检测类
  class Detector
  {
  public:
    Detector(const std::string& config_path, bool debug = true);

    std::list<Armor> detect(const cv::Mat& bgr_img, int frame_count = -1);
    std::list<Armor> detect_onnx(const cv::Mat& frame, int frame_count = -1);

  private:
    Classifier classifier_;

    double threshold_;
    double max_angle_error_;
    double min_lightbar_ratio_, max_lightbar_ratio_;
    double min_lightbar_length_;
    double min_armor_ratio_, max_armor_ratio_;
    double max_side_ratio_;
    double min_confidence_;
    double max_rectangular_error_;
    std::string model_path;
    std::string device;

    bool armor_debug_;
    std::string save_path_;
    ov::Core core;
    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;

    bool check_geometry(const Lightbar& lightbar) const;
    bool check_geometry(const Armor& armor) const;
    bool check_name(const Armor& armor) const;
    bool check_type(const Armor& armor) const;

    std::list<Armor>
    detect_lightbar_armors(const cv::Mat& bgr_img,
                           const std::vector<std::tuple<int, float, cv::Rect>>& dl_results,
                           int frame_count = -1);

    Color get_color(const cv::Mat& bgr_img, const std::vector<cv::Point>& contour) const;
    cv::Mat get_pattern(const cv::Mat& bgr_img, const Armor& armor) const;
    ArmorType get_type(const Armor& armor);
    cv::Point2f get_center_norm(const cv::Mat& bgr_img, const cv::Point2f& center) const;

    void save(const Armor& armor) const;
    void show_result(const cv::Mat& binary_img, const cv::Mat& bgr_img,
                     const std::list<Lightbar>& lightbars, const std::list<Armor>& armors,
                     int frame_count) const;

    std::list<Lightbar> last_lightbars;
  };
} // namespace xz_vision