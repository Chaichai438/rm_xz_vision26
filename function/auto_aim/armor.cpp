#include "armor.hpp"

#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>

namespace xz_vision
{
  Lightbar::Lightbar(const cv::RotatedRect& rotated_rect, std::size_t id)
      : id(id)
      , rotated_rect(rotated_rect)
  {
    std::vector<cv::Point2f> corners(4);
    rotated_rect.points(&corners[0]);
    std::sort(corners.begin(), corners.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
      return (a.y == b.y) ? (a.x < b.x) : (a.y < b.y);
    });

    center = rotated_rect.center;
    top = (corners[0] + corners[1]) * 0.5f;
    bottom = (corners[2] + corners[3]) * 0.5f;
    top2bottom = bottom - top;

    points.emplace_back(top);
    points.emplace_back(bottom);

    length = cv::norm(top2bottom);
    float top_w = cv::norm(corners[0] - corners[1]);
    float bot_w = cv::norm(corners[2] - corners[3]);
    width = (top_w + bot_w) * 0.5f;

    ratio = (width > 1e-6) ? (length / width) : 0.0;

    angle = std::atan2(top2bottom.y, top2bottom.x);
    if (angle < 0)
      angle += CV_PI;

    angle_error = std::abs(angle - CV_PI / 2);
    if (angle_error > CV_PI / 2)
      angle_error = CV_PI - angle_error; // 对称误差

    if (length < width)
      std::swap(length, width);
  };

  Armor::Armor(const Lightbar& left, const Lightbar& right)
      : left(left)
      , right(right)
      , duplicated(false)
  {
    color = left.color;
    center = (left.center + right.center) / 2;

    points.emplace_back(left.top);
    points.emplace_back(right.top);
    points.emplace_back(right.bottom);
    points.emplace_back(left.bottom);

    auto left2right = right.center - left.center;
    auto width = cv::norm(left2right);

    double avg_length = (left.length + right.length) * 0.5;
    ratio = width / avg_length;
    // 平均计算灯条可能会更平滑？为什么用 max_lightbar_length
    // 而不是平均值：取最大值是更保守的策略（偏严谨）。例如一侧灯条被截断或检测偏短，用最大值能避免把宽度除以一个异常小值导致ratio
    // 夸大

    double max_lightbar_length = std::max(left.length, right.length);
    double min_lightbar_length = std::min(left.length, right.length);
    side_ratio = max_lightbar_length / min_lightbar_length;

    rectangular_error = ComputeRectangularError(left, right);
  };

  // 自动权重调节 + 综合矩形误差计算
  double Armor::ComputeRectangularError(const Lightbar& left, const Lightbar& right)
  {
    // 平均灯条方向
    double angle_diff = std::abs(left.angle - right.angle);
    angle_diff = std::fmod(angle_diff, CV_PI);
    if (angle_diff > CV_PI / 2)
      angle_diff = CV_PI - angle_diff;

    // 与中轴线垂直程度
    auto left2right = right.center - left.center;
    double roll = std::atan2(left2right.y, left2right.x);
    double left_rect_err = std::abs(left.angle - roll - CV_PI / 2);
    double right_rect_err = std::abs(right.angle - roll - CV_PI / 2);
    double rect_err = std::max(left_rect_err, right_rect_err);

    // 动态权重计算
    double distance = cv::norm(left.center - right.center);
    double avg_length = 0.5 * (left.length + right.length);
    double ratio = distance / avg_length;

    // 根据比例自动调整权重（ratio: 0.2 ~ 1.0）
    double weight_angle = std::clamp(0.3 + 0.4 * ratio, 0.3, 0.7);
    double weight_rect = 1.0 - weight_angle;

    // 综合误差
    double rectangular_error = weight_angle * angle_diff + weight_rect * rect_err;

    return rectangular_error;
  }
  // 神经网络构造函数
  Armor::Armor(int class_id, float confidence, const cv::Rect& box,
               std::vector<cv::Point2f> armor_keypoints)
      : class_id(class_id)
      , confidence(confidence)
      , box(box)
      , points(armor_keypoints)
  {
    center =
        (armor_keypoints[0] + armor_keypoints[1] + armor_keypoints[2] + armor_keypoints[3]) / 4;
    auto left_width = cv::norm(armor_keypoints[0] - armor_keypoints[3]);
    auto right_width = cv::norm(armor_keypoints[1] - armor_keypoints[2]);
    auto max_width = std::max(left_width, right_width);
    auto top_length = cv::norm(armor_keypoints[0] - armor_keypoints[1]);
    auto bottom_length = cv::norm(armor_keypoints[3] - armor_keypoints[2]);
    auto max_length = std::max(top_length, bottom_length);
    auto left_center = (armor_keypoints[0] + armor_keypoints[3]) / 2;
    auto right_center = (armor_keypoints[1] + armor_keypoints[2]) / 2;
    auto left2right = right_center - left_center;
    auto roll = std::atan2(left2right.y, left2right.x);
    auto left_rectangular_error = std::abs(std::atan2((armor_keypoints[3] - armor_keypoints[0]).y,
                                                      (armor_keypoints[3] - armor_keypoints[0]).x) -
                                           roll - CV_PI / 2);
    auto right_rectangular_error =
        std::abs(std::atan2((armor_keypoints[2] - armor_keypoints[1]).y,
                            (armor_keypoints[2] - armor_keypoints[1]).x) -
                 roll - CV_PI / 2);
    rectangular_error = std::max(left_rectangular_error, right_rectangular_error);

    ratio = max_length / max_width;
    // color = class_id == 0 ? Color::blue : Color::red;

    if (class_id >= 0 && class_id < armor_properties.size()) {
      auto [color, name, type] = armor_properties[class_id];
      this->color = color;
      this->name = name;
      this->type = type;
    } else {
      this->color = blue;     // Default
      this->name = not_armor; // Default
      this->type = small;     // Default
    }
  }

  // 神经网络ROI构造函数
  Armor::Armor(int class_id, float confidence, const cv::Rect& box,
               std::vector<cv::Point2f> armor_keypoints, cv::Point2f offset)
      : class_id(class_id)
      , confidence(confidence)
      , box(box)
      , points(armor_keypoints)
  {
    std::transform(armor_keypoints.begin(), armor_keypoints.end(), armor_keypoints.begin(),
                   [&offset](const cv::Point2f& point) { return point + offset; });
    std::transform(points.begin(), points.end(), points.begin(),
                   [&offset](const cv::Point2f& point) { return point + offset; });
    center =
        (armor_keypoints[0] + armor_keypoints[1] + armor_keypoints[2] + armor_keypoints[3]) / 4;
    auto left_width = cv::norm(armor_keypoints[0] - armor_keypoints[3]);
    auto right_width = cv::norm(armor_keypoints[1] - armor_keypoints[2]);
    auto max_width = std::max(left_width, right_width);
    auto top_length = cv::norm(armor_keypoints[0] - armor_keypoints[1]);
    auto bottom_length = cv::norm(armor_keypoints[3] - armor_keypoints[2]);
    auto max_length = std::max(top_length, bottom_length);
    auto left_center = (armor_keypoints[0] + armor_keypoints[3]) / 2;
    auto right_center = (armor_keypoints[1] + armor_keypoints[2]) / 2;
    auto left2right = right_center - left_center;
    auto roll = std::atan2(left2right.y, left2right.x);
    auto left_rectangular_error = std::abs(std::atan2((armor_keypoints[3] - armor_keypoints[0]).y,
                                                      (armor_keypoints[3] - armor_keypoints[0]).x) -
                                           roll - CV_PI / 2);
    auto right_rectangular_error =
        std::abs(std::atan2((armor_keypoints[2] - armor_keypoints[1]).y,
                            (armor_keypoints[2] - armor_keypoints[1]).x) -
                 roll - CV_PI / 2);
    rectangular_error = std::max(left_rectangular_error, right_rectangular_error);

    ratio = max_length / max_width;
    // color = class_id == 0 ? Color::blue : Color::red;

    if (class_id >= 0 && class_id < armor_properties.size()) {
      auto [color, name, type] = armor_properties[class_id];
      this->color = color;
      this->name = name;
      this->type = type;
    } else {
      this->color = blue;     // Default
      this->name = not_armor; // Default
      this->type = small;     // Default
    }
  } // YOLOV5构造函数
  Armor::Armor(int color_id, int num_id, float confidence, const cv::Rect& box,
               std::vector<cv::Point2f> armor_keypoints)
      : confidence(confidence)
      , box(box)
      , points(armor_keypoints)
  {
    center =
        (armor_keypoints[0] + armor_keypoints[1] + armor_keypoints[2] + armor_keypoints[3]) / 4;
    auto left_width = cv::norm(armor_keypoints[0] - armor_keypoints[3]);
    auto right_width = cv::norm(armor_keypoints[1] - armor_keypoints[2]);
    auto max_width = std::max(left_width, right_width);
    auto top_length = cv::norm(armor_keypoints[0] - armor_keypoints[1]);
    auto bottom_length = cv::norm(armor_keypoints[3] - armor_keypoints[2]);
    auto max_length = std::max(top_length, bottom_length);
    auto left_center = (armor_keypoints[0] + armor_keypoints[3]) / 2;
    auto right_center = (armor_keypoints[1] + armor_keypoints[2]) / 2;
    auto left2right = right_center - left_center;
    auto roll = std::atan2(left2right.y, left2right.x);
    auto left_rectangular_error = std::abs(std::atan2((armor_keypoints[3] - armor_keypoints[0]).y,
                                                      (armor_keypoints[3] - armor_keypoints[0]).x) -
                                           roll - CV_PI / 2);
    auto right_rectangular_error =
        std::abs(std::atan2((armor_keypoints[2] - armor_keypoints[1]).y,
                            (armor_keypoints[2] - armor_keypoints[1]).x) -
                 roll - CV_PI / 2);
    rectangular_error = std::max(left_rectangular_error, right_rectangular_error);

    ratio = max_length / max_width;
    color = color_id == 0 ? Color::blue : color_id == 1 ? Color::red : Color::extinguish;
    name = num_id == 0  ? ArmorName::sentry
           : num_id > 5 ? ArmorName(num_id)
                        : ArmorName(num_id - 1); // TODO 考虑Bb
    type = num_id == 1 ? ArmorType::big : ArmorType::small;
  }

  // YOLOV5+ROI构造函数
  Armor::Armor(int color_id, int num_id, float confidence, const cv::Rect& box,
               std::vector<cv::Point2f> armor_keypoints, cv::Point2f offset)
      : confidence(confidence)
      , box(box)
      , points(armor_keypoints)
  {
    std::transform(armor_keypoints.begin(), armor_keypoints.end(), armor_keypoints.begin(),
                   [&offset](const cv::Point2f& point) { return point + offset; });
    std::transform(points.begin(), points.end(), points.begin(),
                   [&offset](const cv::Point2f& point) { return point + offset; });
    center =
        (armor_keypoints[0] + armor_keypoints[1] + armor_keypoints[2] + armor_keypoints[3]) / 4;
    auto left_width = cv::norm(armor_keypoints[0] - armor_keypoints[3]);
    auto right_width = cv::norm(armor_keypoints[1] - armor_keypoints[2]);
    auto max_width = std::max(left_width, right_width);
    auto top_length = cv::norm(armor_keypoints[0] - armor_keypoints[1]);
    auto bottom_length = cv::norm(armor_keypoints[3] - armor_keypoints[2]);
    auto max_length = std::max(top_length, bottom_length);
    auto left_center = (armor_keypoints[0] + armor_keypoints[3]) / 2;
    auto right_center = (armor_keypoints[1] + armor_keypoints[2]) / 2;
    auto left2right = right_center - left_center;
    auto roll = std::atan2(left2right.y, left2right.x);
    auto left_rectangular_error = std::abs(std::atan2((armor_keypoints[3] - armor_keypoints[0]).y,
                                                      (armor_keypoints[3] - armor_keypoints[0]).x) -
                                           roll - CV_PI / 2);
    auto right_rectangular_error =
        std::abs(std::atan2((armor_keypoints[2] - armor_keypoints[1]).y,
                            (armor_keypoints[2] - armor_keypoints[1]).x) -
                 roll - CV_PI / 2);
    rectangular_error = std::max(left_rectangular_error, right_rectangular_error);

    ratio = max_length / max_width;
    color = color_id == 0 ? Color::blue : color_id == 1 ? Color::red : Color::extinguish;
    name = num_id == 0 ? ArmorName::sentry : num_id > 5 ? ArmorName(num_id) : ArmorName(num_id - 1);
    type = num_id == 1 ? ArmorType::big : ArmorType::small;
  }

  // 神经网络构造函数
  Armor::Armor(int id, float confidence, const cv::Rect& box, const Lightbar& left,
               const Lightbar& right)
      : class_id(id)
      , confidence(confidence)
      , box(box)
      , left(left)
      , right(right)
  {
    color = left.color;
    center = (left.center + right.center) / 2;

    points.emplace_back(left.top);
    points.emplace_back(right.top);
    points.emplace_back(right.bottom);
    points.emplace_back(left.bottom);

    auto left2right = right.center - left.center;
    auto width_val = cv::norm(left2right);

    double avg_length = (left.length + right.length) * 0.5;
    ratio = width_val / avg_length;

    double max_lightbar_length = std::max(left.length, right.length);
    double min_lightbar_length = std::min(left.length, right.length);
    side_ratio = max_lightbar_length / min_lightbar_length;

    rectangular_error = ComputeRectangularError(left, right);

    if (id >= 0 && id < armor_properties.size()) {
      auto [color, name, type] = armor_properties[id];
      this->color = color;
      this->name = name;
      this->type = type;
    } else {
      this->color = blue;
      this->name = not_armor;
      this->type = small;
    }
  }

} // namespace xz_vision