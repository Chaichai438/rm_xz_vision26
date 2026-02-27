#pragma once

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace xz_vision
{
  // clang-format off
    enum Color
    {
        red,
        blue,
        extinguish,
        purple,
        gray
    };
    const std::vector<std::string> COLOR = {
        "red", 
        "blue", 
        "extinguish", 
        "purple",
        "gray"
    };

    // 装甲板类型
    enum ArmorType
    {
        big,
        small
    };
    const std::vector<std::string> ARMOR_TYPE = {
        "big", 
        "small"
    };

    // 装甲板名
    enum ArmorName
    {
        one ,
        two,
        three,
        four,
        five,
        sentry,
        outpost,
        base,
        not_armor
    };
    const std::vector<std::string>
        ARMOR_NAME = {
            "one", 
            "two", 
            "three", 
            "four", 
            "sentry", 
            "outpost", 
            "base", 
            "unkowned"
        };

    // 装甲板优先级
    enum ArmorPriority
    {
        first = 1,
        second,
        third,
        forth,
        fifth
    };

    
    const std::vector<std::tuple<Color, ArmorName, ArmorType>> armor_properties = {
    {blue, sentry, small},     {red, sentry, small},     {gray, sentry, small},
    {blue, one, small},        {red, one, small},        {gray, one, small},
    {blue, two, small},        {red, two, small},        {gray, two, small},
    {blue, three, small},      {red, three, small},      {gray, three, small},
    {blue, four, small},       {red, four, small},       {gray, four, small},
    {blue, outpost, small},    {red, outpost, small},    {gray, outpost, small},
    {blue, base, big},         {red, base, big},         {gray, base, big},      {purple, base, big},       
    {blue, base, small},       {red, base, small},       {gray, base, small},    {purple, base, small},    
    };
  // clang-format on

  // 灯条
  struct Lightbar {
    std::size_t id;                  // 灯条ID
    Color color;                     // 灯条颜色
    cv::Point2f center;              // 中心点
    cv::Point2f top, bottom;         // 顶点与底点中点
    cv::Point2f top2bottom;          // 顶到底方向向量
    std::vector<cv::Point2f> points; // 顶点集合（四点）
    double angle;                    // 倾斜角（单位：度）
    double angle_error;              // 角度误差
    double length;                   // 灯条长度
    double width;                    // 灯条宽度
    double ratio;                    // 长宽比
    cv::RotatedRect rotated_rect;    // 旋转矩形

    Lightbar(const cv::RotatedRect& rotated_rect, std::size_t id);
    Lightbar() {};
  };

  // 装甲板
  struct Armor {
    Color color;
    Lightbar left, right;            // 左右灯条
    cv::Point2f center;              // 不是对角线交点，不能作为实际中心！
    cv::Point2f center_norm;         // 归一化坐标
    std::vector<cv::Point2f> points; // 四个角点（顶点）

    double ratio;             // 两灯条的中点连线与长灯条的长度之比
    double side_ratio;        // 长灯条与短灯条的长度之比
    double rectangular_error; // 灯条和中点连线所成夹角与π/2的差值

    ArmorType type;
    ArmorName name;
    ArmorPriority priority;

    int class_id = -1;       // 分类ID
    cv::Mat pattern;         // 装甲板识别图案
    double confidence = 0.0; // 模型置信度
    bool duplicated;         // 是否重复识别
    cv::Rect box;
    Armor(const Lightbar& left, const Lightbar& right); // 传统视觉构造函数
    double ComputeRectangularError(const Lightbar& left, const Lightbar& right);
    Armor(int class_id, float confidence, const cv::Rect& box,
          std::vector<cv::Point2f> armor_keypoints);
    Armor(int class_id, float confidence, const cv::Rect& box,
          std::vector<cv::Point2f> armor_keypoints, cv::Point2f offset);
    Armor(int color_id, int num_id, float confidence, const cv::Rect& box,
          std::vector<cv::Point2f> armor_keypoints);
    Armor(int color_id, int num_id, float confidence, const cv::Rect& box,
          std::vector<cv::Point2f> armor_keypoints, cv::Point2f offset);

    Eigen::Vector3d xyz_in_camera; // 在相机坐标系下的位置
    Eigen::Vector3d xyz_in_gimbal; // 在云台坐标系下的位置
    Eigen::Vector3d xyz_in_world;  // 在世界坐标系下的位置

    // 姿态信息（单位：弧度）
    Eigen::Vector3d ypr_in_gimbal; // 在云台坐标系下的偏航(yaw)、俯仰(pitch)、滚转(roll),装甲板平面相对于云台指向的偏角。例如，如果
                                   // yaw 为 0，说明装甲板正对着你的枪口。
    Eigen::Vector3d ypr_in_world;  // 在世界坐标系下的欧拉角,装甲板相对于整个赛场的偏角。
    Eigen::Vector3d ypd_in_world;  // 在世界坐标系下的偏航(yaw)、俯仰(pitch)、距离(distance)

    // 用于优化的原始数据
    double yaw_raw = 0.0;
  };
} // namespace xz_vision
