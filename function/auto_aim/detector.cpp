#include "detector.hpp"

#include <filesystem>
namespace xz_vision
{
  Detector::Detector(const std::string& config_path, bool debug)
      : classifier_(config_path)
      , armor_debug_(0)

  {
    auto yaml = YAML::LoadFile(config_path);

    // armor_debug_ = yaml["armor_debug"].as<bool>();
    threshold_ = yaml["threshold"].as<double>();
    max_angle_error_ = yaml["max_angle_error"].as<double>() / 57.3; // degree to rad
    min_lightbar_ratio_ = yaml["min_lightbar_ratio"].as<double>();
    max_lightbar_ratio_ = yaml["max_lightbar_ratio"].as<double>();
    min_lightbar_length_ = yaml["min_lightbar_length"].as<double>();
    min_armor_ratio_ = yaml["min_armor_ratio"].as<double>();
    max_armor_ratio_ = yaml["max_armor_ratio"].as<double>();
    max_side_ratio_ = yaml["max_side_ratio"].as<double>();
    min_confidence_ = yaml["min_confidence"].as<double>();
    max_rectangular_error_ = yaml["max_rectangular_error"].as<double>() / 57.3; // degree to rad

    // save_path_ = "patterns";
    // std::filesystem::create_directory(save_path_);

    if (yaml["model_path"]) {
      model_path = yaml["model_path"].as<std::string>();
    } else {
      std::cerr << "错误：配置文件中缺少 'model_path'！" << std::endl;
    }

    std::cout << "model_path: " << model_path << std::endl;

    if (yaml["device"]) {
      device = yaml["device"].as<std::string>();
    }

    auto model = core.read_model(model_path);

    // 神经网络模型加载
    ov::preprocess::PrePostProcessor ppp(model);
    ppp.input()
        .tensor()
        .set_element_type(ov::element::u8)
        .set_layout("NHWC")
        .set_color_format(ov::preprocess::ColorFormat::BGR);
    ppp.input()
        .preprocess()
        .convert_element_type(ov::element::f32)
        .convert_color(ov::preprocess::ColorFormat::RGB)
        .scale(255.0f);
    ppp.input().model().set_layout("NCHW");

    model = ppp.build();
    compiled_model = core.compile_model(model, device);
    infer_request = compiled_model.create_infer_request();
  }

  // 传统视觉
  std::list<Armor> Detector::detect(const cv::Mat& bgr_img, int frame_count)
  {
    // 彩色图转灰度图
    cv::Mat gray_img;
    cv::cvtColor(bgr_img, gray_img, cv::COLOR_BGR2GRAY);

    // 进行二值化
    cv::Mat binary_img;
    cv::threshold(gray_img, binary_img, threshold_, 255, cv::THRESH_BINARY);
    // cv::imshow("binary_img", binary_img);

    // 获取轮廓点
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    // 获取灯条
    std::size_t lightbar_id = 0;
    std::list<Lightbar> lightbars;
    for (const auto& contour : contours) {
      auto rotated_rect = cv::minAreaRect(contour);
      auto lightbar = Lightbar(rotated_rect, lightbar_id);

      if (!check_geometry(lightbar))
        continue;

      lightbar.color = get_color(bgr_img, contour);
      lightbars.emplace_back(lightbar);
      lightbar_id += 1;
    }

    // 将灯条从左到右排序
    lightbars.sort([](const Lightbar& a, const Lightbar& b) { return a.center.x < b.center.x; });

    // 获取装甲板
    std::list<Armor> armors;
    for (auto left = lightbars.begin(); left != lightbars.end(); left++) {
      for (auto right = std::next(left); right != lightbars.end(); right++) {
        if (left->color != right->color)
          continue;

        auto armor = Armor(*left, *right);
        if (!check_geometry(armor))
          continue;

        armor.pattern = get_pattern(bgr_img, armor);
        classifier_.classify(armor);
        if (!check_name(armor))
          continue;

        armor.type = get_type(armor);
        if (!check_type(armor))
          continue;

        armor.center_norm = get_center_norm(bgr_img, armor.center);
        armors.emplace_back(armor);
      }
    }

    // 检查装甲板是否存在共用灯条的情况
    for (auto armor1 = armors.begin(); armor1 != armors.end(); armor1++) {
      for (auto armor2 = std::next(armor1); armor2 != armors.end(); armor2++) {
        if (armor1->left.id != armor2->left.id && armor1->left.id != armor2->right.id &&
            armor1->right.id != armor2->left.id && armor1->right.id != armor2->right.id) {
          continue;
        }

        // 装甲板重叠, 保留roi小的
        if (armor1->left.id == armor2->left.id || armor1->right.id == armor2->right.id) {
          auto area1 = armor1->pattern.cols * armor1->pattern.rows;
          auto area2 = armor2->pattern.cols * armor2->pattern.rows;
          if (area1 < area2)
            armor2->duplicated = true;
          else
            armor1->duplicated = true;
        }

        // 装甲板相连，保留置信度大的
        if (armor1->left.id == armor2->right.id || armor1->right.id == armor2->left.id) {
          if (armor1->confidence < armor2->confidence)
            armor1->duplicated = true;
          else
            armor2->duplicated = true;
        }
      }
    }

    armors.remove_if([&](const Armor& a) { return a.duplicated; });

    if (armor_debug_)
      show_result(binary_img, bgr_img, lightbars, armors, frame_count);

    return armors;
  }

  std::list<Armor> Detector::detect_onnx(const cv::Mat& frame, int frame_count)
  {
    // auto& cfg = rm_config::GetConfig();
    std::list<Armor> results;

    if (frame.empty()) {
      return results;
    }

    // ========== 1. 深度学习检测获取分类信息 ==========
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(416, 416));

    ov::Tensor input_tensor(ov::element::u8, {1, 416, 416, 3}, resized.data);
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();

    auto output = infer_request.get_output_tensor(0);
    float* data = output.data<float>();
    auto shape = output.get_shape();
    int channels = shape[1];
    int anchors = shape[2];

    cv::Mat res_mat(channels, anchors, CV_32F, data);
    res_mat = res_mat.t();

    float sx = (float)frame.cols / 416.0f;
    float sy = (float)frame.rows / 416.0f;

    // 存储深度学习结果
    std::vector<std::tuple<int, float, cv::Rect>> dl_results;

    for (int i = 0; i < anchors; i++) {
      float* row = res_mat.ptr<float>(i);

      auto max_ptr = std::max_element(row + 4, row + 4 + 15);
      float conf = *max_ptr;
      int cls_id = std::distance(row + 4, max_ptr);

      if (conf < min_confidence_)
        continue;

      float cx = row[0] * sx;
      float cy = row[1] * sy;
      float w = row[2] * sx;
      float h = row[3] * sy;

      cv::Rect box(cx - w / 2, cy - h / 2, w, h);
      dl_results.emplace_back(cls_id, conf, box);
    }

    // ========== 2. 调用灯条检测版本获取精确角点 ==========
    auto lightbar_armors = detect_lightbar_armors(frame, dl_results);

    // ========== 3. 为每个灯条结果匹配分类信息 ==========
    for (auto& armor : lightbar_armors) {
      bool matched = false;
      for (const auto& [cls_id, conf, box] : dl_results) {
        if (box.contains(armor.center)) {
          armor.class_id = cls_id;
          armor.confidence = conf;
          armor.box = box;

          // 根据class_id设置color、name、type
          if (cls_id >= 0 && cls_id < armor_properties.size()) {
            auto [color, name, type] = armor_properties[cls_id];
            armor.color = color;
            armor.name = name;
            armor.type = get_type(armor);
          }
          matched = true;
          break;
        }
      }

      // 即使没有匹配的深度学习结果，也保留灯条检测结果（使用默认分类）
      results.push_back(armor);
    }

    return results;
  }

  // ========== 灯条检测和配对函数 ==========
  std::list<Armor>
  Detector::detect_lightbar_armors(const cv::Mat& bgr_img,
                                   const std::vector<std::tuple<int, float, cv::Rect>>& dl_results,
                                   int frame_count)
  {
    // 彩色图转灰度图
    cv::Mat gray_img;
    cv::cvtColor(bgr_img, gray_img, cv::COLOR_BGR2GRAY);

    // 进行二值化
    cv::Mat binary_img;
    cv::threshold(gray_img, binary_img, threshold_, 255, cv::THRESH_BINARY);
    // cv::imshow("binary", binary_img);

    // 获取轮廓点
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    // 获取灯条
    std::size_t lightbar_id = 0;
    std::list<Lightbar> lightbars;
    for (const auto& contour : contours) {
      auto rotated_rect = cv::minAreaRect(contour);
      auto lightbar = Lightbar(rotated_rect, lightbar_id);

      if (!check_geometry(lightbar))
        continue;

      lightbar.color = get_color(bgr_img, contour);
      lightbars.emplace_back(lightbar);
      lightbar_id += 1;
    }

    // 将灯条从左到右排序
    lightbars.sort([](const Lightbar& a, const Lightbar& b) { return a.center.x < b.center.x; });

    // 废弃原因：这个装甲板匹配并没有用到严格的几何筛选导致，不妨可以试一试
    // 获取装甲板
    // std::list<Armor> armors;
    // for (auto left = lightbars.begin(); left != lightbars.end(); left++) {
    //   for (auto right = std::next(left); right != lightbars.end(); right++) {
    //     if (left->color != right->color)
    //       continue;

    //     // 暂时用-1的class_id，后面会用深度学习结果更新
    //     auto armor = Armor(-1, 0.0f, cv::Rect(), *left, *right);

    //     if (!check_geometry(armor))
    //       continue;

    //     // if (!check2_type(armor))
    //     //   continue;

    //     armor.center_norm = get_center_norm(bgr_img, armor.center);
    //     armors.emplace_back(armor);
    //   }
    // }

    std::list<Armor> armors;
    // 遍历每一个深度学习检测到的框 (BBox)
    for (const auto& [cls_id, conf, box] : dl_results) {
      // 获取该框对应的属性（颜色、类型等）
      auto [expected_color, name, type] = armor_properties[cls_id];

      // 筛选出在该 BBox 范围内的灯条
      std::vector<Lightbar*> candidate_bars;
      for (auto& lb : lightbars) {
        if (box.contains(lb.center)) {
          // 增加颜色约束：如果灯条颜色与模型预测颜色不符，排除
          // 注意：如果你的get_color不够准，可以适当放宽这个条件
          if (lb.color == expected_color) {
            candidate_bars.push_back(&lb);
          }
        }
      }

      // 只在同一个 BBox 内部的灯条之间进行两两匹配
      if (candidate_bars.size() < 2)
        continue;

      for (size_t i = 0; i < candidate_bars.size(); ++i) {
        for (size_t j = i + 1; j < candidate_bars.size(); ++j) {
          auto& left = *candidate_bars[i];
          auto& right = *candidate_bars[j];

          // 创建临时装甲板进行几何校验
          auto armor = Armor(cls_id, conf, box, left, right);
          armor.color = expected_color;
          armor.name = name;
          armor.type = type;

          // 核心：此时 check_geometry 可以根据 armor.type (大小装甲板)
          // 使用不同的阈值（比例、角度等）进行精确过滤
          if (!check_geometry(armor))
            continue;

          armor.center_norm = get_center_norm(bgr_img, armor.center);
          armors.emplace_back(armor);
        }
      }
    }

    // 检查装甲板是否存在共用灯条的情况
    for (auto armor1 = armors.begin(); armor1 != armors.end(); armor1++) {
      for (auto armor2 = std::next(armor1); armor2 != armors.end();) {
        if (armor1->left.id != armor2->left.id && armor1->left.id != armor2->right.id &&
            armor1->right.id != armor2->left.id && armor1->right.id != armor2->right.id) {
          armor2++;
          continue;
        }

        // 装甲板重叠, 保留面积小的
        if (armor1->left.id == armor2->left.id || armor1->right.id == armor2->right.id) {
          auto area1 = cv::contourArea(armor1->points);
          auto area2 = cv::contourArea(armor2->points);
          if (area1 < area2) {
            armor2 = armors.erase(armor2);
          } else {
            armor1 = armors.erase(armor1);
            break;
          }
          continue;
        }

        // 装甲板相连，保留置信度大的
        if (armor1->left.id == armor2->right.id || armor1->right.id == armor2->left.id) {
          if (armor1->confidence < armor2->confidence) {
            armor1 = armors.erase(armor1);
            break;
          } else {
            armor2 = armors.erase(armor2);
            continue;
          }
        }
      }
    }

    // 保存灯条供外部使用
    last_lightbars.assign(lightbars.begin(), lightbars.end());

    if (armor_debug_)
      show_result(binary_img, bgr_img, lightbars, armors, frame_count);

    return armors;
  }

  bool Detector::check_geometry(const Lightbar& lightbar) const
  {
    auto angle_ok = lightbar.angle_error < max_angle_error_;
    auto ratio_ok = lightbar.ratio > min_lightbar_ratio_ && lightbar.ratio < max_lightbar_ratio_;
    auto length_ok = lightbar.length > min_lightbar_length_;
    return angle_ok && ratio_ok && length_ok;
  }

  bool Detector::check_geometry(const Armor& armor) const
  {
    // 1. 获取两灯条的高度比 (长/短)
    double h_min = std::min(armor.left.length, armor.right.length);
    double h_max = std::max(armor.left.length, armor.right.length);
    double height_ratio = h_min / h_max;
    if (height_ratio < 0.6)
      return false; // 高度差太大，剔除

    // 2. 获取长宽比 (两灯条中心间距 / 平均灯条高度)
    double avg_h = (armor.left.length + armor.right.length) / 2.0;
    double dist = cv::norm(armor.left.center - armor.right.center);
    double aspect_ratio = dist / avg_h;

    // 3. 根据装甲板类型判断比例是否合规
    if (armor.type == ArmorType::big) {
      // 大装甲板标准 (Hero)
      if (aspect_ratio < 1.5 || aspect_ratio > 5.0)
        return false;
    } else {
      // 小装甲板标准 (Infantry / Sentry / Outpost)
      if (aspect_ratio < 1.5 || aspect_ratio > 3.0)
        return false;
    }

    // 4. 灯条平行度（夹角误差）
    double angle_diff = std::abs(armor.left.angle - armor.right.angle);
    if (angle_diff > 8.0)
      return false;

    return true;
  }

  bool Detector::check_name(const Armor& armor) const
  {
    auto name_ok = armor.name != ArmorName::not_armor;
    auto confidence_ok = armor.confidence > min_confidence_;

    // 保存不确定的图案，用于分类器的迭代
    if (name_ok && !confidence_ok)
      save(armor);

    // // 出现 5号 则显示 debug 信息。但不过滤。
    // if (armor.name == ArmorName::five)
    //   tools::logger()->debug("See pattern 5");

    return name_ok && confidence_ok;
  }

  bool Detector::check_type(const Armor& armor) const
  {
    auto name_ok = armor.type == ArmorType::small
                       ? (armor.name != ArmorName::one && armor.name != ArmorName::base)
                       : (armor.name == ArmorName::one || armor.name == ArmorName::base);

    // 保存异常的图案，用于分类器的迭代
    if (!name_ok) {
      tools::logger()->debug("see strange armor: {} {}", ARMOR_TYPE[armor.type],
                             ARMOR_NAME[armor.name]);
      save(armor);
    }

    return name_ok;
  }

  Color Detector::get_color(const cv::Mat& bgr_img, const std::vector<cv::Point>& contour) const
  {
    int red_sum = 0, blue_sum = 0;

    for (const auto& point : contour) {
      red_sum += bgr_img.at<cv::Vec3b>(point)[2];
      blue_sum += bgr_img.at<cv::Vec3b>(point)[0];
    }

    return blue_sum > red_sum ? Color::blue : Color::red;
  }

  cv::Mat Detector::get_pattern(const cv::Mat& bgr_img, const Armor& armor) const
  {
    // 延长灯条获得装甲板角点
    // 1.125 = 0.5 * armor_height / lightbar_length = 0.5 * 126mm / 56mm
    auto tl = armor.left.center - armor.left.top2bottom * 1.125;
    auto bl = armor.left.center + armor.left.top2bottom * 1.125;
    auto tr = armor.right.center - armor.right.top2bottom * 1.125;
    auto br = armor.right.center + armor.right.top2bottom * 1.125;

    auto roi_left = std::max<int>(std::min(tl.x, bl.x), 0);
    auto roi_top = std::max<int>(std::min(tl.y, tr.y), 0);
    auto roi_right = std::min<int>(std::max(tr.x, br.x), bgr_img.cols);
    auto roi_bottom = std::min<int>(std::max(bl.y, br.y), bgr_img.rows);
    auto roi_tl = cv::Point(roi_left, roi_top);
    auto roi_br = cv::Point(roi_right, roi_bottom);
    auto roi = cv::Rect(roi_tl, roi_br);

    return bgr_img(roi);
  }

  ArmorType Detector::get_type(const Armor& armor)
  {
    /// 优先根据当前armor.ratio判断
    /// TODO: 25赛季是否还需要根据比例判断大小装甲？能否根据图案直接判断？

    if (armor.ratio > 3.0) {
      tools::logger()->debug("[Detector] get armor type by ratio: BIG {} {:.2f}",
                             ARMOR_NAME[armor.name], armor.ratio);
      return ArmorType::big;
    }

    if (armor.ratio < 2.5) {
      tools::logger()->debug("[Detector] get armor type by ratio: SMALL {} {:.2f}",
                             ARMOR_NAME[armor.name], armor.ratio);
      return ArmorType::small;
    }

    // tools::logger()->debug("[Detector] get armor type by name: {}", ARMOR_NAMES[armor.name]);

    // 英雄、基地只能是大装甲板
    if (armor.name == ArmorName::one) {
      return ArmorType::big;
    }

    // 其他所有（工程、哨兵、前哨站、步兵）都是小装甲板
    /// TODO: 基地顶装甲是小装甲板
    return ArmorType::small;
  }
  cv::Point2f Detector::get_center_norm(const cv::Mat& bgr_img, const cv::Point2f& center) const
  {
    auto h = bgr_img.rows;
    auto w = bgr_img.cols;
    return {center.x / w, center.y / h};
  }

  void Detector::save(const Armor& armor) const
  {
    auto file_name = fmt::format("{:%Y-%m-%d_%H-%M-%S}", std::chrono::system_clock::now());
    auto img_path = fmt::format("{}/{}_{}.jpg", save_path_, ARMOR_NAME[armor.name], file_name);
    cv::imwrite(img_path, armor.pattern);
  }

  // void Detector::show_result(const cv::Mat& binary_img, const cv::Mat& bgr_img,
  //                            const std::list<Lightbar>& lightbars, const std::list<Armor>&
  //                            armors, int frame_count) const
  // {
  //   auto detection = bgr_img.clone();
  //   tools::draw_text(detection, fmt::format("[{}]", frame_count), {10, 30}, {255, 255, 255});

  //   for (const auto& lightbar : lightbars) {
  //     auto info = fmt::format("{:.1f} {:.1f} {:.1f} {}", lightbar.angle_error * 57.3,
  //                             lightbar.ratio, lightbar.length, COLOR[lightbar.color]);
  //     tools::draw_text(detection, info, lightbar.top, {0, 255, 255});
  //     tools::draw_points(detection, lightbar.points, {0, 255, 255}, 3);
  //   }

  //   for (const auto& armor : armors) {
  //     auto info = fmt::format("{:.2f} {:.2f} {:.1f} {:.2f} {} {}", armor.ratio, armor.side_ratio,
  //                             armor.rectangular_error * 57.3, armor.confidence,
  //                             ARMOR_NAME[armor.name], ARMOR_TYPE[armor.type]);
  //     tools::draw_points(detection, armor.points, {0, 255, 0});
  //     tools::draw_text(detection, info, armor.left.bottom, {0, 255, 0});
  //   }

  //   cv::Mat binary_img2;
  //   cv::resize(binary_img, binary_img2, {}, 0.5, 0.5); // 显示时缩小图片尺寸
  //   cv::resize(detection, detection, {}, 0.5, 0.5);    // 显示时缩小图片尺寸

  //    cv::imshow("threshold", binary_img2);
  //   cv::imshow("detection", detection);
  // }

  void Detector::show_result(const cv::Mat& binary_img, const cv::Mat& bgr_img,
                             const std::list<Lightbar>& lightbars, const std::list<Armor>& armors,
                             int frame_count) const
  {
    auto detection = bgr_img.clone();
    tools::draw_text(detection, fmt::format("[{}]", frame_count), {10, 30}, {255, 255, 255});

    // 绘制灯条
    for (const auto& lightbar : lightbars) {
      // auto info = fmt::format("{:.1f} {:.1f} {:.1f} {}", lightbar.angle_error * 57.3,
      //                         lightbar.ratio, lightbar.length, COLOR[lightbar.color]);
      // tools::draw_text(detection, info, lightbar.top, {0, 255, 255});

      // if (lightbar.ratio > 1.1 && lightbar.ratio < 20.0 &&
      //     lightbar.width > 0.3 && lightbar.length > 2.0 && lightbar.length < 150)
      // {
      //     if ((lightbar.angle <= 180 && lightbar.angle >= 150) ||
      //         (lightbar.angle <= 30 && lightbar.angle >= 0))
      //     {
      //         tools::draw_points(detection, lightbar.points, {0, 255, 255}, 3);
      //     }
      // }

      // 改为：允许更大角度范围的灯条显示
      if (lightbar.ratio > 1.1 && lightbar.ratio < 20.0 && lightbar.width > 0.3 &&
          lightbar.length > 2.0 && lightbar.length < 150) {
        // 放宽角度限制：允许60°到120°的垂直灯条
        if ((lightbar.angle <= 180 && lightbar.angle >= 120) ||
            (lightbar.angle <= 60 && lightbar.angle >= 0)) {
          tools::draw_points(detection, lightbar.points, {0, 255, 255}, 3);
        }
      } else {
        continue;
      }
    }

    // 绘制装甲板
    for (const auto& armor : armors) {
      // auto info = fmt::format("{:.2f} {:.2f} {:.1f} {:.2f} {} {}", armor.ratio, armor.side_ratio,
      //                         armor.rectangular_error * 57.3, armor.confidence,
      //                         ARMOR_NAME[armor.name], ARMOR_TYPE[armor.type]);

      // 修改这里：从装甲板的左灯条获取颜色
      auto info = fmt::format("{} {:.2f} {} {}", COLOR[armor.left.color], armor.confidence,
                              ARMOR_NAME[armor.name], ARMOR_TYPE[armor.type]);

      tools::draw_points(detection, armor.points, {0, 255, 0});
      tools::draw_text(detection, info, armor.left.bottom, {0, 255, 0});
    }

    cv::Mat binary_img2;
    cv::resize(binary_img, binary_img2, {}, 0.5, 0.5); // 显示时缩小图片尺寸
    cv::resize(detection, detection, {}, 0.5, 0.5);    // 显示时缩小图片尺寸

    cv::imshow("threshold", binary_img2);
    cv::imshow("detection", detection);
  }

} // namespace xz_vision
