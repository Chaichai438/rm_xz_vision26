#include "rune_detector.hpp"

namespace auto_buff
{
  std::mutex MUTEX;

  /**
   @brief 初始化参数
    */
  BuffDetection::BuffDetection(const std::string& config)
  // 全局 roi 初始设置为图像的全部范围，注意图像大小要与下面两个参数对应
  {
    tools::logger()->debug("-----BuffDetection环境配置中-----");
    auto yaml = YAML::LoadFile(config);

    std::string colorStr = yaml["color"].as<std::string>();

    std::transform(colorStr.begin(), colorStr.end(), colorStr.begin(), ::tolower);

    if (colorStr == "red") {
      color = Color::RED;
    } else if (colorStr == "blue") {
      color = Color::BLUE;
    } else {
      tools::logger()->error("unknown color:{} ", colorStr);
    }

    DRAW_COLOR = color == Color::BLUE ? RED : BLUE;

    // 调试文件
    whether_use_debug_pre = yaml["whether_use_debug_pre"].as<bool>();
    whether_use_debug_arrow = yaml["whether_use_debug_arrow"].as<bool>();
    whether_use_debug_local_roi = yaml["whether_use_debug_local_roi"].as<bool>();
    whether_use_debug_armor = yaml["whether_use_debug_armor"].as<bool>();
    whether_use_debug_centerR = yaml["whether_use_debug_centerR"].as<bool>();

    image_width = yaml["image_width"].as<float>();
    image_height = yaml["image_height"].as<float>();

    m_local_mask = cv::Mat::zeros(image_height, image_width, CV_8U);
    m_global_roi = cv::Rect2f(0, 0, image_width, image_height);

    param_thresh = yaml["param_thresh"].as<double>();
    param_maxval = yaml["param_maxval"].as<double>();
    param_kernel_width = yaml["param_kernel_width"].as<int>();
    param_kernel_height = yaml["param_kernel_height"].as<int>();

    min_arrow_lightline_area = yaml["min_arrow_lightline_area"].as<double>();
    max_arrow_lightline_area = yaml["max_arrow_lightline_area"].as<double>();

    max_arrow_lightline_aspect_ratio = yaml["max_arrow_lightline_aspect_ratio"].as<double>();

    min_arrow_lightline_num = yaml["min_arrow_lightline_num"].as<int>();
    max_arrow_lightline_num = yaml["max_arrow_lightline_num"].as<int>();

    min_arrow_aspect_ratio = yaml["min_arrow_aspect_ratio"].as<double>();
    max_arrow_aspect_ratio = yaml["max_arrow_aspect_ratio"].as<double>();

    max_arrow_area = yaml["max_arrow_area"].as<double>();
    max_same_arrow_area_ratio = yaml["max_same_arrow_area_ratio"].as<double>();
    // roi 矩形的距离和宽度
    local_roi_distance_ratio = yaml["local_roi_distance_ratio"].as<double>();
    local_roi_width = yaml["local_roi_width"].as<float>();

    min_armor_lightline_area = yaml["min_armor_lightline_area"].as<double>();
    max_armor_lightline_area = yaml["max_armor_lightline_area"].as<double>();

    min_armor_lightline_aspect_ratio = yaml["min_armor_lightline_aspect_ratio"].as<double>();
    max_armor_lightline_aspect_ratio = yaml["max_armor_lightline_aspect_ratio"].as<double>();

    min_armor_lightline_contour_area = yaml["min_armor_lightline_contour_area"].as<double>();
    max_armor_lightline_contour_area = yaml["max_armor_lightline_contour_area"].as<double>();

    max_same_armor_area_ratio = yaml["max_same_armor_area_ratio"].as<double>();

    min_same_armor_distance = yaml["min_same_armor_distance"].as<double>();
    max_same_armor_distance = yaml["max_same_armor_distance"].as<double>();

    min_center_area = yaml["min_center_area"].as<double>();
    max_center_area = yaml["max_center_area"].as<double>();

    max_center_aspect_ratio = yaml["max_center_aspect_ratio"].as<double>();

    armor_center_vertical_distance_threshold =
        yaml["armor_center_vertical_distance_threshold"].as<float>();

    global_roi_length_ratio = yaml["global_roi_length_ratio"].as<double>();

    armor_outside_width = yaml["armor_outside_width"].as<double>();
    armor_inside_width = yaml["armor_inside_width"].as<double>();

    armor_inside_y = yaml["armor_inside_y"].as<double>();
    armor_outside_y = yaml["armor_outside_y"].as<double>();
    armor_outside_height = yaml["armor_outside_height"].as<double>();

    m_startTime = std::chrono::steady_clock::now();
    m_lightArmorNum = 0;
  };

  void Frame::set(const cv::Mat& image, const std::chrono::steady_clock::time_point& time)
  {
    m_image = image;
    m_time = time;
  }

  // 提供一种方式在对象创建后重新设置其内容，避免频繁构造新对象（可能用于性能优化或复用对象）
  void Frame::set(const cv::Mat& image, const std::chrono::steady_clock::time_point& time,
                  double pitch, double yaw, double roll)
  {
    m_image = image;
    m_time = time;
    m_roll = roll;
    m_pitch = pitch;
    m_yaw = yaw;
  }

  void BuffDetection::reset_on_failure(const cv::Mat& image)
  {
    m_global_roi = cv::Rect2f(0, 0, static_cast<float>(image.cols), static_cast<float>(image.rows));
    m_lightArmorNum = 0;
  }

  /**
   * @brief 检测箭头，装甲板和中心。如果所有检测均成功，则返回 true，否则返回 false。
   * @param[in] Frame        从相机传来的一帧图像，包括图像本身和其时间戳
   * @return true
   * @return false
   */
  bool BuffDetection::detect(const Frame& frame)
  {
    preprocess_imgs(frame);

    if (!check_arrow()) {
      m_status = Status::ARROW_FAILURE;
      reset_on_failure(frame.m_image);
      return false;
    }

    set_local_roi();

    // 尝试最多两次
    for (int attempt = 0; attempt < 2; ++attempt) {
      if (detect_armor() && detect_centerR()) {
        set_armor();
        set_global_roi();
        m_status = Status::SUCCESS;
        return true;
      }

      if (attempt == 0) {
        std::swap(m_center_roi, m_armor_roi);
      }
    }

    // 两次都失败
    m_status = Status::CENTER_FAILURE;
    reset_on_failure(frame.m_image);
    return false;
  }

  // bool BuffDetection::detect(const Frame& frame)
  // {
  //   // 表示尚未进行过 ROI 交换
  //   bool reverse = false;
  //   preprocess_imgs(frame);
  //   // 预处理
  //   if (check_arrow() == false) {
  //     m_status = Status::ARROW_FAILURE;
  //     goto FAIL;
  //   }
  //   set_local_roi();

  // RESTART:
  //   if (detect_armor() == false) {
  //     m_status = Status::ARMOR_FAILURE;
  //     goto FAIL;
  //   }
  //   if (detect_centerR() == false) {
  //     m_status = Status::CENTER_FAILURE;
  //     if (reverse == false) {
  //       std::swap(m_center_roi, m_armor_roi);
  //       reverse = true;
  //       goto RESTART;
  //     } else {
  //       goto FAIL;
  //     }
  //   }
  //   set_armor();
  //   set_global_roi();
  //   m_status = Status::SUCCESS;
  //   return true;

  // FAIL:
  //   // 如果检测失败，则将全局 roi 设为和原图片一样大小
  //   m_global_roi = {0, 0, static_cast<float>(frame.m_image.cols),
  //                   static_cast<float>(frame.m_image.rows)};
  //   m_lightArmorNum = 0;
  //   return false;
  // };

  /**
    @brief 图像预处理
     */
  void BuffDetection::preprocess_imgs(const Frame& frame)
  {
    // tools::logger()->debug("-----preprocess_imgs-----");
    if (frame.m_image.empty()) {
      tools::logger()->error("[BuffDetection::preprocess_imgs] Error: Received an empty image!");
      return;
    }

    m_image_show = frame.m_image.clone();

    cv::Mat gray_img; // 转灰度图
    cv::cvtColor(frame.m_image, gray_img, cv::COLOR_BGR2GRAY);

    cv::Mat binary_img; // 二值化
    cv::threshold(gray_img, binary_img, param_thresh, param_maxval, cv::THRESH_BINARY);

    cv::Mat dilated_img; // 膨胀
    cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(param_kernel_width, param_kernel_height)); // 使用矩形核
    cv::dilate(binary_img, dilated_img, kernel, cv::Point(-1, -1), 1);

    m_image_arrow = dilated_img;
    m_image_armor = dilated_img;

    // 设置局部 roi
    m_local_mask.setTo(0);

    if (whether_use_debug_pre) {
      cv::imshow("binary_img", m_image_show);
      cv::imshow("m_image_arrow", m_image_arrow);
      // cv::imshow("m_image_armor", m_image_armor); // 调试用}
    }
  };

  //========================Arrow========================

  /**
   * @brief 设置箭头
   * @param[in] points        点集
   * @param[in] roi
   */
  void Arrow::set(const std::vector<LightLine>& lightlines, const cv::Point2f& roi)
  {
    std::vector<cv::Point2f> arrowPoints;
    double fillArea = 0.0;
    double pointLineThresh = 0.0;
    std::for_each(lightlines.begin(), lightlines.end(), [&](const LightLine& l) {
      arrowPoints.insert(arrowPoints.end(), l.m_contour.begin(), l.m_contour.end());
      fillArea += l.m_contour_area;
      pointLineThresh += l.m_length / lightlines.size();
    });

    // 滤除距离较大的点
    m_contour.clear();
    cv::Vec4f line;
    cv::fitLine(arrowPoints, line, cv::DIST_L2, 0, 0.01, 0.01);
    for (const auto& point : arrowPoints) {
      if (tools::pointLineDistance(point, line) < pointLineThresh) {
        m_contour.push_back(point);
      }
    }
    // 设置成员变量
    m_rotated_rect = cv::minAreaRect(m_contour);
    m_center = m_rotated_rect.center + roi;
    m_length = m_rotated_rect.size.height;
    m_width = m_rotated_rect.size.width;
    // RotatedRect::angle 范围为 -90~0. 这里根据长宽长度关系，将角度扩展到 -90~90
    if (m_length < m_width) {
      m_angle = m_rotated_rect.angle;
      // 长的为 length
      std::swap(m_length, m_width);
    } else {
      m_angle = m_rotated_rect.angle + 90;
    }
    m_aspect_ratio = m_length / m_width;
    m_area = m_length * m_width;
    m_fill_ratio = fillArea / m_area;
    return;
  }

  /**
   * @brief 比较两个灯条是否满足在一个箭头内的条件，是则返回 true，否则为 false
   * @param[in] l1
   * @param[in] l2
   * @return true
   * @return false
   */
  bool BuffDetection::is_same_arrow(const LightLine& l1, const LightLine& l2)
  {
    // 判断面积比
    double areaRatio{l1.m_area / l2.m_area};
    if (tools::inRange(areaRatio, 1.0 / max_same_arrow_area_ratio, max_same_arrow_area_ratio) ==
        false) {
      return false;
    }
    // 判断距离
    double distance{tools::p2p_distance(l1.m_rotated_rect.center, l2.m_rotated_rect.center)};
    double maxDistance{1.2 * (l1.m_width + l2.m_width)};
    if (distance > maxDistance) {
      return false;
    }
    return true;
  };

  /**
   * @brief 寻找箭头，找到则返回 true，否则返回 false
   * @return true
   * @return false
   */
  bool BuffDetection::check_arrow()
  {
    std::vector<LightLine> lightlines;
    find_arrow_lightlines(m_image_arrow, lightlines, m_global_roi);

    // 灯条匹配箭头
    if (match_arrow(m_arrow, lightlines, m_global_roi) == false) {
      return false;
    }

    if (whether_use_debug_arrow) {
      for (const auto& lightline : lightlines) {
        draw(lightline, GREEN);
      }
      draw(m_arrow.m_rotated_rect, WHITE, 2);
      cv::imshow("arrow", m_image_show);
    }

    return true;
  };

  /**
   * @brief 根据提取的灯条匹配箭头，成功返回 true，否则返回 false
   * @param[in] lightlines    灯条向量
   * @param[in] arrowPtr      指向箭头的指针
   * @param[in] roi           roi，用来设置箭头的正确位置
   * @return true
   * @return false
   */
  bool BuffDetection::match_arrow(Arrow& arrow, const std::vector<LightLine>& lightlines,
                                  const cv::Rect2f& roi)
  {
    // 利用 cv::partition 匹配箭头
    std::vector<int> labels;
    cv::partition(lightlines, labels, is_same_arrow);
    // data 记录了标识号和其对应次数
    std::vector<std::pair<int, int>> data;
    for (auto label : labels) {
      // 对每个 label，从已记录的数据中寻找是否有这个条目，有则对应计数项 +1，否则新增一个条目
      auto iter = std::find_if(data.begin(), data.end(), [label](const std::pair<int, int>& unit) {
        return unit.first == label;
      });
      if (iter == data.end()) {
        data.emplace_back(label, 1);
      } else {
        iter->second += 1;
      }
    }
    if (data.empty() == true) {
      return false;
    }
    // 寻找出现次数最多的 label 和其对应的 num
    auto [maxLabel, maxNum]{*std::max_element(
        data.begin(), data.end(), [](const std::pair<int, int>& i, const std::pair<int, int>& j) {
          return i.second < j.second;
        })};
    // 判断 num 是否符合要求
    if (tools::inRange(maxNum, min_arrow_lightline_num, max_arrow_lightline_num) == false) {
      return false;
    }
    // 再次遍历 labels，选取和 maxLabel 相同的 label，并存入一个向量
    std::vector<int> arrowIndices;
    for (unsigned int i = 0; i < labels.size(); ++i) {
      if (labels[i] == maxLabel) {
        arrowIndices.push_back(i);
      }
    }
    // 根据这个向量，将其对应的灯条轮廓点集中每个点存入箭头点的向量中
    std::vector<LightLine> arrowLightlines;
    for (auto index : arrowIndices) {
      arrowLightlines.push_back(lightlines.at(index));
    }
    // 设置这个箭头
    arrow.set(arrowLightlines, roi.tl());
    // 判断长宽比
    if (tools::inRange(arrow.m_aspect_ratio, min_arrow_aspect_ratio, max_arrow_aspect_ratio) ==
        false) {
      return false;
    }
    // 判断面积

    std::cout << "arrow_area: " << arrow.m_area << std::endl;
    if (arrow.m_area > max_arrow_area) {
      return false;
    }
    return true;
  };

  /**
   * @brief 寻找符合箭头要求的灯条轮廓并存入灯条容器
   * @param[in] binary        二值图
   * @param[in] lightlines    输出的灯条向量
   * @param[in] roi           roi，用来设置灯条的正确位置
   */
  void BuffDetection::find_arrow_lightlines(const cv::Mat& binary,
                                            std::vector<LightLine>& lightlines,
                                            const cv::Rect2f& roi)
  {
    // 寻找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    for (const auto& contour : contours) {
      LightLine lightline(contour, roi);

      // 判断面积
      if (tools::inRange(lightline.m_area, min_arrow_lightline_area, max_arrow_lightline_area) ==
          false) {
        continue;
      }

      std::cout << "lightline.m_area: " << lightline.m_area << std::endl;

      // 判断长宽比
      if (lightline.m_aspect_ratio > max_arrow_lightline_aspect_ratio) {
        continue;
      }
      // 符合要求，则存入
      lightlines.emplace_back(std::move(lightline));
    }
  };

  //========================Armor========================

  /**
   * @brief 设置装甲板参数
   * @param[in] l1
   * @param[in] l2
   */
  void Armor::set(const LightLine& l1, const LightLine& l2)
  {
    if (l1.m_contour_area > l2.m_contour_area) {
      m_inside = l1;
      m_outside = l2;
    } else {
      m_outside = l1;
      m_inside = l2;
    }
    m_center = (m_inside.m_center + m_outside.m_center) * 0.5;
    m_x = m_center.x, m_y = m_center.y;
    m_tlIn = m_inside.m_tl;
    m_trIn = m_inside.m_tr;
    m_blIn = m_inside.m_bl;
    m_brIn = m_inside.m_br;
    m_tlOut = m_outside.m_tl;
    m_trOut = m_outside.m_tr;
    m_blOut = m_outside.m_bl;
    m_brOut = m_outside.m_br;
    return;
  }

  /**
   * @brief 寻找符合装甲板要求的边框灯条，并将其存入一个向量中。成功返回 true，否则返回
   * false。
   * @param[in] image         二值图
   * @param[in] lightlines    边框灯条
   * @param[in] globalRoi     全局roi，用来设置灯条的正确位置
   * @param[in] localRoi      局部roi，用来设置灯条的正确位置
   * @return true
   * @return false
   */
  bool BuffDetection::find_armor_lightlines(const cv::Mat& image,
                                            std::vector<LightLine>& lightlines,
                                            const cv::Rect2f& globalRoi, const cv::Rect2f& localRoi)
  {
    // 由于可能存在匹配失败后重新匹配的情况，因此 lightlines
    // 向量可能保存了匹配失败时的数据。因此需要将其清空
    lightlines.clear();
    // 寻找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(image, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    for (const auto& contour : contours) {
      LightLine lightline(contour, globalRoi, localRoi);
      // 判断面积
      if (tools::inRange(lightline.m_area, min_armor_lightline_aspect_ratio,
                         max_armor_lightline_aspect_ratio) == false) {
        continue;
      }
      // 由于会有将箭头灯条囊括,因此最大面积放得非常宽，因此需要再判断轮廓面积
      if (tools::inRange(lightline.m_contour_area, min_armor_lightline_contour_area,
                         max_armor_lightline_contour_area) == false) {
        continue;
      }
      // 判断长宽比
      if (tools::inRange(lightline.m_aspect_ratio, min_armor_lightline_aspect_ratio,
                         max_armor_lightline_aspect_ratio) == false) {
        continue;
      }
      // 全部符合条件，则存入该灯条
      lightlines.emplace_back(std::move(lightline));
    }
    // 如果符合要求的装甲板数目不是 2，则检测失败
    if (lightlines.empty()) {
      return false;
    }
    return true;
  }

  /**
   * @brief 根据提取的灯条匹配装甲板，成功返回 true，否则返回 false
   * @param[in] frames        边框灯条
   * @param[in] centers       中心灯条
   * @param[in] armorPtr      装甲板
   * @param[in] arrowPtr      箭头
   * @return true
   * @return false
   */
  bool BuffDetection::find_armor(Armor& armor, const std::vector<LightLine>& frames,
                                 const Arrow& arrow)
  {
    std::vector<int> labels;
    // 使用 sameArmor 函数匹配
    cv::partition(frames, labels, is_same_armor);
    for (size_t i = 0; i < labels.size() - 1; ++i) {
      for (size_t j = i + 1; j < labels.size(); ++j) {
        if (labels[i] == labels[j]) {
          cv::Point2f center = 0.5 * (frames.at(i).m_center + frames.at(j).m_center);
          // 此处判断装甲板中心与箭头中心的距离，如果不符则检测失败
          if (tools::inRange(tools::p2p_distance(center, arrow.m_center), arrow.m_length * 0.8,
                             arrow.m_length * 1.5)) {
            armor.set(frames[i], frames[j]);
            return true;
          }
        }
      }
    }
    return false;
  };

  /**
   * @brief 装甲板四个角点需要根据其与中心的位置关系重新设置，以便于后面的 PnP
   * 解算。之前的四个角点设置与 Lightline 构造函数的角点一致。
   */
  void BuffDetection::set_armor()
  {
    // 装甲板里外边框的重排序，之前根据面积确定边框，但可能不准确，现在根据到中心的距离确定
    if (tools::p2p_distance(m_armor.m_inside.m_center, m_centerR.m_center_R) >
        tools::p2p_distance(m_armor.m_outside.m_center, m_centerR.m_center_R)) {
      std::swap(m_armor.m_inside, m_armor.m_outside);
      std::swap(m_armor.m_tlIn, m_armor.m_tlOut);
      std::swap(m_armor.m_trIn, m_armor.m_trOut);
      std::swap(m_armor.m_blIn, m_armor.m_blOut);
      std::swap(m_armor.m_brIn, m_armor.m_brOut);
    }
    /**
     * 当中心 R
     * 的纵坐标小于装甲板的纵坐标时，左上角和右下角、左下角和右上角的角点是相反的，需要交换这两对点。
     * 但由于图像上存在误差，因此实际上设置中心 R
     * 的纵坐标明显小于装甲板的纵坐标时才交换，其它情况下使用另一种判断方式，确保正确判断。
     */
    if (m_centerR.m_center_R.y < m_armor.m_center.y - armor_center_vertical_distance_threshold) {
      std::swap(m_armor.m_tlIn, m_armor.m_brIn);
      std::swap(m_armor.m_tlOut, m_armor.m_brOut);
      std::swap(m_armor.m_trIn, m_armor.m_blIn);
      std::swap(m_armor.m_trOut, m_armor.m_blOut);
    }
    /**
     * 另一种判断方式根据装甲板的横坐标和中心的横坐标，以及装甲板左上角和左下角横坐标的位置关系得到。
     * 在中心 R 的纵坐标明显大于装甲板中心时不需要交换，因此在纵坐标关系不明显时才进行判断。
     * 在上面的约束下，如果装甲板横坐标大于中心，则装甲板明显在中心右侧，此时根据之前的角点设置，左上角点横坐标大于左下角点横坐标时，判断装甲板在中心上面，否则为下面。
     * 装甲板横坐标小于中心时同理。
     * 注：前一种判断方式在纵坐标差距不明显时会出现误差，而后一种在横坐标差距不明显时会出现误差。
     * 因此将两者结合起来使用，即在纵坐标差距明显时使用前一种，纵坐标差距不明显时使用后一种，可以实现全场景下的覆盖。
     */
    else if (m_centerR.m_center_R.y <
             m_armor.m_center.y + armor_center_vertical_distance_threshold) {
      if (((m_centerR.m_center_R.x < m_armor.m_center.x &&
            m_armor.m_inside.m_tl.x > m_armor.m_inside.m_bl.x) ||
           (m_centerR.m_center_R.x > m_armor.m_center.x &&
            m_armor.m_inside.m_tl.x < m_armor.m_inside.m_bl.x)) == false) {
        std::swap(m_armor.m_tlIn, m_armor.m_brIn);
        std::swap(m_armor.m_trIn, m_armor.m_blIn);
      }
      if (((m_centerR.m_center_R.x < m_armor.m_center.x &&
            m_armor.m_outside.m_tl.x > m_armor.m_outside.m_bl.x) ||
           (m_centerR.m_center_R.x > m_armor.m_center.x &&
            m_armor.m_outside.m_tl.x < m_armor.m_outside.m_bl.x)) == false) {
        std::swap(m_armor.m_tlOut, m_armor.m_brOut);
        std::swap(m_armor.m_trOut, m_armor.m_blOut);
      }
    }
#if CONSOLE_OUTPUT >= 2
    MUTEX.lock();
    std::cout << "feature camera points: ";
    auto cameraPoints{getCameraPoints()};
    std::for_each(cameraPoints.begin(), cameraPoints.end(),
                  [](auto&& it) { std::cout << it << " "; });
    std::cout << std::endl;
    MUTEX.unlock();
#endif
  }

  /**
   * @brief 寻找装甲板，找到则返回 true，否则为 false
   * @return true
   * @return false
   */
  bool BuffDetection::detect_armor()
  {
    // 1. 结构体改为值传递，避免引用绑定的麻烦
    struct TrialRegion {
      cv::Mat image;
      cv::Rect roi;

      // 使用 cv::Rect 的基类或具体类型，确保兼容性
      TrialRegion(cv::Mat img, cv::Rect r)
          : image(img)
          , roi(r)
      {
      }
    };

    std::vector<TrialRegion> trials;

    // 使用 emplace_back，它会自动处理 cv::Rect2f 到 cv::Rect 的转换
    trials.emplace_back((m_image_armor & m_local_mask)(m_armor_roi), m_armor_roi);
    trials.emplace_back((m_image_armor & m_local_mask)(m_center_roi), m_center_roi);

    std::vector<LightLine> lightlines;
    bool found = false;

    for (auto& trial : trials) {
      lightlines.clear();
      // 注意：find_armor_lightlines 的参数类型需与 trial.roi 一致
      if (find_armor_lightlines(trial.image, lightlines, m_global_roi, trial.roi)) {
        if (find_armor(m_armor, lightlines, m_arrow)) {
          // 如果成功，更新类成员变量（如果需要保持逻辑一致性）
          m_armor_roi = trial.roi;
          found = true;
          break;
        }
      }
    }

    if (!found)
      return false;

    // ... Debug 绘制逻辑 ...
    if (whether_use_debug_armor) {
      for (const auto& lightline : lightlines) {
        draw(lightline, DRAW_COLOR, 1, m_armor_roi);
      }
      cv::circle(m_image_show, m_armor.m_center, m_armor.m_outside.m_length * 0.45,
                 cv::Scalar(225, 225, 225), 2);
    }

    return true;
  }
  //   bool BuffDetection::detect_armor()
  //   {
  //     // armor roi 区域的图像为检测图像，center roi 区域为备用图像
  //     cv::Mat detect = (m_image_armor & m_local_mask)(m_armor_roi);
  //     cv::Mat backup = (m_image_armor & m_local_mask)(m_center_roi);
  //     std::vector<LightLine> lightlines;
  //     // 调换标志位，如果检测不到，则调换检测图像和备用图像，并将其置为 true
  //     bool reverse = false;
  //   RESTART:
  //     // 寻找符合装甲板边框要求的灯条
  //     if (find_armor_lightlines(detect, lightlines, m_global_roi, m_armor_roi) == false) {
  //       // 如果找不到并且已经调换过图像了，则检测失败
  //       if (reverse == true) {
  //         return false;
  //       }
  //       // 如果找不到并且没有调换过，则调换图像并置标志位
  //       std::swap(detect, backup);
  //       std::swap(m_armor_roi, m_center_roi);
  //       reverse = true;
  //       // 回到检测装甲板灯条处
  //       goto RESTART;
  //     }

  // #endif
  //     // 根据灯条匹配装甲板
  //     if (find_armor(m_armor, lightlines, m_arrow) == false) {
  //       if (reverse == true) {
  //         return false;
  //       }
  //       std::swap(detect, backup);
  //       std::swap(m_armor_roi, m_center_roi);
  //       reverse = true;
  //       goto RESTART;
  //     }
  //     if (whether_use_debug_armor)
  //       for (const auto& lightline : lightlines) {
  //         draw(lightline, DRAW_COLOR, 1, m_armor_roi);
  //       }
  //     cv::circle(m_image_show, m_armor.m_center, m_armor.m_outside.m_length * 0.45,
  //                cv::Scalar(225, 225, 225), 2);
  // #endif
  //     return true;
  //   };

  /**
   * @brief 判断两个灯条是否满足在一个装甲板内的条件，是则返回 true，否则为 false
   * @param[in] l1
   * @param[in] l2
   * @return true
   * @return false
   */
  bool BuffDetection::is_same_armor(const LightLine& l1, const LightLine& l2)
  {
    // 判断面积比
    double ratio{l1.m_contour_area / l2.m_contour_area};
    if (tools::inRange(ratio, 1 / max_same_armor_area_ratio, max_same_armor_area_ratio) == false) {
      return false;
    }
    // 判断距离
    double distance{tools::p2p_distance(l1.m_center, l2.m_center)};
    if (distance < min_same_armor_distance || distance > max_same_armor_distance) {
      return false;
    }
    // 判断角度
    double angle = calAngleBetweenLightlines(l1, l2);
    if (angle > 10 && angle < 170) {
      return false;
    }
    return true;
  }

  //========================CenterR========================

  /**
   * @brief 设置中心 R
   * @param[in] lightline
   */
  void CenterR::set(const LightLine& lightline)
  {
    m_lightline = lightline;
    m_bounding_rect = cv::boundingRect(lightline.m_contour);
    // 由于灯条角点和中心点已经设置过 roi，因此这里不需要重新设置
    m_center_R = lightline.m_center;
    m_x = m_center_R.x, m_y = m_center_R.y;
    return;
  }

  /**
   * @brief 根据中心灯条寻找并设置中心，成功返回 true，失败返回 false
   * @param[in] center        中心
   * @param[in] lightlines    中心灯条向量
   * @param[in] arrow         箭头
   * @param[in] armor         装甲板
   * @return true
   * @return false
   */
  bool BuffDetection::find_centerR(CenterR& center, const std::vector<LightLine>& lightlines,
                                   const Arrow& arrow, const Armor& armor)
  {
    // 设置中心 R 到外侧装甲板灯条的距离范围
    // 不用到内侧装甲板灯条的位置的原因是内侧装甲板灯条识别的时候可能和箭头灯条连在一起导致距离出现误差
    const double distance2OutsideArmor{(armor.m_outside.m_length + armor.m_inside.m_length) *
                                       power_rune_radius * 1.13 /
                                       (armor_outside_width + armor_inside_width)};
    const double ratio = 0.85;
    const double maxDistance2OutsideArmor{distance2OutsideArmor / ratio};
    const double minDistance2OutsideArmor{distance2OutsideArmor * ratio};
    // 设置中心 R 到箭头所在直线的最大距离
    const double maxDistance2ArrowLine{0.3 * armor.m_inside.m_width};
    std::vector<LightLine> filteredLightlines;
    for (auto iter = lightlines.begin(); iter != lightlines.end(); ++iter) {
      /**
       * 之前设置装甲板内外灯条的时候是根据面积判断的，外部灯条面积小于内部灯条。但有可能出现误差导致内部灯条面积更小。
       * 所以在这里比较可能的中心灯条到装甲板内部灯条和外部灯条的距离，距离大的就是外部灯条。
       * 内外灯条的设置在 Detector::setArmor() 中设置
       */
      cv::Point2f armorOutsideCenter =
          tools::p2p_distance(iter->m_center, armor.m_inside.m_center) >
                  tools::p2p_distance(iter->m_center, armor.m_outside.m_center)
              ? armor.m_inside.m_center
              : armor.m_outside.m_center;
      double p2p{tools::p2p_distance(iter->m_center, armorOutsideCenter)};
      double p2l{tools::pointLineDistance(iter->m_center, armor.m_center, arrow.m_center)};
      // 判断到装甲板外部灯条的距离
      if (tools::inRange(p2p, minDistance2OutsideArmor, maxDistance2OutsideArmor) == false) {
        continue;
      }
      // 判断到箭头所在直线的距离
      if (p2l > maxDistance2ArrowLine) {
        continue;
      }
      filteredLightlines.push_back(*iter);
    }
    if (filteredLightlines.empty()) {
      return false;
    }
    // 取所有符合要求的灯条中面积最大的为中心 R 灯条并设置中心 R
    LightLine target{*std::max_element(
        filteredLightlines.begin(), filteredLightlines.end(),
        [](const LightLine& l1, const LightLine& l2) { return l1.m_area < l2.m_area; })};
    center.set(target);
    return true;
  }

  /**
   * @brief 寻找中心 R ，找到则返回 true，否则返回 false
   * @return true
   * @return false
   */
  bool BuffDetection::detect_centerR()
  {
    m_image_center = (m_image_armor & m_local_mask)(m_center_roi);

    cv::imshow("centerR", m_image_center);

    // 寻找中心灯条，可能是多个
    std::vector<LightLine> lightlines;
    if (findCenterLightlines(m_image_center, lightlines, m_global_roi, m_center_roi) == false) {
      return false;
    }

    // 从灯条中寻找中心 R
    if (find_centerR(m_centerR, lightlines, m_arrow, m_armor) == false) {
      return false;
    }

    if (whether_use_debug_centerR) {
      for (const auto& lightline : lightlines) {
        draw(lightline, YELLOW, 1, m_center_roi); // 黄色框
      }

      draw(m_centerR.m_bounding_rect, cv::Scalar(225, 225, 225), 2, m_center_roi);

      cv::imshow("arrow", m_image_show);
    }

    return true;
  }

  /**
   * @brief 寻找符合中心 R 要求的中心灯条，并将其存入一个向量中。成功返回 true，否则返回
   * false。
   * @param[in] image         带有中心区域的图片
   * @param[in] lightlines    存储的灯条向量
   * @param[in] globalRoi     全局 roi
   * @param[in] localRoi      局部 roi
   * @return true
   * @return false
   */
  bool BuffDetection::findCenterLightlines(const cv::Mat& image, std::vector<LightLine>& lightlines,
                                           const cv::Rect2f& globalRoi, const cv::Rect2f& localRoi)
  {
    // 寻找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(image, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    for (const auto& contour : contours) {
      LightLine lightline(contour, globalRoi, localRoi);
      // 判断面积
      if (tools::inRange(lightline.m_area, min_center_area, max_center_area) == false) {
        continue;
      }
      // 判断长宽比
      if (lightline.m_aspect_ratio > max_center_aspect_ratio) {
        continue;
      }
      // 如果全部符合，则存入向量中
      lightlines.emplace_back(std::move(lightline));
    }
    // 符合要求灯条的数量为 0 则失败
    if (lightlines.empty()) {
      return false;
    }
    return true;
  }

  /**
   * @brief 计算两个灯条长边的夹角
   * @param[in] l1
   * @param[in] l2
   * @return double
   */
  double BuffDetection::calAngleBetweenLightlines(const LightLine& l1, const LightLine& l2)
  {
    // 长边对应方向向量
    std::array<std::array<cv::Point2f, 4>, 2> pointsArray;
    std::array<double, 2> lengths{l1.m_length, l2.m_length};
    l1.m_rotated_rect.points(pointsArray.at(0).begin());
    l2.m_rotated_rect.points(pointsArray.at(1).begin());
    std::array<cv::Point2f, 2> vecs;
    constexpr double eps = 1e-3;
    // 遍历灯条的四个点，寻找长边对应的两个点
    for (size_t i = 0; i < 2; ++i) {
      for (size_t j = 0; j < 4; ++i) {
        bool flag = false;
        for (size_t k = j; k < 4; ++j) {
          if (std::abs(tools::p2p_distance(pointsArray.at(i).at(j), pointsArray.at(i).at(k)) -
                       lengths.at(i)) < eps) {
            flag = true;
            vecs.at(i) = pointsArray.at(i).at(j) - pointsArray.at(i).at(k);
            break;
          }
        }
        if (flag == true) {
          break;
        }
      }
    }
    // 算向量之间夹角，取绝对值
    double dotProduct = vecs.at(0).x * vecs.at(1).x + vecs.at(0).y * vecs.at(1).y;
    double magnitude1 = std::sqrt(vecs.at(0).x * vecs.at(0).x + vecs.at(0).y * vecs.at(0).y);
    double magnitude2 = std::sqrt(vecs.at(1).x * vecs.at(1).x + vecs.at(1).y * vecs.at(1).y);
    double angle = tools::radian2Angle(std::acos(dotProduct / (magnitude1 * magnitude2)));
    return angle;
  }

  /**
   * @brief 设置全局
   * roi，选定符所在的区域，然后在下一个循环预处理时进行裁剪，可以加快运算时间，提高帧率
   */
  void BuffDetection::set_global_roi()
  {
    double width{global_roi_length_ratio * 2 *
                 tools::p2p_distance(m_armor.m_center, m_centerR.m_center_R)};
    m_global_roi =
        cv::Rect2f(m_centerR.m_x - 0.5 * width, m_centerR.m_y - 0.5 * width, width, width);
    tools::reset_roi(m_global_roi, image_height, image_width);

    cv::rectangle(m_image_show, m_global_roi, DRAW_COLOR);
    cv::imshow("global_roi", m_image_show);
  }

  /**
   * @brief 设置局部 roi，局部 roi 包括中心 R 的 roi 和装甲板的 roi，根据箭头的两个端点进行提取。
   */
  void BuffDetection::set_local_roi()
  {
    // 设置两个 roi 矩形的距离和宽度
    double distance{m_arrow.m_length * local_roi_distance_ratio};
    float width{local_roi_width};
    // 确定两个 roi 的中心点
    float x = distance * std::cos(tools::angle2Radian(m_arrow.m_angle));
    float y = distance * std::sin(tools::angle2Radian(m_arrow.m_angle));
    cv::Point2f centerUp{m_arrow.m_center.x - m_global_roi.x + x,
                         m_arrow.m_center.y - m_global_roi.y + y};
    cv::Point2f centerDown{m_arrow.m_center.x - x - m_global_roi.x,
                           m_arrow.m_center.y - m_global_roi.y - y};
    /**
     * 用类似旋转矩形先与原图做掩码，可以减少箭头灯条在装甲板区域的个数，避免箭头灯条与装甲板连在一起从而误识别的情况
     * 家里的符箭头灯条和装甲板灯条亮度差距过大，导致装甲板区域如果存在箭头灯条的话，二值化后的图像所有箭头灯条会和装甲板内部灯条连在一起，导致特征识别失败
     */
    cv::RotatedRect rectUp{centerUp, cv::Size(width, width), (float)m_arrow.m_angle};
    cv::RotatedRect rectDown{centerDown, cv::Size(width, width), (float)m_arrow.m_angle};

    std::array<std::array<cv::Point2f, 4>, 2> roiPoints;
    rectUp.points(roiPoints.at(0).begin());
    rectDown.points(roiPoints.at(1).begin());
    // 调整角点坐标不要越界，否则程序会直接中断退出
    for (auto& points : roiPoints) {
      for (auto& point : points) {
        if (point.x < 0) {
          point.x = 0;
        }
        if (point.x > m_global_roi.width) {
          point.x = m_global_roi.width;
        }
        if (point.y < 0) {
          point.y = 0;
        }
        if (point.y > m_global_roi.height) {
          point.y = m_global_roi.height;
        }
      }
    }
    /**
     * 注意 local mask 是和检测图像大小一致，检测部分为255，其余部分为0
     * 的掩码，通过与检测图像进行与操作滤除 armor 和 center roi
     * 是矩形，通过检测图像的裁剪得到目标图像
     */
    for (const auto& points : roiPoints) {
      // cv::fillConvexPoly 只支持 cv::Point 数组，因此在这里需要转换一下
      std::vector<cv::Point> _points;
      for (const auto& point : points) {
        _points.push_back(static_cast<cv::Point>(point));
      }
      cv::fillConvexPoly(m_local_mask, _points, cv::Scalar(255, 255, 255));
    }
    m_armor_roi = cv::Rect2f(centerUp.x - width * 0.5, centerUp.y - width * 0.5, width, width);
    m_center_roi = cv::Rect2f(centerDown.x - width * 0.5, centerDown.y - width * 0.5, width, width);
    // 调整 roi 不超过图像的边界
    tools::reset_roi(m_armor_roi, m_global_roi);
    tools::reset_roi(m_center_roi, m_global_roi);
    // 如果上一帧中心 R 坐标不在中心 roi 中，则交换装甲板和中心 R 的 roi
    cv::Rect2f centerRoiGlobal{m_center_roi.x + m_global_roi.x, m_center_roi.y + m_global_roi.y,
                               m_center_roi.width, m_center_roi.height};
    if (tools::in_rect(m_centerR.m_center_R, centerRoiGlobal) == false) {
      std::swap(m_armor_roi, m_center_roi);
    }

    if (whether_use_debug_local_roi) {
      draw(m_armor_roi, YELLOW);
      draw(m_center_roi, DRAW_COLOR);
    }
  }

  /**
   * @brief Construct a new Lightline:: Lightline object
   * @param[in] contour       轮廓点集
   * @param[in] roi           roi 用来设置正确的中心及角点
   */
  LightLine::LightLine(const std::vector<cv::Point>& contour, const cv::Rect2f& global_roi,
                       const cv::Rect2f& local_roi)
      : m_contour(contour)
      , m_contour_area(cv::contourArea(contour))
      , m_rotated_rect(cv::minAreaRect(contour))
  {

    // 长的为 length，短的为 width
    m_width = m_rotated_rect.size.width, m_length = m_rotated_rect.size.height;
    if (m_width > m_length) {
      std::swap(m_width, m_length);
    }
    m_aspect_ratio = m_length / m_width;
    m_center = m_rotated_rect.center;
    m_angle = m_rotated_rect.angle;
    m_area = m_rotated_rect.size.width * m_rotated_rect.size.height;
    std::array<cv::Point2f, 4> points;
    m_rotated_rect.points(points.begin());
    /**
     * OpenCV 中 RotatedRect::points() 角点顺序为顺时针，p[0]
     * 为纵坐标最大的点。若有多个纵坐标最大，则取其中横坐标最大的点。 p[0] 到 p[3] 的边为
     * width，其邻边为 height。
     * 根据上述关系可以确立四个角点位置。如果是装甲板灯条，则其还需要结合中心 R 来得到中心 R
     * 参照下的角点位置。
     */
    if (m_rotated_rect.size.width > m_rotated_rect.size.height) {
      m_tl = points[1];
      m_tr = points[2];
      m_bl = points[0];
      m_br = points[3];
    } else {
      m_tl = points[0];
      m_tr = points[1];
      m_bl = points[3];
      m_br = points[2];
    }
    // 得到相对原图的角点和中心位置
    m_tl += local_roi.tl() + global_roi.tl();
    m_tr += local_roi.tl() + global_roi.tl();
    m_bl += local_roi.tl() + global_roi.tl();
    m_br += local_roi.tl() + global_roi.tl();
    m_center += local_roi.tl() + global_roi.tl();
    m_x = m_center.x, m_y = m_center.y;
  };

  //========================绘图函数===========================
  /**
   * @brief 绘制灯条
   * @param[in] lightline     灯条
   * @param[in] color         颜色
   * @param[in] thickness     线条宽度
   * @param[in] localRoi      局部 roi
   */
  void BuffDetection::draw(const LightLine& lightline, const cv::Scalar& color, const int thickness,
                           const cv::Rect2f& localRoi)
  {
    draw(lightline.m_rotated_rect, color, thickness, localRoi);
  }

  /**
   * @brief 绘制旋转矩形
   * @param[in] rotatedRect   旋转矩形
   * @param[in] color         颜色
   * @param[in] thickness     线条宽度
   * @param[in] localRoi      局部 roi
   */
  void BuffDetection::draw(const cv::RotatedRect& rotatedRect, const cv::Scalar& color,
                           const int thickness, const cv::Rect2f& localRoi)
  {
    std::array<cv::Point2f, 4> vertices;
    rotatedRect.points(vertices.begin());
    draw(vertices.begin(), vertices.size(), color, thickness, localRoi);
  }

  /**
   * @brief 绘制正矩形
   * @param[in] rect          矩形
   * @param[in] color         颜色
   * @param[in] thickness     线条宽度
   * @param[in] localRoi      局部 roi
   */
  void BuffDetection::draw(const cv::Rect2f& rect, const cv::Scalar& color, const int thickness,
                           const cv::Rect2f& localRoi)
  {
    cv::Rect2f temp = rect;
    temp.x += localRoi.x + m_global_roi.x;
    temp.y += localRoi.y + m_global_roi.y;
    cv::rectangle(m_image_show, temp, color, thickness);
  }

  /**
   * @brief 绘制多边形，输入为向量
   * @param[in] points        多边形点集
   * @param[in] color         颜色
   * @param[in] thickness     线条宽度
   * @param[in] localRoi      局部 roi
   */
  void BuffDetection::draw(const std::vector<cv::Point2f>& points, const cv::Scalar& color,
                           const int thickness, const cv::Rect2f& localRoi)
  {
    for (size_t i = 0; i < points.size(); ++i) {
      cv::line(m_image_show, points[i] + localRoi.tl() + m_global_roi.tl(),
               points[(i + 1) % points.size()] + localRoi.tl() + m_global_roi.tl(), color,
               thickness);
    }
  }

  /**
   * @brief 绘制多边形，输入为 c 数组
   * @param[in] points        数组第一个元素的指针
   * @param[in] size          数组元素数量
   * @param[in] color         颜色
   * @param[in] thickness     线条宽度
   * @param[in] localRoi      局部 roi
   */
  void BuffDetection::draw(const cv::Point2f* points, const size_t size, const cv::Scalar& color,
                           const int thickness, const cv::Rect2f& localRoi)
  {
    for (size_t i = 0; i < size; ++i) {
      cv::line(m_image_show, points[i] + localRoi.tl() + m_global_roi.tl(),
               points[(i + 1) % size] + localRoi.tl() + m_global_roi.tl(), color, thickness);
    }
  }

  /**
   * @brief
   * 设置装甲板角点，注意大小为4，顺序必须为内部灯条左上角点、内部灯条右上角点、外部灯条左下角点、外部灯条右下角点
   * @param[in] points        输入的角点向量
   */
  void Armor::setCornerPoints(const std::vector<cv::Point2f>& points)
  {
    m_tlIn = points.at(0);
    m_trIn = points.at(1);
    m_blOut = points.at(2);
    m_brOut = points.at(3);
  }
} // namespace auto_buff