#include <thread>
#include "tools/exiter.hpp"

#include "function/auto_buff/power_rune.hpp"

const std::string keys = "{help h usage ? |     | 输出命令行参数说明 }"
                         "{@config-path c |     | yaml配置文件的路径}"
                         "{video_path v   |     |视频路径}";

int main(int argc, char* argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }

  auto config_path = cli.get<std::string>(0);
  auto video_path = cli.get<std::string>("video_path");
  tools::logger()->debug("Video initialized with config: {}", video_path);

  tools::Exiter exiter;
  auto_buff::PowerRune pr(config_path);
  cv::VideoCapture cap(video_path);
  if (!cap.isOpened()) { // 缺少检查
    tools::logger()->error("无法打开视频: {}", video_path);
    return -1;
  }

  while (!exiter.exit()) {
    auto start{std::chrono::steady_clock::now()};

    cv::Mat raw_img;
    cap >> raw_img;
    if (raw_img.empty())
      break;

    pr.run_once_debug(raw_img, 0.0, 0.0);
    // pr.run_once(raw_img, 0.0, 0.0);

    // 控制主循环的帧率
    auto future_time = start + std::chrono::milliseconds(1000 / auto_buff::BuffSolver::get_fps());
    if (std::chrono::steady_clock::now() < future_time) {
      std::this_thread::sleep_until(future_time);
    }

    // 按 ESC 退出, 按空格暂停
    int key = cv::waitKey(30);
    if (key == 27) // ESC
      break;
    else if (key == 32) { // Space
      cv::waitKey(0);
    }
  }
  return 0;
}