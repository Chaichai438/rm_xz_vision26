#include "function/auto_buff/rune_detector.hpp"
#include "tools/exiter.hpp"

using namespace cv;
using namespace std;

const std::string keys = "{help h usage ? |     | 输出命令行参数说明 }"
                         "{@config-path c | ../configs/test.yaml | "
                         "yaml配置文件的路径}";

int main(int argc, char* argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  auto config_path = cli.get<std::string>(0);

  VideoCapture cap("/home/chaichai/project/rm_xz_vision26/assets/04_origin.mp4");
  if (!cap.isOpened()) {
    std::cerr << "无法打开视频文件!" << std::endl;
    return -1;
  }

  auto_buff::BuffDetection detector(config_path);

  while (true) {
    cv::Mat raw_img;
    cap >> raw_img;

    if (raw_img.empty())
      break;

    auto t_now = std::chrono::steady_clock::now();
    auto_buff::Frame frame(raw_img, t_now, 0, 0, 0);

    detector.detect(frame);

    // 按 ESC 退出, 按空格暂停
    int key = cv::waitKey(30);
    if (key == 27) // ESC
      break;
    else if (key == 32) { // Space
      cv::waitKey(0);
    }
  }

  cap.release();
  cv::destroyAllWindows();
  return 0;
}