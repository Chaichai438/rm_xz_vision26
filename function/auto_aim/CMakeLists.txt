cmake_minimum_required(VERSION 3.16)

set(OpenVINO_DIR "/home/c/opt/intel/openvino_ubuntu22_2025/runtime/cmake")

add_library(auto_aim STATIC
    detector.cpp
    armor.cpp
    classifier.cpp
    solver.cpp
    yolo.cpp
    yolos/yolov8.cpp
    target.cpp
    tracker.cpp
    aimer.cpp
)

target_include_directories(auto_aim PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/function/auto_aim
    /home/c/opt/intel/openvino_ubuntu22_2025/runtime/include # 手动添加 OpenVINO 包含路径
)

target_link_libraries(auto_aim PRIVATE
    openvino::runtime
)

# 可选：如果需要 OpenCV，添加
if(OpenCV_LIBS)
    target_link_libraries(auto_aim PRIVATE ${OpenCV_LIBS})
endif()