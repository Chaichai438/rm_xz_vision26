cmake_minimum_required(VERSION 3.16.3)

project(rm_xz_vision)

add_definitions(-DSPDLOG_FMT_EXTERNAL=1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
message(STATUS "--------------------CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}--------------------")

# 查找库部分
find_library(MVCAMERA_CONTROL_LIB
    NAMES MvCameraControl
    PATHS ${CMAKE_SOURCE_DIR}/ecu/hikrobot/lib/amd64
    REQUIRED
)

# 查找libusb库
find_library(LIBUSB_LIB
    NAMES usb-1.0
    PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu
    REQUIRED
)
set(LIBUSB_LIBRARIES ${LIBUSB_LIB})
set(MVCAMERA_CONTROL_LIBRARIES ${MVCAMERA_CONTROL_LIB})

find_package(OpenVINO REQUIRED)
find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(spdlog REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(nlohmann_json REQUIRED)

# set(OpenVINO_DIR "/home/c/opt/intel/openvino_ubuntu22_2025/runtime/cmake/")
find_package(OpenVINO REQUIRED)
find_package(Ceres REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})

# include_directories(/usr/include/libusb-1.0/home/c/opt/intel/openvino_ubuntu22_2025/runtime/include) # 直接添加libusb头文件路径
add_subdirectory(ecu)
add_subdirectory(function/auto_buff)
add_subdirectory(function/auto_aim)
add_subdirectory(tools)

# Add auto_buff library
add_library(auto_buff OBJECT
    function/auto_buff/rune_detector.cpp
    function/auto_buff/rune_solver.cpp
    function/auto_buff/power_rune.cpp
)

target_link_libraries(auto_buff openvino::runtime ${CERES_LIBRARIES} yaml-cpp)

# ================= test =================
add_executable(auto_buff_test test/auto_buff_test.cpp)
add_executable(auto_buff_main test/auto_buff_main.cpp)
add_executable(trad_cv_test test/trad_cv_test.cpp)
add_executable(camera_detect_test test/camera_detect_test.cpp)
add_executable(auto_aim_test test/auto_aim_test.cpp)

# add_executable(gimbal_response_test test/gimbal_response_test.cpp)
add_executable(plotjuggler_test test/plotjuggler_test.cpp)
add_executable(gimbal_test test/gimbal_test.cpp)
add_executable(onnx_test test/onnx_test.cpp)

# ================= src =================
add_executable(auto_aim_debug src/auto_aim_debug.cpp)

target_link_libraries(auto_buff_test ${OpenCV_LIBS} auto_buff ecu yaml-cpp tools fmt::fmt)
target_link_libraries(auto_buff_main ${OpenCV_LIBS} auto_buff yaml-cpp tools fmt::fmt)

# target_link_libraries(gimbal_response_test ${OpenCV_LIBS} ecu yaml-cpp tools fmt::fmt)
target_link_libraries(plotjuggler_test ${OpenCV_LIBS} yaml-cpp tools fmt::fmt)
target_link_libraries(gimbal_test ${OpenCV_LIBS} ecu yaml-cpp tools fmt::fmt serial)

target_link_libraries(auto_aim_debug ${OpenCV_LIBS} auto_aim ecu yaml-cpp tools fmt::fmt)

target_link_libraries(camera_detect_test PRIVATE
    ${OpenCV_LIBS}
    auto_buff
    tools
    ecu
    auto_aim
    yaml-cpp
    fmt::fmt
    openvino::runtime
    ${CERES_LIBRARIES}
    ${MVCAMERA_CONTROL_LIB} # 添加海康相机库
    ${LIBUSB_LIB} # 添加libusb库
)

target_link_libraries(auto_aim_test PRIVATE
    ${OpenCV_LIBS}
    auto_aim
    tools
    ecu
    yaml-cpp
    fmt::fmt
    spdlog::spdlog
    openvino::runtime
    ${CERES_LIBRARIES}
    ${MVCAMERA_CONTROL_LIB}
    ${LIBUSB_LIB}
)

target_link_libraries(onnx_test PRIVATE
    ${OpenCV_LIBS}
    auto_aim
    tools
    ecu
    yaml-cpp
    fmt::fmt
    spdlog::spdlog
    openvino::runtime
    ${CERES_LIBRARIES}
    ${MVCAMERA_CONTROL_LIB}
    ${LIBUSB_LIB}
)

# target_link_libraries(auto_buff_test PRIVATE
# ${OpenCV_LIBS}
# auto_buff
# tools
# ecu
# auto_aim
# yaml-cpp
# fmt::fmt
# openvino::runtime
# ${CERES_LIBRARIES}
# ${MVCAMERA_CONTROL_LIB} # 添加海康相机库
# ${LIBUSB_LIB} # 添加libusb库
# )
target_link_libraries(trad_cv_test PRIVATE
    ${OpenCV_LIBS}
    tools
    ecu
    auto_aim
    yaml-cpp
    fmt::fmt
    openvino::runtime
    ${CERES_LIBRARIES}
    ${MVCAMERA_CONTROL_LIB} # 添加海康相机库
    ${LIBUSB_LIB} # 添加libusb库
)

# target_link_libraries(exit_test PRIVATE
# ${OpenCV_LIBS}
# auto_buff
# tools
# ecu
# auto_aim
# yaml-cpp
# fmt::fmt
# openvino::runtime
# ${CERES_LIBRARIES}
# ${MVCAMERA_CONTROL_LIB} # 添加海康相机库
# ${LIBUSB_LIB} # 添加libusb库
# )
add_executable(test_vision test/test_vision.cpp)

target_link_libraries(test_vision
    ${OpenCV_LIBS}
    tools
    ecu
    auto_aim
    yaml-cpp
    fmt::fmt
    openvino::runtime
    ${CERES_LIBRARIES}
)