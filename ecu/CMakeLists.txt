cmake_minimum_required(VERSION 3.16)
set(OpenVINO_DIR "/home/c/opt/intel/openvino_ubuntu22_2025/runtime/cmake")
set(DAHENG_ROOT ${CMAKE_SOURCE_DIR}/daheng)

# 移除重复的find_package，依赖在主CMakeLists.txt中已经查找
find_package(yaml-cpp REQUIRED)

# find_package(OpenVINO REQUIRED)
# find_package(Ceres REQUIRED)
add_subdirectory(serial)

# 使用STATIC或SHARED库，而不是OBJECT库
add_library(ecu STATIC
    camera.cpp
    hikrobot/hikrobot.cpp
    gimbal/gimbal.cpp
    daheng/daheng.cpp
)

# hikrobot
target_include_directories(ecu PUBLIC hikrobot/include)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    target_link_directories(ecu PUBLIC hikrobot/lib/amd64)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    target_link_directories(ecu PUBLIC hikrobot/lib/arm64)
else()
    message(FATAL_ERROR "Unsupported architecture: ${CMAKE_HOST_SYSTEM_PROCESSOR}!")
endif()

target_include_directories(ecu PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/hikrobot/include
    ${CMAKE_CURRENT_SOURCE_DIR}/serial/include
)

target_link_libraries(ecu PUBLIC
    serial
    yaml-cpp
    MvCameraControl # 链接海康 SDK 核心库
    usb-1.0 # 链接系统 libusb 库 (解决 libusb_init 等错误)
    pthread # 线程库支持
)

# 1. 设置头文件路径
target_include_directories(ecu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/daheng/include)

# 2. 根据架构选择对应的库路径
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    set(DAHENG_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/daheng/lib/x86_64)
else()
    message(FATAL_ERROR "Unsupported architecture for Daheng SDK: ${CMAKE_SYSTEM_PROCESSOR}!")
endif()

# 3. 添加库搜索目录
target_link_directories(ecu PUBLIC ${DAHENG_LIB_DIR})

# 4. 链接大恒的核心动态库
# gxiapi: 基础相机控制
# dximageproc: 图像处理（如 DxRaw8toRGB24）
target_link_libraries(ecu PUBLIC
    -lpthread
    -lboost_system
    libgxiapi.so
    gxiapi
)

# 移除直接链接，在可执行文件中链接
# target_link_libraries(ecu openvino::runtime ${CERES_LIBRARIES} yaml-cpp)