cmake_minimum_required(VERSION 3.16)
set(OpenVINO_DIR "/home/c/opt/intel/openvino_ubuntu22_2025/runtime/cmake")

# 移除重复的find_package，依赖在主CMakeLists.txt中已经查找
# find_package(yaml-cpp REQUIRED)
# find_package(OpenVINO REQUIRED)
# find_package(Ceres REQUIRED)

# 使用STATIC或SHARED库，而不是OBJECT库
add_library(ecu STATIC
    camera.cpp
    serial_port.cpp
    hikrobot/hikrobot.cpp
)

target_include_directories(ecu PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/hikrobot/include
)

target_link_directories(ecu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/hikrobot/lib/amd64)

# 移除直接链接，在可执行文件中链接
# target_link_libraries(ecu openvino::runtime ${CERES_LIBRARIES} yaml-cpp)