cmake_minimum_required(VERSION 3.16)
set(OpenVINO_DIR "/home/c/opt/intel/openvino_ubuntu22_2025/runtime/cmake")

# 移除重复的find_package，依赖在主CMakeLists.txt中已经查找
find_package(yaml-cpp REQUIRED)

# find_package(OpenVINO REQUIRED)
# find_package(Ceres REQUIRED)
add_subdirectory(serial)

# 使用STATIC或SHARED库，而不是OBJECT库
add_library(ecu STATIC
    camera.cpp
    serial_port.cpp
    hikrobot/hikrobot.cpp
    gimbal/gimbal.cpp
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

target_link_directories(ecu PUBLIC serial yaml-cpp)

# 移除直接链接，在可执行文件中链接
# target_link_libraries(ecu openvino::runtime ${CERES_LIBRARIES} yaml-cpp)