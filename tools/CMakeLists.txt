cmake_minimum_required(VERSION 3.16)

add_library(tools OBJECT
    exiter.cpp
    logger.cpp
    imgs_tool.cpp
    draw_tool.cpp
    plotter.cpp
    rotary_tool.cpp
    extended_kalman_filter.cpp
    trajectory.cpp
    crc.cpp
)