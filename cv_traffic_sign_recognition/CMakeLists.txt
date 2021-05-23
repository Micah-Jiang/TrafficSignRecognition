cmake_minimum_required(VERSION 3.17.5)
project(opencv)

set(CMAKE_CXX_STANDARD 14)

add_executable(opencv main.cpp)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(opencv ${OpenCV_LIBS})