cmake_minimum_required(VERSION 3.10)
project(OpticalFlow)

set(OpenCV_DIR D:/opencv/build)

find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(${PROJECT_NAME} src/optical_flow_cpu.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})