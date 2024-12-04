cmake_minimum_required(VERSION 3.10)
project(OpticalFlow)
 
find_package(OpenCV REQUIRED)
set(OpenCV_INCLUDE_DIRS C:/opencv/build/include)
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(${PROJECT_NAME} src/optical_flow_cpu.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})