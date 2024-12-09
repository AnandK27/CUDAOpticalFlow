cmake_minimum_required(VERSION 3.10)
project(OpticalFlow LANGUAGES CXX CUDA)

set(OpenCV_DIR D:/opencv/build)
set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_TOOLKIT_ROOT_DIR})
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

find_package(OpenCV REQUIRED)
find_package(CUDA 12.6)
find_package(NVTX REQUIRED)

set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-lnvToolExt -gencode arch=compute_89,code=sm_89)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++1y" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1y")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()


include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include ${OpenCV_INCLUDE_DIRS} )

add_executable(${PROJECT_NAME} src/optical_flow_gpu.cu)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
set_target_properties(OpticalFlow PROPERTIES CUDA_ARCHITECTURES "35;50;72;89")