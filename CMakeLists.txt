
cmake_minimum_required(VERSION 2.8)
add_definitions(-std=c++11)
project(image-stitching)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(cmake_c_flags_debug "-g -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -O0 -g -pthread -fopenmp -gdwarf-2 ")

SET(OpenCV_DIR /home/ld/project/opencv/build)
find_package(OpenCV 4 REQUIRED)

find_package(Eigen3 REQUIRED)
find_package(CUDA REQUIRED)
find_package(yaml-cpp REQUIRED)


include_directories(
        ${OpenCV_INCLUDE_DIRS}
        include
)
set(PROJECT_DIR "/home/ld/project/SuperPoint-SuperGlue-TensorRT" )

include_directories(
  ${PROJECT_DIR}
  ${PROJECT_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIR}
)

include_directories(/home/ld/software/TensorRT-10.13.3.9/include)

link_directories(/home/ld/project/SuperPoint-SuperGlue-TensorRT/build)
link_directories(/home/ld/project/SuperPoint-SuperGlue-TensorRT/3rdparty/tensorrtbuffer/lib)
link_directories(/home/ld/software/TensorRT-10.13.3.9/targets/x86_64-linux-gnu/lib)


add_executable(${PROJECT_NAME}
        src/app.cc include/app.h
        src/sensor_data_interface.cc include/sensor_data_interface.h
        src/stitching_param_generater.cc include/stitching_param_generater.h
        src/image_stitcher.cc include/image_stitcher.h)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} -lpthread -lm libsuperpointglue_lib.so   nvinfer nvonnxparser ${CUDA_LIBRARIES} yaml-cpp libtensorrtbuffer.so)
