cmake_minimum_required(VERSION 2.8)
project( algo_streak_detection)

find_package( OpenCV REQUIRED )
find_package( CUDA REQUIRED )

# Setup variables: source and include paths
set( SRC_PATH "src")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --gpu-architecture sm_20)

#set(CUDA_SRC src/medianKernel.cu)

set(CUDA_SRC  ${SRC_PATH}/main.cpp 
              ${SRC_PATH}/main_simple.cpp
              ${SRC_PATH}/main_2.cpp
              ${SRC_PATH}/main_fits.cpp
              ${SRC_PATH}/main_median.cu
              ${SRC_PATH}/function.cpp
              ${SRC_PATH}/function_GPU.cpp)
              #${SRC_PATH}/externalClass.cu)

#add_executable
CUDA_ADD_EXECUTABLE( algo_streak_detection ${CUDA_SRC})

target_link_libraries( algo_streak_detection ${OpenCV_LIBS} )

set(CMAKE_CXX_FLAGS "-g -Wall -std=c++11")
#set(CMAKE_CXX_FLAG "-g -Wall -std=gnu++11")-std=c++11