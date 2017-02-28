cmake_minimum_required(VERSION 2.8)
project( algo_streak_detection) #project( algo_streak_detection CXX) 

set(MAINTGT algo_streak_detection)

find_package(  OpenCV REQUIRED )
find_package( CFITSIO REQUIRED )
#### find_package( CUDA REQUIRED )

if(UNIX)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=gnu++0x")
endif()

# Setup variables: source and include paths
set(SRC_PATH "src")
### set(INC_PATH "inc")


#### set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --gpu-architecture sm_20)
#### 
#### #set(CUDA_SRC src/medianKernel.cu)
#### 
#### set(CUDA_SRC  ${SRC_PATH}/main.cpp 
####               ${SRC_PATH}/main_simple.cpp
####               #${SRC_PATH}/main_2.cpp
####               #${SRC_PATH}/main_fits.cpp
####               ${SRC_PATH}/main_GPU.cpp
####               ${SRC_PATH}/main_GPU_cuda.cu
####               ${SRC_PATH}/function.cpp
####               ${SRC_PATH}/function_GPU.cpp)
####               #${SRC_PATH}/externalClass.cu)
#### 
#### #add_executable
#### CUDA_ADD_EXECUTABLE( algo_streak_detection ${CUDA_SRC})

if (${WIN32})
set(SOURCES_OS ${SRC_PATH}/function_os_win.cpp)
elseif (${UNIX})
set(SOURCES_OS ${SRC_PATH}/function_os_posix.cpp)
else   ()
  message("Building for unknown system.")
endif  ()


set(SOURCES  
    ${SRC_PATH}/main.cpp
    ${SRC_PATH}/file_selection.cpp
    ${SRC_PATH}/algo_selection.cpp
			 ${SRC_PATH}/main_simple.cpp
    ${SRC_PATH}/main_sigmaClip.cpp
	${SRC_PATH}/main_sigmaClipSimple.cpp
			 #${SRC_PATH}/main_2.cpp
			 ${SRC_PATH}/main_fits.cpp
			 #${SRC_PATH}/main_GPU.cpp
			 #${SRC_PATH}/main_GPU_cuda.cu
			 ${SRC_PATH}/function.cpp
			 #${SRC_PATH}/function_GPU.cpp)
			 #${SRC_PATH}/externalClass.cu
			 )

#add_executable			 
add_executable( ${MAINTGT} ${SOURCES} ${SOURCES_OS})
	
# Update the central variable containing all external libraries
set(LIBS
  ${OpenCV_LIBS}
  ${CFITSIO_LIBRARY}
)

# Update the central variable containing all external includes
set(INCS
  ${CFITSIO_INCLUDE_DIRS}
)

target_include_directories(${MAINTGT} PRIVATE ${INCS}) #${INC_PATH} 

target_link_libraries( ${MAINTGT} ${LIBS} )

set(CMAKE_CXX_FLAGS "-g -Wall -std=c++11")
	
#set(CMAKE_CXX_FLAG "-g -Wall -std=gnu++11")-std=c++11


