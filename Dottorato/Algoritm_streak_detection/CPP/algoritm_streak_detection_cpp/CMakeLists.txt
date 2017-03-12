cmake_minimum_required(VERSION 2.8)
project( algo_streak_detection) #project( algo_streak_detection CXX) 

set(MAINTGT algo_streak_detection)

find_package(  OpenCV REQUIRED )
find_package( CFITSIO REQUIRED )
find_package(    CUDA    QUIET )

message("Find CUDA: " ${CUDA_FOUND})


if(UNIX)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=gnu++0x")
endif()

# Setup variables: source and include paths
set(SRC_PATH "src")
set(INC_PATH "inc")

# Opereting system functions
if (${WIN32})
set(SOURCES_OS ${SRC_PATH}/function_os_win.cpp)
set(HEADERS_OS    ${INC_PATH}/function_os_win.h)
elseif (${UNIX})
set(SOURCES_OS ${SRC_PATH}/function_os_posix.cpp)
set(HEADERS_OS    ${INC_PATH}/function_os_posix.h)
else   ()
  message("Building for unknown system.")
endif  ()

#CUDA Settings
if (${CUDA_FOUND})

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --gpu-architecture sm_20)

#set(CUDA_SRC src/medianKernel.cu)

set(CUDA_SRC  
    ${SRC_PATH}/main.cpp
    ${SRC_PATH}/file_selection.cpp
    ${SRC_PATH}/algo_selection.cpp
    ${SRC_PATH}/main_simple.cpp
    ${SRC_PATH}/main_sigmaClip.cpp
	   ${SRC_PATH}/main_sigmaClipSimple.cpp
    ${SRC_PATH}/main_sigmaClipSimpleBig.cpp
    ${SRC_PATH}/main_fits.cpp
    ${SRC_PATH}/main_GPU.cpp
    ${SRC_PATH}/main_GPU_cuda.cu
    ${SRC_PATH}/function.cpp
    ${SRC_PATH}/function_GPU.cpp)
    #${SRC_PATH}/externalClass.cu)

# Set header files
set(HEADERS
  ${INC_PATH}/file_selection.h
  ${INC_PATH}/algo_selection.h
		${INC_PATH}/main_simple.h
  ${INC_PATH}/main_sigmaClip.h
	 ${INC_PATH}/main_sigmaClipSimple.h
  ${INC_PATH}/main_sigmaClipSimpleBig.h
  ${INC_PATH}/main_fits.h
  ${INC_PATH}/main_GPU.h
  ${INC_PATH}/function.h
  ${INC_PATH}/function_GPU.h
  ${INC_PATH}/function_os.h
  ${HEADERS_OS}
  )

#add_executable
CUDA_ADD_EXECUTABLE( ${MAINTGT} ${CUDA_SRC} ${HEADERS})


else   ()
# Set source files
set(SOURCES  
    ${SRC_PATH}/main.cpp
    ${SRC_PATH}/file_selection.cpp
    ${SRC_PATH}/algo_selection.cpp
			 ${SRC_PATH}/main_simple.cpp
    ${SRC_PATH}/main_sigmaClip.cpp
	   ${SRC_PATH}/main_sigmaClipSimple.cpp
    ${SRC_PATH}/main_sigmaClipSimpleBig.cpp
			 ${SRC_PATH}/main_fits.cpp
			 ${SRC_PATH}/function.cpp
    ${SOURCES_OS}
			 )

# Set header files
set(HEADERS
  ${INC_PATH}/file_selection.h
  ${INC_PATH}/algo_selection.h
		${INC_PATH}/main_simple.h
  ${INC_PATH}/main_sigmaClip.h
	 ${INC_PATH}/main_sigmaClipSimple.h
  ${INC_PATH}/main_sigmaClipSimpleBig.h
  ${INC_PATH}/main_fits.h
  ${INC_PATH}/function.h
  ${INC_PATH}/function_os.h
  ${HEADERS_OS}
  )

#add_executable
add_executable( ${MAINTGT} ${SOURCES} ${HEADERS})

endif  ()


	
# Update the central variable containing all external libraries
set(LIBS
  ${OpenCV_LIBS}
  ${CFITSIO_LIBRARY}
)

# Update the central variable containing all external includes
set(INCS
  ${CFITSIO_INCLUDE_DIRS}
)


target_include_directories(${MAINTGT} PRIVATE ${INCS} ${INC_PATH})

target_link_libraries( ${MAINTGT} ${LIBS} )

set(CMAKE_CXX_FLAGS "-g -Wall -std=c++11")

#set(CMAKE_CXX_FLAG "-g -Wall -std=gnu++11")-std=c++11
