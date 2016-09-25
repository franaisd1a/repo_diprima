cmake_minimum_required(VERSION 2.8)
project( algo_streak_detection CXX)

# Declare name of main target (don't use '-' in names)
set(    MAINTGT algo_streak_detection)
# Configure basic behaviors on include directories
set(CMAKE_MODULE_PATH "${${MAINTGT}_SOURCE_DIR}/cmake/modules")

find_package( OpenCV REQUIRED )
#find_package(CFITSIO REQUIRED)
find_package( CCFits REQUIRED )

# Setup variables: source and include paths
set(SRC_PATH "src")
set(INC_PATH "inc")

if(UNIX)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=gnu++0x")
endif()
add_executable( algo_streak_detection 
				      ${SRC_PATH}//main.cpp 
                                      ${SRC_PATH}//main_simple.cpp
                                      ${SRC_PATH}//main_2.cpp
                                      ${SRC_PATH}//main_fits.cpp                                                                           
                                      ${SRC_PATH}//cookbook.cxx
				      ${SRC_PATH}//function.cpp)
#./src/main_GPU.cpp
#                                      ./src/function_GPU.cpp)
target_link_libraries( algo_streak_detection ${OpenCV_LIBS} )

