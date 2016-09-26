cmake_minimum_required(VERSION 2.8)
project( algo_streak_detection CXX)
find_package( OpenCV REQUIRED )
if(UNIX)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=gnu++0x")
endif()
add_executable( algo_streak_detection ./src/main.cpp 
                                      ./src/main_simple.cpp
                                      ./src/main_2.cpp
                                      ./src/function.cpp
				                              ./src/main_GPU.cpp
                                      ./src/function_GPU.cpp)
target_link_libraries( algo_streak_detection ${OpenCV_LIBS} )

