cmake_minimum_required(VERSION 3.16.3)

project(particle_filter)

find_package( OpenCV REQUIRED )


add_executable ( run main.cpp )
target_include_directories(run PRIVATE includes/)
target_link_libraries ( run ${OpenCV_LIBS})