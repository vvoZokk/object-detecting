cmake_minimum_required(VERSION 2.8)

project(object-detecting)
find_package(OpenCV REQUIRED)

add_executable(generate generator.cpp)

target_link_libraries(generate ${OpenCV_LIBS})
