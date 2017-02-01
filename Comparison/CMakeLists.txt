cmake_minimum_required(VERSION 2.8)

project(object-detecting)
find_package(OpenCV REQUIRED)

include_directories(include)

#add_executable(examine_tm examine_tm.cpp
#	template_matching_examination.cpp
#	template_matching.cpp)
#target_link_libraries(examine_tm ${OpenCV_LIBS})

#add_executable(examine_cm examine_cm.cpp
#	contour_matching_examination.cpp
#	contour_matching.cpp)
#target_link_libraries(examine_cm ${OpenCV_LIBS})

add_executable(examine_fm examine_fm.cpp
	feature_matching_examination.cpp
	feature_matching.cpp)
target_link_libraries(examine_fm ${OpenCV_LIBS})
