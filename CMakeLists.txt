cmake_minimum_required(VERSION 3.10)
project(onnx_debugger)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(OpenCV_DIR "/usr/local/lib/cmake/opencv4")
# Find OpenCV
find_package(OpenCV REQUIRED)

message(STATUS "OpenCV version ${OpenCV_VERSION}")
message(STATUS "OpenCV include path ${OpenCV_INCLUDE_DIRS}")

include_directories(${OpenCV_INCLUDE_DIRS})

# 新增：特征提取与匹配示例
add_executable(feature_extractor_matcher_demo example_for_featur_extractor_matcher.cpp)
target_link_libraries(feature_extractor_matcher_demo PRIVATE ${OpenCV_LIBS})

# add_executable(test_debug debuglayers.cpp)
# add_executable(test_superpoint superpoint_test.cpp)
# add_executable(test_lightglue test_lightglue.cpp)
# add_executable(test_aliked test_aliked.cpp)
# add_executable(test_xfeat test_xfeat.cpp)

# target_link_libraries(test_debug PRIVATE ${OpenCV_LIBS})
# target_link_libraries(test_superpoint PRIVATE ${OpenCV_LIBS})
# target_link_libraries(test_lightglue PRIVATE ${OpenCV_LIBS})
# target_link_libraries(test_aliked PRIVATE ${OpenCV_LIBS})
# target_link_libraries(test_xfeat PRIVATE ${OpenCV_LIBS})