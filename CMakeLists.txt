cmake_minimum_required(VERSION 3.10)
project(onnx_debugger)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OpenCV_DIR "/usr/local/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)

message(STATUS "OpenCV version ${OpenCV_VERSION}")
message(STATUS "OpenCV include path ${OpenCV_INCLUDE_DIRS}")

include_directories(${OpenCV_INCLUDE_DIRS})

# 🔥 自动查找所有 cpp 文件
file(GLOB CPP_FILES *.cpp)

# 🔥 遍历每个 cpp，生成一个可执行文件
foreach(cpp_file ${CPP_FILES})
    # 提取文件名（不带扩展名）
    get_filename_component(target_name ${cpp_file} NAME_WE)

    add_executable(${target_name} ${cpp_file})
    target_link_libraries(${target_name} PRIVATE ${OpenCV_LIBS})

    message(STATUS "Add executable: ${target_name}")
endforeach()