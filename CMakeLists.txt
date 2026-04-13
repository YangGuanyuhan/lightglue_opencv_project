cmake_minimum_required(VERSION 3.10)
project(onnx_debugger)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Allow command-line override: cmake -DOpenCV_DIR=/path/to/opencv/build
if(NOT DEFINED OpenCV_DIR)
    set(OpenCV_DIR "/home/christylinux/opencv_build/opencv/build")
endif()
find_package(OpenCV REQUIRED)

message(STATUS "OpenCV version ${OpenCV_VERSION}")
message(STATUS "OpenCV_DIR ${OpenCV_DIR}")
message(STATUS "OpenCV include path ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libs ${OpenCV_LIBS}")

include_directories(${OpenCV_INCLUDE_DIRS})

# Auto-discover all cpp files
file(GLOB CPP_FILES *.cpp)

# Build one executable per cpp file
foreach(cpp_file ${CPP_FILES})
    # Extract filename without extension
    get_filename_component(target_name ${cpp_file} NAME_WE)

    add_executable(${target_name} ${cpp_file})
    target_link_libraries(${target_name} PRIVATE ${OpenCV_LIBS})

    message(STATUS "Add executable: ${target_name}")
endforeach()