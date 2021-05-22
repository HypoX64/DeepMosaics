cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 14)

project(DeepMosaics)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib) #链接库路径

set(Torch_DIR /home/hypo/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)

set(OpenCV_DIR /home/hypo/opencv-4.4.0)
find_package(OpenCV REQUIRED)

# Add sub directories
add_subdirectory(example)
add_subdirectory(utils)

# set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)
# cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
# project(main)
# set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib) #链接库路径

# set(Torch_DIR /home/hypo/libtorch/share/cmake/Torch)
# find_package(Torch REQUIRED)

# set(OpenCV_DIR /home/hypo/opencv-4.4.0)
# find_package(OpenCV REQUIRED)

# # 查找当前目录下的所有源文件
# # 并将名称保存到 DIR_SRCS 变量
# # aux_source_directory(. DIR_SRCS)
# add_subdirectory(utils)

# add_executable(main main.cpp)
# # target_link_libraries(main )
# # include_directories( "${OpenCV_INCLUDE_DIRS}" )
# target_link_libraries( main "${TORCH_LIBRARIES}" "${OpenCV_LIBS}" utils)

# set_property(TARGET main PROPERTY CXX_STANDARD 14)
