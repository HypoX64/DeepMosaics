# Set the project name
project (utils)
aux_source_directory(./src DIR_LIB_SRCS)

add_library(${PROJECT_NAME} SHARED ${DIR_LIB_SRCS})

# 添加.h文件查找路径
target_include_directories( ${PROJECT_NAME}
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)

# 添加掉用的第三方库
target_link_libraries(  "${PROJECT_NAME}" 
                        "${OpenCV_LIBS}")