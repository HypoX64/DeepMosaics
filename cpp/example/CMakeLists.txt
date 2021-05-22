# project(example)
# add_executable("${PROJECT_NAME}" deepmosaic.cpp)
# target_link_libraries(  "${PROJECT_NAME}" 
#                         "${TORCH_LIBRARIES}" 
#                         "${OpenCV_LIBS}"
#                         utils)

file(GLOB_RECURSE srcs RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
foreach(sourcefile IN LISTS srcs)
    string( REPLACE ".cpp" "" binname ${sourcefile})
    add_executable( ${binname} ${sourcefile} )
    target_link_libraries(  ${binname} 
                            "${TORCH_LIBRARIES}" 
                            "${OpenCV_LIBS}"
                            utils)
    # set_property(TARGET ${binname} PROPERTY CXX_STANDARD 14)
endforeach()