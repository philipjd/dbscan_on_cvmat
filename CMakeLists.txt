cmake_minimum_required(VERSION 3.5.1)
project(test_dbscan)

set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE RelWithDebInfo)

LIST(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/Modules/)
message(STATUS "  CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")

find_package(OpenCV REQUIRED)
#include_directories(${OpenCV_INCLUDE_DIRS})
message(STATUS "OpenCV_VERSION: ${OpenCV_VERSION}")
message("OpenCV include dir: ${OpenCV_INCLUDE_DIRS}")

find_package(Gflags REQUIRED)
include_directories(${GFLAGS_INCLUDE_DIRS})

find_package(Glog REQUIRED)
include_directories(${GLOG_INCLUDE_DIRS})

set(SOURCE_FILES dbscan.cpp main.cpp)
add_executable(test_dbscan ${SOURCE_FILES})
target_link_libraries(test_dbscan ${OpenCV_LIBS} ${GFLAGS_LIBRARIES} ${GLOG_LIBRARIES})
