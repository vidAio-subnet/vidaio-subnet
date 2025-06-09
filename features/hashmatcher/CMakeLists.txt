cmake_minimum_required(VERSION 3.10)
project(VideoMatcher)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "cmake/" CACHE INTERNAL "")

# Dependencies
find_package(OpenCV REQUIRED)
find_package(FFTW REQUIRED)
find_package(pybind11 REQUIRED)

# ========== cmatcher Python module ==========
pybind11_add_module(
    cmatcher
    matcher.cpp
)

target_link_libraries(
    cmatcher
    PRIVATE
    ${OpenCV_LIBS}
    ${FFTW_LIBRARIES}
    pthread
)