# cribbed from 
# https://github.com/NH89/Morphognesis
# https://github.com/KhronosGroup/OpenCL-Guide/blob/main/chapters/getting_started_linux.md
# NB OpenCV-4 uses C++11 by default. 

cmake_minimum_required(VERSION 3.16   FATAL_ERROR) # 3.1 << C_STANDARD 11

cmake_policy(SET CMP0054 NEW)

project(Dynamic_slam LANGUAGES C CXX)
#add_subdirectory (utils)

set(CMAKE_C_STANDARD 17)
#set(CMAKE_CPP_STANDARD 11) # set(CMAKE_CPP_STANDARD 17)  NB need (<=11 for VTP)  &&  (>+11 for C++ threads)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)

find_package(Boost  REQUIRED COMPONENTS filesystem) # see https://cliutils.gitlab.io/modern-cmake/chapters/packages/Boost.html
message(STATUS "Boost version: ${Boost_VERSION}")

# This is needed if your Boost version is newer than your CMake version
# or if you have an old version of CMake (<3.5)
if(NOT TARGET Boost::filesystem)
    add_library(Boost::filesystem IMPORTED INTERFACE)
    set_property(TARGET Boost::filesystem PROPERTY
        INTERFACE_INCLUDE_DIRECTORIES ${Boost_INCLUDE_DIR})
    set_property(TARGET Boost::filesystem PROPERTY
        INTERFACE_LINK_LIBRARIES ${Boost_LIBRARIES})
endif()

find_package(OpenCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(jsoncpp REQUIRED)
#find_package(libtiff5 REQUIRED) #by my current build of opencv
#find_package(vtk REQUIRED)

if( OpenCV_FOUND )
  # Additional Include Directories
  include_directories( ${OpenCV_INCLUDE_DIRS} )

  # Additional Library Directories
  link_directories( ${OpenCV_LIB_DIR} )

  # Additional Dependencies
  
endif()

#### from https://docs.opencv.org/4.x/db/df5/tutorial_linux_gcc_cmake.html

#cmake_minimum_required(VERSION 2.8)
#project( DisplayImage )
#find_package( OpenCV REQUIRED )
#include_directories( ${OpenCV_INCLUDE_DIRS} )
#add_executable( DisplayImage DisplayImage.cpp )
#target_link_libraries( DisplayImage ${OpenCV_LIBS} )

####


add_executable(Dynamic_slam
        src/main.cpp

         src/Dynamic_slam/Dynamic_slam.hpp
         src/Dynamic_slam/Dynamic_slam.cpp
         src/Dynamic_slam/Dynamic_slam_binocular_disparity.cpp
         src/Dynamic_slam/Dynamic_slam_keyframe.cpp
         src/Dynamic_slam/Dynamic_slam_mapping.cpp
         src/Dynamic_slam/Dynamic_slam_patch_slam.cpp
         src/Dynamic_slam/Dynamic_slam_tracking.cpp

         src/Dynamic_slam/testing/Dynamic_slam_ground_truth.cpp
         src/Dynamic_slam/testing/Dynamic_slam_results.cpp

         src/kernels/kernels_macros.h
         src/RunCL/RunCL.hpp
         src/RunCL/RunCL.cpp
         src/RunCL/RunCL_binocular_disparity.cpp
         src/RunCL/RunCL_DownloadAndSave.cpp
         src/RunCL/RunCL_load_image.cpp
         src/RunCL/RunCL_mapping.cpp
         src/RunCL/RunCL_patch_slam.cpp
         src/RunCL/RunCL_setup_keyframe.cpp
         src/RunCL/RunCL_tracking.cpp
         src/RunCL/RunCL_utils.cpp

         src/utils/CV_chk.hpp
         src/utils/CV_chk.cpp
         src/utils/time_utils.hpp
         src/utils/time_utils.cpp
         src/utils/conf_params.hpp
         src/utils/conf_params.cpp
         src/utils/print_functions.hpp
         src/utils/print_functions.cpp
         src/utils/convertTransforms.hpp
         src/utils/convertTransforms.cpp
         src/utils/convertAhandaPovRayToStandard.hpp
         src/utils/convertAhandaPovRayToStandard.cpp

         )

target_link_libraries(${PROJECT_NAME} PRIVATE 
        OpenCL::OpenCL  
        ${OpenCV_LIBS} 
        Boost::filesystem
        jsoncpp_lib


        )

set_target_properties(${PROJECT_NAME} PROPERTIES C_STANDARD 17
                                                 C_STANDARD_REQUIRED ON
                                                 C_EXTENSIONS OFF)

target_compile_definitions(${PROJECT_NAME} PRIVATE CL_TARGET_OPENCL_VERSION=300) #200 #120) # i.e. OpenCL 1.2
