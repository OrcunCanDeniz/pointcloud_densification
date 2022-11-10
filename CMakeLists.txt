cmake_minimum_required(VERSION 3.5)
project(pointcloud_densification)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()


find_package(
        catkin REQUIRED COMPONENTS
        roscpp
        pcl_ros
        autoware_perception_msgs
        autoware_utils
        tf2_geometry_msgs
)

set(CUDA_VERBOSE OFF)

# set flags for CUDA availability
option(CUDA_AVAIL "CUDA available" OFF)
find_package(CUDA)
if(CUDA_FOUND)
  find_library(CUBLAS_LIBRARIES cublas HINTS
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
    ${CUDA_TOOLKIT_ROOT_DIR}/lib
  )
  if(CUDA_VERBOSE)
    message("CUDA is available!")
    message("CUDA Libs: ${CUDA_LIBRARIES}")
    message("CUDA Headers: ${CUDA_INCLUDE_DIRS}")
  endif()
  # Note: cublas_device was depreciated in CUDA version 9.2
  #       https://forums.developer.nvidia.com/t/where-can-i-find-libcublas-device-so-or-libcublas-device-a/67251/4
  #       In LibTorch, CUDA_cublas_device_LIBRARY is used.
  unset(CUDA_cublas_device_LIBRARY CACHE)
  set(CUDA_AVAIL ON)
else()
  message("CUDA NOT FOUND")
  set(CUDA_AVAIL OFF)
endif()

  catkin_package(
          INCLUDE_DIRS
          include
          CATKIN_DEPENDS
          roscpp
          pcl_ros
          autoware_utils
  )

  include_directories(
    include
    lib/include
    ${catkin_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
  )

  ### centerpoint ###
  add_library(densification SHARED
    lib/src/pointcloud_densification.cpp
  )
  cuda_add_library(densificationCUDA lib/src/kernels.cu)

  target_link_libraries(densification
    ${catkin_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${CUBLAS_LIBRARIES}
    ${CUDA_curand_LIBRARY}
    densificationCUDA
  )

  ## node ##
  add_executable(densification_node 
    src/node.cpp
    src/main.cpp
  )

  target_link_libraries(densification_node
  densification
    ${catkin_LIBRARIES}

  )

  add_dependencies(densification_node ${catkin_EXPORTED_TARGETS})

  install(
          TARGETS
          densification_node
          densification
          ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
          LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
          RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  )

  install(DIRECTORY include/${PROJECT_NAME}/
          DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
          )

  install(DIRECTORY launch/
          DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
          )

  install(DIRECTORY data/
          DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/data
          )