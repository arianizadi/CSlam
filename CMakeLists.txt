cmake_minimum_required(VERSION 3.10)
project(CSlam VERSION 1.0)

# Specify C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set build type to Debug if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type (Debug/Release)" FORCE)
endif()

# Add debug flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0") # Include -g for debugging symbols and -O0 to disable optimizations

# Find required packages
find_package(OpenCV REQUIRED)
find_package(GTest REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(g2o REQUIRED)
find_package(Pangolin REQUIRED)

# Add include directories
include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${G2O_INCLUDE_DIRS}
  ${Pangolin_INCLUDE_DIRS}
)

# Collect source files
file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE HEADERS "include/*.hpp" "include/*.h")

# Create main library
add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})
target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBS}
  ${G2O_LIBRARIES}
  ${Pangolin_LIBRARIES}
  Eigen3::Eigen
)

# Create main executable
add_executable(${PROJECT_NAME}_run src/main.cpp)
target_link_libraries(${PROJECT_NAME}_run ${PROJECT_NAME})

# Tests
enable_testing()
file(GLOB_RECURSE TEST_SOURCES "tests/*.cpp")
add_executable(${PROJECT_NAME}_tests ${TEST_SOURCES})
target_link_libraries(${PROJECT_NAME}_tests
  ${PROJECT_NAME}
  GTest::GTest
  GTest::Main
)

# Add tests to CTest
add_test(NAME ${PROJECT_NAME}_tests COMMAND ${PROJECT_NAME}_tests)
