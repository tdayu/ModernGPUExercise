cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

project(ModernGPUExercise VERSION 0.1 LANGUAGES C CXX CUDA)

add_library(particles STATIC
  randomize.cpp
  randomize.h
  particle.cu
  particle.h
  v3.cu
  v3.h
)

# set(CMAKE_CXX_EXTENSIONS OFF) # Generate -std=c++17 instead of -std=gnu++17
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DNDEBUG")
# set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
# set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DALLEN_DEBUG")
# set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

