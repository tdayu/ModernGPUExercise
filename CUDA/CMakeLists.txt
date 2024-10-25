cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

project(ModernGPUExercise VERSION 0.1 LANGUAGES C CXX CUDA)

file(GLOB sources "src/*.cu")
file(GLOB targets "targets/*.cu")

add_library(ModernGPUExercise STATIC
  ${sources}
)

target_include_directories(ModernGPUExercise PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

allen_add_executable(Allen main/src/main.cpp)
add_dependencies(Allen Sequences)

target_link_libraries(Allen PRIVATE AllenLib)
set_target_properties(Allen PROPERTIES BUILD_RPATH_USE_ORIGIN TRUE)

foreach(target ${targets})
  get_filename_component(BARENAME ${target} NAME)
  string(REPLACE ".cu" "" OUTNAME ${BARENAME})
  add_executable(${OUTNAME} ${target})
  target_link_libraries(${OUTNAME} PRIVATE ModernGPUExercise)
endforeach()

# set(CMAKE_CXX_EXTENSIONS OFF) # Generate -std=c++17 instead of -std=gnu++17
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DNDEBUG")
# set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
# set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DALLEN_DEBUG")
# set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

