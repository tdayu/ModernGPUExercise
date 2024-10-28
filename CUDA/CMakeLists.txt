cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

project(ModernGPUExercise VERSION 0.1 LANGUAGES C CXX CUDA)
# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# file(GLOB SOURCES "src/*.cu")
file(GLOB TEST_SOURCES "tests/*.cu")
file(GLOB BENCHMARK_SOURCES "benchmarks/*.cu")

# add_library(ModernGPUExercise STATIC
#   ${SOURCES}
# )

# target_include_directories(ModernGPUExercise
#   PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#   PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# target_link_libraries(Allen PRIVATE AllenLib)
# set_target_properties(Allen PROPERTIES BUILD_RPATH_USE_ORIGIN TRUE)

# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

foreach(TEST_SRC ${TEST_SOURCES})
  get_filename_component(OUTNAME ${TEST_SRC} NAME_WE)
  add_executable(${OUTNAME} ${TEST_SRC})
  set_target_properties(${OUTNAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "tests")
  target_include_directories(${OUTNAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endforeach()

foreach(BENCHMARK_SRC ${BENCHMARK_SOURCES})
  get_filename_component(OUTNAME ${BENCHMARK_SRC} NAME_WE)
  add_executable(${OUTNAME} ${BENCHMARK_SRC})
  set_target_properties(${OUTNAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "benchmarks")
  target_include_directories(${OUTNAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endforeach()
