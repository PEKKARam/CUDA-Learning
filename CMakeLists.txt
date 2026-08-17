cmake_minimum_required(VERSION 3.24)

project(cuda_learning LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Profile Release)
set(CMAKE_CXX_FLAGS_PROFILE "-O3 -g" CACHE STRING "C++ flags for Profile builds")
set(CMAKE_CUDA_FLAGS_PROFILE "-O3 -lineinfo" CACHE STRING
    "CUDA flags for Profile builds")

if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures")
endif()

option(CUDA_LEARNING_BUILD_TESTS "Build standalone correctness tests" ON)
option(CUDA_LEARNING_BUILD_BENCHMARKS "Build standalone benchmarks" ON)

file(GLOB_RECURSE CUDA_KERNEL_SOURCES CONFIGURE_DEPENDS src/*.cu)
add_library(cuda_kernels STATIC ${CUDA_KERNEL_SOURCES})
target_include_directories(cuda_kernels PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(cuda_kernels PUBLIC CUDA::cudart)
set_target_properties(cuda_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
target_compile_options(cuda_kernels PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:-O0;-g>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-O0;-G>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Profile>>:-O3;-g>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Profile>>:-O3;-lineinfo>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-O3>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3>
)

if(CUDA_LEARNING_BUILD_TESTS)
    enable_testing()

    foreach(test_name IN ITEMS square reduce sgemm)
        add_executable(test_${test_name} tests/cpp/test_${test_name}.cu)
        target_link_libraries(test_${test_name} PRIVATE cuda_kernels)
        add_test(NAME ${test_name} COMMAND test_${test_name})
    endforeach()
endif()

if(CUDA_LEARNING_BUILD_BENCHMARKS)
    foreach(benchmark_name IN ITEMS square reduce sgemm)
        add_executable(benchmark_${benchmark_name}
            benchmarks/benchmark_${benchmark_name}.cu
        )
        target_link_libraries(benchmark_${benchmark_name}
            PRIVATE cuda_kernels CUDA::nvtx3
        )
    endforeach()
    target_link_libraries(benchmark_sgemm PRIVATE CUDA::cublas)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
