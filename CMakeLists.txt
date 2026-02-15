cmake_minimum_required(VERSION 3.20)
project(Radix2FFT LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_testing()

add_library(fft_lib
    src/fft_dispatch.cpp
    src/fft_radix2_iterative.cpp
    src/fft_radix2_recursive.cpp
    src/dft_reference.cpp
)

target_include_directories(fft_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(fft_lib PUBLIC cxx_std_20)

function(configure_target_warnings target_name)
    if(MSVC)
        target_compile_options(${target_name} PRIVATE /W4 /permissive- $<$<CONFIG:Release>:/O2>)
    else()
        target_compile_options(${target_name} PRIVATE -Wall -Wextra -Wpedantic $<$<CONFIG:Release>:-O3>)
    endif()
endfunction()

configure_target_warnings(fft_lib)

add_executable(fft_demo src/fft_demo.cpp)
target_link_libraries(fft_demo PRIVATE fft_lib)
configure_target_warnings(fft_demo)

add_executable(test_fft tests/test_fft.cpp)
target_link_libraries(test_fft PRIVATE fft_lib)
configure_target_warnings(test_fft)

add_executable(fft_benchmark src/fft_benchmark.cpp)
target_link_libraries(fft_benchmark PRIVATE fft_lib)
configure_target_warnings(fft_benchmark)

add_test(NAME test_fft COMMAND test_fft)
