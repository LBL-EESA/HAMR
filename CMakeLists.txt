cmake_minimum_required(VERSION 3.18)
project(hamr)

include(CheckCXXSourceCompiles)
include(CheckCXXCompilerFlag)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_C_VISIBILITY_PRESET hidden)

set(hamr_libs)

# if we forgot to set the build type default to release
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release"
      CACHE STRING "One of: Debug Release RelWithDebInfo MinSizeRel"
      FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY
        STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()
message(STATUS "HAMR: Configuring a ${CMAKE_BUILD_TYPE} build")

# enable Python
set(HAMR_ENABLE_PYTHON OFF CACHE BOOL "Enable Python bindings")
if (HAMR_ENABLE_PYTHON)
    message(STATUS "HAMR: Python features -- enabled")
else()
    message(STATUS "HAMR: Python features -- disabled")
endif()

# enable verbose debugging reports
set(HAMR_VERBOSE OFF CACHE BOOL
    "Enable debug messages when HAMR_VERBOSE is found in the environment")

# enable CUDA
set(HAMR_ENABLE_CUDA OFF CACHE BOOL
    "Enable features supporting the CUDA memory model")

set(HAMR_CUDA_OBJECTS ON CACHE BOOL
    "Disable use of classes in CUDA allocators etc")

set(HAMR_CUDA_ARCHITECTURES "60;70;75" CACHE
    STRING "Compile for these CUDA virtual and real architectures")

if (HAMR_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if (CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        message(STATUS "HAMR: CUDA features -- enabled (${HAMR_CUDA_ARCHITECTURES})")
    else()
        message(FATAL_ERROR "CUDA is required for hamr but was not found")
    endif()
    find_package(CUDAToolkit REQUIRED)
else()
    message(STATUS "HAMR: CUDA features -- disabled")
endif()

# enable HIP
set(HAMR_ENABLE_HIP OFF CACHE BOOL
    "Enable features supporting the HIP memory model")

set(HAMR_HIP_OBJECTS ON CACHE BOOL
    "Disable use of classes in HIP allocators etc")

if (HAMR_ENABLE_HIP)
    # to get the GPU_TARGETS use: rocminfo | grep gfx
    set(GPU_TARGETS "gfx1030" CACHE STRING "GPU targets to compile for.")
    list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)
    find_package(hip REQUIRED)
    message(STATUS "HAMR: HIP features -- enabled")
else()
    message(STATUS "HAMR: HIP features -- disabled")
endif()

# enable OpenMP
set(HAMR_ENABLE_OPENMP OFF CACHE BOOL
    "Enable features supporting the OpenMP memory model")

set(HAMR_OPENMP_OBJECTS ON CACHE BOOL
    "Disable use of classes in OpenMP allocators etc")

# For AMD this could be gfx1030, rocminfo | grep gfx. for Intel spir64
set(OPENMP_GPU_TARGETS "" CACHE STRING "OpenMP GPU targets to compile for.")
if (HAMR_ENABLE_OPENMP AND NOT OPENMP_GPU_TARGETS)
    message(FATAL_ERROR "OPENMP_GPU_TARGETS was not set. An OpenMP GPU target must be provided")
endif()

if (HAMR_ENABLE_OPENMP)
    message(STATUS "HAMR: OpenMP features -- enabled. Target ${OPENMP_GPU_TARGETS}")
else()
    message(STATUS "HAMR: OpenMP features -- disabled")
endif()


# add the requisite flags. CMake enthusiasts will say that this is "not the
# CMake way". However, CMake has spotty coverage, is inconsistent in
# mechanisms, and often it does not work. Nonetheless, one may override our
# settings here by specifying them on the command line.
if (NOT MSVC)
    # C++
    if (NOT CMAKE_CXX_FLAGS)
        set(tmp "-fPIC -std=c++17 -Wall -Wextra -fvisibility=hidden")
        if ((APPLE) AND ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang"))
            set(tmp "${tmp} -stdlib=libc++")
        endif()
        if ("${CMAKE_BUILD_TYPE}" MATCHES "Release")
            set(tmp "${tmp} -O3 -march=native -mtune=native")
        endif()

        if (HAMR_ENABLE_OPENMP)
            set(nvidia_omp_flags "-mp=gpu")
            set(clang_amd_omp_flags "-target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=${OPENMP_GPU_TARGETS}")
            set(clang_nvidia_omp_flags "-Xopenmp-target -march=sm_75 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda --cuda-path=${CUDA_DIR}")
	    set(intel_icpx_omp_flags "-qopenmp -fopenmp-targets=${OPENMP_GPU_TARGETS}")
            set(tmp_loop "distribute parallel for")
            check_cxx_compiler_flag(${nvidia_omp_flags} nvidia_omp)
            check_cxx_compiler_flag(${clang_amd_omp_flags} clang_amd_omp)
            check_cxx_compiler_flag(${clang_nvidia_omp_flags} clang_nvidia_omp)
            check_cxx_compiler_flag(${intel_icpx_omp_flags} intel_omp)
            if (nvidia_omp)
                set(tmp "${tmp} ${nvidia_omp_flags}")
                set(tmp_loop "loop")
            elseif (clang_amd_omp)
                set(tmp "${tmp} ${clang_amd_omp_flags}")
            elseif (clang_nvidia_omp)
                set(tmp "${tmp} ${clang_nvidia_omp_flags}")
	    elseif (intel_omp)
		set(tmp "${tmp} ${intel_icpx_omp_flags}")
            else()
                message(FATAL_ERROR "A viable OpenMP compiler was not found")
            endif()
            set(HAMR_OPENMP_LOOP ${tmp_loop} CACHE STRING "OpenMP looping construct to use for device off load")
        endif()

        set(CMAKE_CXX_FLAGS "${tmp}"
            CACHE STRING "HAMR CXX compiler defaults"
            FORCE)

        string(REGEX REPLACE "-O[0-9]" "-O3" tmp "${CMAKE_CXX_FLAGS_RELEASE}")
        set(CMAKE_CXX_FLAGS_RELEASE "${tmp}"
            CACHE STRING "HAMR CXX compiler defaults"
            FORCE)
    endif()

    # CUDA
    if (NOT CMAKE_CUDA_FLAGS)
        set(tmp "--default-stream per-thread --expt-relaxed-constexpr")
        if ("${CMAKE_BUILD_TYPE}" MATCHES "Release")
            set(tmp "${tmp} -Xcompiler -Wall,-Wextra,-O3,-march=native,-mtune=native,-fvisibility=hidden")
        elseif ("${CMAKE_BUILD_TYPE}" MATCHES "Debug")
            set(tmp "${tmp} -g -G -Xcompiler -Wall,-Wextra,-O0,-g,-fvisibility=hidden")
        endif()

        set(CMAKE_CUDA_FLAGS "${tmp}"
            CACHE STRING "HAMR CUDA compiler defaults"
            FORCE)

        string(REGEX REPLACE "-O[0-9]" "-O3" tmp "${CMAKE_CUDA_FLAGS_RELEASE}")
        set(CMAKE_CUDA_FLAGS_RELEASE "${tmp}"
            CACHE STRING "HAMR CUDA compiler defaults"
            FORCE)
    endif()
endif()

# set build/install sub dirs for various components
if (NOT LIB_PREFIX)
  set(LIB_PREFIX lib)
endif()
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${LIB_PREFIX})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${LIB_PREFIX})
if (NOT BIN_PREFIX)
  set(BIN_PREFIX bin)
endif()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${BIN_PREFIX})
if (NOT INCLUDE_PREFIX)
  set(INCLUDE_PREFIX include)
endif()

# C++ sources
set(hamr_sources
    #hamr_cpu_memory_resource.cxx
    #hamr_cuda_uva_memory_resource.cxx
    hamr_buffer_allocator.cxx
    hamr_env.cxx
    hamr_stream.cxx
    )

# CUDA sources
if (HAMR_ENABLE_CUDA)
    list(APPEND hamr_libs CUDA::cudart)
    list(APPEND hamr_sources
        #hamr_cuda_uva_memory_resource.cu
        hamr_cuda_device.cu
        hamr_cuda_launch.cu
        )
endif()

# HIP sources
if (HAMR_ENABLE_HIP)
    list(APPEND hamr_libs hip::device)
    list(APPEND hamr_sources
        hamr_hip_device.cxx
        hamr_hip_launch.cxx
        )
endif()

# OpenMP sources
if (HAMR_ENABLE_OPENMP)
    list(APPEND hamr_sources
        hamr_openmp_device.cxx
        )
endif()

# hamr_config.h
configure_file(hamr_config.h.in hamr_config.h)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hamr_config.h DESTINATION include)

# hamr_config.cmake
configure_file(hamr_config.cmake.in
    ${LIB_PREFIX}/cmake/hamr_config.cmake @ONLY)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${LIB_PREFIX}/cmake/hamr_config.cmake
  DESTINATION ${LIB_PREFIX}/cmake)

# add the libaraary
add_library(hamr ${hamr_sources})
target_link_libraries(hamr PUBLIC ${hamr_libs})
if (HAMR_ENABLE_CUDA)
    set_target_properties(hamr PROPERTIES CUDA_ARCHITECTURES "${HAMR_CUDA_ARCHITECTURES}")
endif()

target_include_directories(hamr
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    $<INSTALL_INTERFACE:${INCLUDE_PREFIX}>
    )

file(GLOB hamr_headers LIST_DIRECTORIES FALSE
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.h")

file(GLOB hamr_interfaces LIST_DIRECTORIES FALSE
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.i")

install(FILES ${hamr_headers} ${hamr_interfaces}
    DESTINATION ${INCLUDE_PREFIX})

install(TARGETS hamr
    EXPORT hamr
    INCLUDES DESTINATION ${INCLUDE_PREFIX}
    ARCHIVE DESTINATION ${LIB_PREFIX}
    LIBRARY DESTINATION ${LIB_PREFIX}
    )

install(EXPORT hamr
    DESTINATION ${LIB_PREFIX}
    FILE hamr.cmake
    EXPORT_LINK_INTERFACE_LIBRARIES
    )

# enable Python bindings
find_package(Python3 COMPONENTS Interpreter Development)
if (HAMR_ENABLE_PYTHON AND Python3_FOUND)
    message(STATUS "HAMR: Python features -- enabled")
    add_subdirectory(python)
else()
    message(STATUS "HAMR: Python features -- disabled")
endif()

# enable regression tests
set(BUILD_TESTING OFF CACHE BOOL "Enable regression tests")
if (BUILD_TESTING)
    include(CTest)
    add_subdirectory(test)
endif()
