cmake_minimum_required(VERSION 3.0)
cmake_policy(SET CMP0063 NEW)
project(hamr)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_ARCHITECTURES 75)
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
message(STATUS "Configuring a ${CMAKE_BUILD_TYPE} build")

# add the requisite flags to enable c++11 features on various platforms.
if (MSVC)
    # TODO require vs2015
    # enable exceptions
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc"
            CACHE STRING "" FORCE)
else()
    if (NOT CMAKE_CXX_FLAGS)
        set(tmp "-fPIC -std=c++17 -Wall -Wextra")
        if ((APPLE) AND ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang"))
            set(tmp "${tmp} -stdlib=libc++")
        endif()
        if ("${CMAKE_BUILD_TYPE}" MATCHES "Release")
            set(tmp "${tmp} -O3 -march=native -mtune=native -fno-trapping-math -fno-math-errno")
            if (NOT "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
                set(tmp "${tmp} -fno-signaling-nans")
            endif()
        endif()
        set(CMAKE_CXX_FLAGS "${tmp}"
            CACHE STRING "HAMR build defaults"
            FORCE)
    endif()
    if (NOT CMAKE_Fortran_FLAGS)
        set(tmp "-Wall -Wextra -Wno-conversion -Wno-compare-reals -fno-math-errno")
        if (CMAKE_BUILD_TYPE STREQUAL "Release")
            set(tmp "${tmp} -march=native -mtune=native -fno-trapping-math -fno-signaling-nans -fno-math-errno")
        endif()
        set(CMAKE_Fortran_FLAGS "${tmp}"
            CACHE STRING "HAMR build defaults"
            FORCE)
    endif()
    if (NOT CMAKE_CUDA_FLAGS)
        set(tmp "--default-stream per-thread --expt-relaxed-constexpr")
        if ("${CMAKE_BUILD_TYPE}" MATCHES "Release")
            set(tmp "${tmp} -Xcompiler -Wall,-Wextra,-O3,-march=native,-mtune=native,-fno-trapping-math,-fno-math-errno")
            if (NOT "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
                set(tmp "${tmp},-fno-signaling-nans")
            endif()
        elseif ("${CMAKE_BUILD_TYPE}" MATCHES "Debug")
            set(tmp "${tmp} -g -G -Xcompiler -Wall,-Wextra,-O0,-g")
        endif()
        set(CMAKE_CUDA_FLAGS "${tmp}"
            CACHE STRING "HAMR build defaults"
            FORCE)
        string(REGEX REPLACE "-O[0-9]" "-O3" tmp "${CMAKE_CXX_FLAGS_RELEASE}")
        set(CMAKE_CXX_FLAGS_RELEASE "${tmp}"
            CACHE STRING "HAMR build defaults"
            FORCE)
        string(REGEX REPLACE "-O[0-9]" "-O3" tmp "${CMAKE_CUDA_FLAGS_RELEASE}")
        set(CMAKE_CUDA_FLAGS_RELEASE "${tmp}"
            CACHE STRING "HAMR build defaults"
            FORCE)
    endif()
endif()

# enable Python
set(HAMR_ENABLE_PYTHON OFF CACHE BOOL "Enable Python bindings")

# enable CUDA
set(HAMR_ENABLE_CUDA OFF CACHE BOOL
    "Enable features supporting the CUDA memory model")

set(HAMR_CUDA_OBJECTS ON CACHE BOOL
    "Disable use of classes in CUDA allocators etc")

set(HAMR_VERBOSE OFF CACHE BOOL
    "Enable debug messages when HAMR_VERBOSE is found in the environment")

if (HAMR_ENABLE_CUDA)
    include(CheckLanguage)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES 75)
    set(CMAKE_CUDA_VISIBILITY_PRESET hidden)
    set(CMAKE_CUDA_ARCHITECTURES 75)
    check_language(CUDA)
    if (CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        message(STATUS "hamr CUDA features -- enabled")
        set(tmp ON)
    else()
        message(FATAL_ERROR "CUDA is required for hamr but was not found")
    endif()
else()
    message(STATUS "hamr CUDA features -- disabled")
endif()

# enable HIP
set(HAMR_ENABLE_HIP OFF CACHE BOOL
    "Enable features supporting the HIP memory model")

set(HAMR_HIP_OBJECTS ON CACHE BOOL
    "Disable use of classes in HIP allocators etc")

set(HAMR_VERBOSE OFF CACHE BOOL
    "Enable debug messages when HAMR_VERBOSE is found in the environment")

if (HAMR_ENABLE_HIP)
    # to get the GPU_TARGETS use: rocminfo | grep gfx
    list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)
    set(GPU_TARGETS "gfx1030" CACHE STRING "GPU targets to compile for.")
    find_package(hip REQUIRED)
    message(STATUS "hamr HIP features -- enabled")
else()
    message(STATUS "hamr HIP features -- disabled")
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
    )

# CUDA sources
if (HAMR_ENABLE_CUDA)
    list(APPEND hamr_sources
        #hamr_cuda_uva_memory_resource.cu
        hamr_cuda_device.cu
        hamr_cuda_launch.cu
        )
endif()

# HAMR sources
if (HAMR_ENABLE_HIP)
    list(APPEND hamr_libs hip::device)
    list(APPEND hamr_sources
        hamr_hip_device.cxx
        hamr_hip_launch.cxx
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
    message(STATUS "hamr Python features -- enabled")
    add_subdirectory(python)
else()
    message(STATUS "hamr Python features -- disabled")
endif()

# enable regression tests
set(BUILD_TESTING OFF CACHE BOOL "Enable regression tests")
if (BUILD_TESTING)
    include(CTest)
    add_subdirectory(test)
endif()
