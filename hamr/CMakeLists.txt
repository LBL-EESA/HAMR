cmake_minimum_required(VERSION 3.0)
cmake_policy(SET CMP0063 NEW)
project(hamr)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_C_VISIBILITY_PRESET hidden)

# if we forgot to set the build type default to release
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release"
      CACHE STRING "One of: Debug Release RelWithDebInfo MinSizeRel"
      FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY
        STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()
message(STATUS "Configuring a ${CMAKE_BUILD_TYPE} build")

# enable CUDA
if (NOT DEFINED HAMR_ENABLE_CUDA)
    set(HAMR_ENABLE_CUDA ON CACHE BOOL
        "Enable features supporting the CUDA memory model")
endif()
if (HAMR_ENABLE_CUDA)
    include(CheckLanguage)
    set(CMAKE_CUDA_STANDARD 17)
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
    hamr_env.cxx
    )

# CUDA sources
if (HAMR_ENABLE_CUDA)
    list(APPEND hamr_sources
        #hamr_cuda_uva_memory_resource.cu
        hamr_cuda_launch.cu
        )
endif()

set(HAMR_CUDA_OBJECTS ON CACHE BOOL
    "Disable use of classes in CUDA allocators etc")

# add the libaraary
add_library(hamr ${hamr_sources})
target_link_libraries(hamr PUBLIC ${hamr_libs})

# hamr_config.h
configure_file(hamr_config.h.in hamr_config.h)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
install(FILES ${CMAKE_BINARY_DIR}/hamr_config.h DESTINATION include)

# hamr_config.cmake
configure_file(hamr_config.cmake.in
    ${LIB_PREFIX}/cmake/hamr_config.cmake @ONLY)

install(FILES ${CMAKE_BINARY_DIR}/${LIB_PREFIX}/cmake/hamr_config.cmake
  DESTINATION ${LIB_PREFIX}/cmake)

target_include_directories(hamr
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    $<INSTALL_INTERFACE:${INCLUDE_PREFIX}>
    )

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DESTINATION ${INCLUDE_PREFIX}
    FILES_MATCHING PATTERN "*.h")

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

# enable regression tests
set(BUILD_TESTING OFF CACHE BOOL "Enable regression tests")
if (BUILD_TESTING)
    include(CTest)
    add_subdirectory(test)
endif()
