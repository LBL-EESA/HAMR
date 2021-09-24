project(hamm)
cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_STANDARD 17)

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
if (NOT DEFINED HAMM_ENABLE_CUDA)
set(HAMM_ENABLE_CUDA ON CACHE BOOL "Enable features supporting the CUDA memory model")
endif()
if (HAMM_ENABLE_CUDA)
    include(CheckLanguage)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75)
    endif()
    check_language(CUDA)
    if (CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        message(STATUS "hamm CUDA features -- enabled")
        set(tmp ON)
    else()
        message(FATAL_ERROR "CUDA is required for hamm but was not found")
    endif()
else()
    message(STATUS "hamm CUDA features -- disabled")
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
set(hamm_sources
    hamm_cpu_memory_resource.cxx
    hamm_cuda_mm_memory_resource.cxx
    )

# CUDA sources
if (HAMM_ENABLE_CUDA)
    list(APPEND hamm_sources
        hamm_cuda_mm_memory_resource.cu
        )
endif()

# add the libaraary
add_library(hamm ${hamm_sources})
target_link_libraries(hamm PUBLIC ${hamm_libs})

target_compile_features(hamm PUBLIC cxx_std_17)
set_target_properties(hamm PROPERTIES POSITION_INDEPENDENT_CODE ON)

if (HAMM_ENABLE_CUDA)
  set_target_properties(hamm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif ()

# hamm_config.h
configure_file(hamm_config.h.in hamm_config.h)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
install(FILES ${CMAKE_BINARY_DIR}/hamm_config.h DESTINATION include)

# hamm_config.cmake
configure_file(hamm_config.cmake.in
    ${LIB_PREFIX}/cmake/hamm_config.cmake @ONLY)

install(FILES ${CMAKE_BINARY_DIR}/${LIB_PREFIX}/cmake/hamm_config.cmake
  DESTINATION ${LIB_PREFIX}/cmake)

target_include_directories(hamm
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    $<INSTALL_INTERFACE:${INCLUDE_PREFIX}>
    )

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DESTINATION ${INCLUDE_PREFIX}
    FILES_MATCHING PATTERN "*.h")

install(TARGETS hamm
    EXPORT hamm
    INCLUDES DESTINATION ${INCLUDE_PREFIX}
    ARCHIVE DESTINATION ${LIB_PREFIX}
    LIBRARY DESTINATION ${LIB_PREFIX}
    )

install(EXPORT hamm
    DESTINATION ${LIB_PREFIX}
    FILE hamm.cmake
    EXPORT_LINK_INTERFACE_LIBRARIES
    )

