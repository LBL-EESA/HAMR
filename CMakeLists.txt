cmake_minimum_required(VERSION 3.22)
project(hamr)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(CheckCXXSourceCompiles)
include(CheckCXXCompilerFlag)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
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

set(tmp_nvcc FALSE)
set(tmp_clang FALSE)
set(tmp_nvhpc FALSE)
set(tmp_cuda_arch "60;70;75")
if (HAMR_ENABLE_CUDA)
    set(tmp_have_cuda FALSE)
    if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "NVHPC")
        set(tmp_have_cuda TRUE)
        set(tmp_cuda_arch "cc75")
        set(tmp_nvhpc TRUE)
    else()
        include(CheckLanguage)
        check_language(CUDA)
        if (CMAKE_CUDA_COMPILER)
            enable_language(CUDA)
            set(tmp_have_cuda TRUE)
            if ("${CMAKE_CUDA_COMPILER_ID}" MATCHES "Clang")
                set(tmp_cuda_arch "75")
                set(tmp_clang TRUE)
            elseif ("${CMAKE_CUDA_COMPILER_ID}" MATCHES "NVIDIA")
                set(tmp_nvcc TRUE)
            endif()
        endif()
    endif()
    find_package(CUDAToolkit REQUIRED)
endif()

set(HAMR_NVCC_CUDA ${tmp_nvcc} CACHE
    STRING "Internal: set if the CUDA compiler is nvcc")

set(HAMR_NVHPC_CUDA ${tmp_nvhpc} CACHE
    STRING "Internal: set if the CUDA compiler is nvc++")

set(HAMR_CLANG_CUDA ${tmp_clang} CACHE
    STRING "Internal: set if the CUDA compiler is clang++")

set(HAMR_CUDA_ARCHITECTURES ${tmp_cuda_arch} CACHE
    STRING "Compile for these CUDA virtual and real architectures")

if (HAMR_ENABLE_CUDA)
    if (tmp_have_cuda)
        message(STATUS "HAMR: CUDA features -- enabled (${CMAKE_CUDA_COMPILER_ID}:${HAMR_CUDA_ARCHITECTURES})")

        set(CMAKE_CUDA_ARCHITECTURES ${HAMR_CUDA_ARCHITECTURES})
    else()
        message(FATAL_ERROR "CUDA is required for hamr but was not found")
    endif()
else()
    message(STATUS "HAMR: CUDA features -- disabled")
endif()

# enable HIP
set(HAMR_ENABLE_HIP OFF CACHE BOOL
    "Enable features supporting the HIP memory model")

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

include(hamr_omp_offload)

# clang/AMD: target=amdgcn, arch=gfx1030.
# Intel: target=spir64.
# clang/NVIDIA: target=nvptx64 arch=sm_75.
# gcc/NVIDIA: target=nvptx-none arch=sm_75.
# NVIDIA HPC: arch=cc75
set(HAMR_OPENMP_TARGET "" CACHE STRING "OpenMP targets to compile for.")
set(HAMR_OPENMP_ARCH "" CACHE STRING "OpenMP architectures to compile for.")
set(HAMR_OPENMP_FLAGS "" CACHE STRING "Additional flags for OpenMP offlaod.")

if (HAMR_ENABLE_OPENMP)
    message(STATUS "HAMR: OpenMP features -- enabled (target=${HAMR_OPENMP_TARGET} arch=${HAMR_OPENMP_ARCH})")
else()
    message(STATUS "HAMR: OpenMP features -- disabled")
endif()

# separate implementations
set(tmp OFF)
if (HAMR_ENABLE_OPENMP AND HAMR_ENABLE_CUDA AND NOT HAMR_NVHPC_CUDA)
    set(tmp ON)
endif()
set(HAMR_SEPARATE_IMPL ${tmp} CACHE BOOL
  "Compile to a library with explicit instantiatons for POD types")
if (HAMR_SEPARATE_IMPL)
    message(STATUS "HAMR: Separate implementations -- enabled")
else()
    message(STATUS "HAMR: Separate implementations -- disabled")
endif()

# objects
set(tmp ON)
if (HAMR_SEPRATE_IMPL)
    set(tmp OFF)
endif()
set(HAMR_ENABLE_OBJECTS ${tmp} CACHE BOOL
    "Disable use of user defined classes")
if (HAMR_ENABLE_OBJECTS)
    message(STATUS "HAMR: User defined objects -- enabled")
else()
    message(STATUS "HAMR: User defined objects -- disabled")
endif()


# add the requisite flags. CMake enthusiasts will tell you that this is "not
# the CMake way". However, CMake has spotty coverage, is inconsistent in
# mechanisms, and often it does not work. Nonetheless, one may override our
# settings here by specifying them on the command line.
#
# Current issues:
# * CMake does not propagate -fvisibility=hidden during cuda
#   linking and this leads to a nasty runtime crash when stataic and shared
#   libraries containing cuda code are linked into a fat bin.
#   Filed a bug report w/ NVIDIA and this has been reported to be fixed (Q4 2022)
# * CMake does not handle CUDA w/ nvc++ as the host compiler. (Q1 2023)
# * On some systems CMake will use O2 for release builds rather than O3
# * CMake currently does not detect OpenMP offload flags at all. There is a
#   a CMake bug report about this. (Q1 2023)
if (NOT MSVC)
    # C++
    if (NOT CMAKE_CXX_FLAGS)
        set(tmp "-fPIC -std=c++17 -Wall -Wextra -fvisibility=hidden")

        # this was needed in the early days of C++11 support
        #if ((APPLE) AND ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang"))
        #    set(tmp "${tmp} -stdlib=libc++")
        #endif()

        if (HAMR_NVHPC_CUDA)
            set(tmp "${tmp} -cuda -gpu=${HAMR_CUDA_ARCHITECTURES}")
        endif()

        if ("${CMAKE_BUILD_TYPE}" MATCHES "Release")
            set(tmp "${tmp} -O3 -march=native -mtune=native")
        endif()

        if (HAMR_ENABLE_OPENMP)
            set(tmp_flags)

            get_offload_compile_flags( TARGET ${HAMR_OPENMP_TARGET}
                ARCH ${HAMR_OPENMP_ARCH} ADD_FLAGS ${HAMR_OPENMP_FLAGS}
                RESULT tmp_flags)

            set(tmp "${tmp} ${tmp_flags}")

            set(tmp_loop "distribute parallel for")
            if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "NVHPC")
                set(tmp_loop "loop")
            endif()

            set(HAMR_OPENMP_LOOP ${tmp_loop} CACHE
                STRING "OpenMP looping construct to use for device off load")
        endif()

        set(CMAKE_CXX_FLAGS "${tmp}"
            CACHE STRING "HAMR CXX compiler defaults"
            FORCE)

        string(REGEX REPLACE "-O[0-9]" "-O3" tmp "${CMAKE_CXX_FLAGS_RELEASE}")
        set(CMAKE_CXX_FLAGS_RELEASE "${tmp}"
            CACHE STRING "HAMR CXX compiler defaults"
            FORCE)
    endif()

    set(tmp)
    if (HAMR_ENABLE_OPENMP)
        get_offload_link_flags( TARGET ${HAMR_OPENMP_TARGET}
            ARCH ${HAMR_OPENMP_ARCH} ADD_FLAGS ${HAMR_OPENMP_FLAGS}
            RESULT tmp)
    endif()

    set(HAMR_OPENMP_LINK_FLAGS ${tmp}
        CACHE STRING "HAMR linker flags for OpenMP")

    # CUDA
    if (NOT CMAKE_CUDA_FLAGS AND HAMR_NVCC_CUDA)
        set(tmp "--default-stream per-thread --expt-relaxed-constexpr -ccbin=${CMAKE_CXX_COMPILER}")
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
include(GNUInstallDirs)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}")

# C++ sources
set(hamr_sources
    #hamr_host_memory_resource.cxx
    #hamr_cuda_uva_memory_resource.cxx
    hamr_buffer_allocator.cxx
    hamr_env.cxx
    hamr_stream.cxx
    )

# CUDA sources
if (HAMR_ENABLE_CUDA)
    set(hamr_cu_sources
        #hamr_cuda_uva_memory_resource.cxx
        hamr_cuda_device.cxx
        hamr_cuda_launch.cxx
        )
    if (HAMR_ENABLE_CUDA AND NOT HAMR_NVHPC_CUDA)
        set_source_files_properties(${hamr_cu_sources} PROPERTIES LANGUAGE CUDA)
    endif()
    list(APPEND hamr_libs CUDA::cudart)
    list(APPEND hamr_sources ${hamr_cu_sources})
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

# explicit instantations
if (HAMR_SEPARATE_IMPL)
    list(APPEND hamr_sources
        hamr_malloc_allocator.cxx
        hamr_new_allocator.cxx
        hamr_host_copy.cxx
        hamr_buffer.cxx
        )
    if (HAMR_ENABLE_CUDA)
        set(tmp_sources
            hamr_cuda_malloc_allocator.cxx
            hamr_cuda_malloc_uva_allocator.cxx
            hamr_cuda_malloc_async_allocator.cxx
            hamr_cuda_malloc_host_allocator.cxx
            hamr_cuda_copy_async.cxx
            hamr_cuda_print.cxx
            )
        if (NOT HAMR_NVHPC_CUDA)
            set_source_files_properties(${tmp_sources} PROPERTIES LANGUAGE CUDA)
        endif()
        list(APPEND hamr_sources ${tmp_sources})
    endif()
    if (HAMR_ENABLE_HIP)
        set(tmp_sources
            hamr_hip_malloc_allocator.cxx
            #hamr_hip_malloc_uva_allocator.cxx
            #hamr_hip_malloc_async_allocator.cxx
            #hamr_hip_malloc_host_allocator.cxx
            hamr_hip_copy.cxx
            hamr_hip_print.cxx
            )
        list(APPEND hamr_sources ${tmp_sources})
    endif()
    if (HAMR_ENABLE_OPENMP)
        list(APPEND hamr_sources
            hamr_openmp_allocator.cxx
            hamr_openmp_copy.cxx
            hamr_openmp_print.cxx
            )
    endif()
endif()

# hamr_config.h
configure_file(hamr_config.h.in hamr_config.h)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hamr_config.h DESTINATION include)

# hamr_config.cmake
configure_file(hamr_config.cmake.in
    ${CMAKE_INSTALL_LIBDIR}/cmake/hamr/hamr-config.cmake @ONLY)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/hamr/hamr-config.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hamr)

# add the libarary
set(tmp STATIC)
if (HAMR_SEPARATE_IMPL)
    set(tmp SHARED) # because nvlink segv's during device link w/ clang
endif()
add_library(hamr ${tmp} ${hamr_sources})
target_link_libraries(hamr PUBLIC ${hamr_libs})

if (HAMR_ENABLE_CUDA AND HAMR_NVCC_CUDA)
    set_target_properties(hamr PROPERTIES CUDA_ARCHITECTURES "${HAMR_CUDA_ARCHITECTURES}")
endif()

if (HAMR_ENABLE_OPENMP AND HAMR_ENABLE_CUDA AND NOT HAMR_NVHPC_CUDA)
    target_link_options(hamr PUBLIC ${HAMR_OPENMP_LINK_FLAGS})
endif ()

target_include_directories(hamr
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )

file(GLOB hamr_headers LIST_DIRECTORIES FALSE
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.h")

file(GLOB hamr_interfaces LIST_DIRECTORIES FALSE
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.i")

install(FILES ${hamr_headers} ${hamr_interfaces}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS hamr
    EXPORT hamr
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )

install(EXPORT hamr
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hamr
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
