cmake_policy(SET CMP0078 NEW)
cmake_policy(SET CMP0086 NEW)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
message("++ Executable: ${Python3_EXECUTABLE}")

find_package(SWIG COMPONENTS python)
include(UseSWIG)

set(swig_deps
    ../hamr_buffer_allocator.i
    ../hamr_buffer_handle.i
    ../hamr_buffer.i)

set_property(SOURCE hamr_py.i PROPERTY CPLUSPLUS ON)
set_property(SOURCE hamr_py.i PROPERTY DEPENDS ${swig_deps})
set_property(SOURCE hamr_py.i PROPERTY SWIG_MODULE_NAME hamr_py)

swig_add_library(hamr_py
    TYPE MODULE LANGUAGE python
    SOURCES hamr_py.i
    OUTPUT_DIR "${CMAKE_BINARY_DIR}/lib"
    OUTFILE_DIR "${CMAKE_CURRENT_BINARY_DIR}")

target_link_libraries(hamr_py ${Python3_LIBRARIES} hamr)

target_include_directories(hamr_py
    PRIVATE "${Python3_INCLUDE_DIRS}"
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
    "${CMAKE_CURRENBT_BINARY_DIR}"
    "${CMAKE_CURRENBT_BINARY_DIR}/.."
    )

set_property(TARGET hamr_py
    PROPERTY SWIG_USE_TARGET_INCLUDE_DIRECTORIES ON)

if (HAMR_ENABLE_CUDA)
    set_source_files_properties(
        "${CMAKE_CURRENT_BINARY_DIR}/hamr_pyPYTHON_wrap.cxx"
        PROPERTIES LANGUAGE CUDA)

    set_target_properties(hamr_py PROPERTIES
        CUDA_ARCHITECTURES "${HAMR_CUDA_ARCHITECTURES}")
endif()

configure_file(hamr.py
    ${CMAKE_CURRENT_BINARY_DIR}/../${LIB_PREFIX}/hamr.py
    COPYONLY)

install(TARGETS hamr_py LIBRARY DESTINATION ${LIB_PREFIX})
install(FILES hamr.py DESTINATION ${LIB_PREFIX})
