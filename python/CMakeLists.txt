cmake_policy(SET CMP0078 NEW)
cmake_policy(SET CMP0086 NEW)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
message("++ Executable: ${Python3_EXECUTABLE}")

set(HAMR_PYTHON_SITE
  "${CMAKE_INSTALL_LIBDIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages/"
  CACHE STRING "Where Python modules are compiled and installed.")

set(HAMR_PYTHON_DIR "${HAMR_PYTHON_SITE}/hamr/"
  CACHE STRING "Where HAMR Python bindings are compiled and installed")

message(STATUS "HAMR: Python modules will be installed at \"${HAMR_PYTHON_DIR}\"")

find_package(SWIG COMPONENTS python)
include(UseSWIG)

set(swig_deps
    ../hamr_buffer_allocator.i
    ../hamr_buffer_handle.i
    ../hamr_buffer.i)

set_property(SOURCE hamr_py.i PROPERTY CPLUSPLUS ON)
set_property(SOURCE hamr_py.i PROPERTY DEPENDS ${swig_deps})
set_property(SOURCE hamr_py.i PROPERTY SWIG_MODULE_NAME hamr_py)

list(APPEND hamr_py_sources hamr_py.i)
if (HAMR_SEPARATE_IMPL)
    list(APPEND hamr_py_sources
        ../hamr_python_deleter.cxx
        )
endif()

swig_add_library(hamr_py
    TYPE MODULE LANGUAGE python
    SOURCES ${hamr_py_sources}
    OUTPUT_DIR "${CMAKE_BINARY_DIR}/${HAMR_PYTHON_DIR}"
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

set_target_properties(hamr_py PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${HAMR_PYTHON_DIR}")

if (HAMR_ENABLE_CUDA AND NOT HAMR_NVHPC_CUDA)
    set_source_files_properties(
        "${CMAKE_CURRENT_BINARY_DIR}/hamr_pyPYTHON_wrap.cxx"
        PROPERTIES LANGUAGE CUDA)

    set_target_properties(hamr_py PROPERTIES
        CUDA_ARCHITECTURES "${HAMR_CUDA_ARCHITECTURES}")
endif()

install(TARGETS hamr_py DESTINATION ${HAMR_PYTHON_DIR})
install(FILES ${CMAKE_BINARY_DIR}/${HAMR_PYTHON_DIR}/hamr_py.py
  DESTINATION ${HAMR_PYTHON_DIR})

if (APPLE)
  set_target_properties(hamr_py PROPERTIES INSTALL_RPATH "@loader_path/./")
elseif(UNIX)
  set_target_properties(hamr_py PROPERTIES INSTALL_RPATH "\$ORIGIN/")
endif()

configure_file(hamr.py "${CMAKE_BINARY_DIR}/${HAMR_PYTHON_DIR}/__init__.py"
  COPYONLY)

install(FILES  "${CMAKE_BINARY_DIR}/${HAMR_PYTHON_DIR}/__init__.py"
  DESTINATION "${HAMR_PYTHON_DIR}")

# capture python path for use in automated CI
file(CONFIGURE OUTPUT "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/hamr_python_env.sh"
CONTENT
[=[
#!/bin/bash
export PYTHONPATH=@CMAKE_BINARY_DIR@/@HAMR_PYTHON_SITE@:$PYTHONPATH
]=]
@ONLY)
