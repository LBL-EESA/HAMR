# Get the OpenMP device offload flags for the current C++ compiler.
#
#   TARGET <offload target>
#       names the target for offloading (optional).
#
#   ARCH <target architecture>
#       names the architcure to compiler for (optional).
#
#   ADD_FLAGS <other flags>
#       additional flags that may be needed (optional).
#
#   RESULT <avr name>
#       the flags are stored in this variable.
#
function(get_offload_compile_flags)
    set(opts "")
    set(nvpo ARCH TARGET ADD_FLAGS RESULT)
    set(mvo)
    cmake_parse_arguments(PARSE_ARGV 0 OMP_DO "${opts}" "${nvpo}" "${mvo}")
    set(tmp)
    if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
        set(tmp "-fopenmp --offload-new-driver")
        if (OMP_DO_TARGET)
            set(tmp "${tmp} -fopenmp-targets=${OMP_DO_TARGET}")
        endif()
        if (OMP_DO_ARCH)
            set(tmp "${tmp} --offload-arch=${OMP_DO_ARCH}")
        endif()
    elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "IntelLLVM")
        set(tmp "-qopenmp")
        if (OMP_DO_TARGET)
            set(tmp "${tmp} -fopenmp-targets=${OMP_DO_TARGET}")
        endif()
    elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
        set(tmp "-fopenmp")
        if (OMP_DO_TARGET)
            set(tmp "${tmp} -foffload=${OMP_DO_TARGET}")
        endif()
        if (OMP_DO_ARCH)
            set(tmp "${tmp} --offload-options=${OMP_DO_TARGET}=-march=${OMP_DO_ARCH}")
        endif()
    elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "NVHPC")
        set(tmp "-mp=gpu")
        if (OMP_DO_ARCH)
            set(tmp "${tmp} -gpu=${OMP_DO_ARCH}")
        endif()
    endif()
    if (OMP_DO_ADD_FLAGS)
        set(tmp "${tmp} ${OMP_DO_ADD_FLAGS}")
    endif()
    if ("${tmp}" STREQUAL "")
        message(WARNING "OpenMP offload compiler flags not known for ${CMAKE_CXX_COMPILER_ID}")
    else()
        message(STATUS "OpenMP offload compiler flags for ${CMAKE_CXX_COMPILER_ID} are ${tmp}")
    endif()
    set(${OMP_DO_RESULT} ${tmp} PARENT_SCOPE)
endfunction()

# Get the OpenMP device offload flags for the current C++ compiler.
#
#   TARGET <offload target>
#       names the target for offloading (optional).
#
#   ARCH <target architecture>
#       names the architcure to compiler for (optional).
#
#   ADD_FLAGS <other flags>
#       additional flags that may be needed (optional).
#
#   RESULT <avr name>
#       the flags are stored in this variable.
#
function(get_offload_link_flags)
    set(opts "")
    set(nvpo ARCH TARGET ADD_FLAGS RESULT)
    set(mvo)
    cmake_parse_arguments(PARSE_ARGV 0 OMP_DO "${opts}" "${nvpo}" "${mvo}")
    set(tmp)
    if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
        list(APPEND tmp -fopenmp --offload-new-driver)
        if (OMP_DO_TARGET)
            list(APPEND tmp -fopenmp-targets=${OMP_DO_TARGET})
        endif()
        if (OMP_DO_ARCH)
            list(APPEND tmp --offload-arch=${OMP_DO_ARCH})
        endif()
    elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "IntelLLVM")
    elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
    elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "NVHPC")
    endif()
    if (OMP_DO_ADD_FLAGS)
        set(tmp "${tmp} ${OMP_DO_ADD_FLAGS}")
    endif()
    message(STATUS "OpenMP offload linker flags for ${CMAKE_CXX_COMPILER_ID} are ${tmp}")
    set(${OMP_DO_RESULT} ${tmp} PARENT_SCOPE)
endfunction()
