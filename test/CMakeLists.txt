
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/test)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    $<TARGET_PROPERTY:hamr,INTERFACE_INCLUDE_DIRECTORIES>
    )

if (HAMR_ENABLE_CUDA)
    if (NOT HAMR_NVHPC_CUDA)
        set_source_files_properties(test_hamr_pipeline_cuda.cpp PROPERTIES LANGUAGE CUDA)
    endif()
    add_executable(test_hamr_pipeline_cuda test_hamr_pipeline_cuda.cpp)
    target_link_libraries(test_hamr_pipeline_cuda hamr)
    set_target_properties(test_hamr_pipeline_cuda PROPERTIES CUDA_ARCHITECTURES "${HAMR_CUDA_ARCHITECTURES}")
    add_test(NAME test_hamr_pipeline_cuda COMMAND test_hamr_pipeline_cuda)
    set_tests_properties(test_hamr_pipeline_cuda PROPERTIES ENVIRONMENT HAMR_VERBOSE=1)

    if (NOT HAMR_NVHPC_CUDA)
        set_source_files_properties(test_hamr_stream_cuda.cpp PROPERTIES LANGUAGE CUDA)
    endif()
    add_executable(test_hamr_stream_cuda test_hamr_stream_cuda.cpp)
    target_link_libraries(test_hamr_stream_cuda hamr CUDA::cublas)
    if (NOT HAMR_NVHPC_CUDA)
        set_target_properties(test_hamr_stream_cuda PROPERTIES CUDA_ARCHITECTURES "${HAMR_CUDA_ARCHITECTURES}")
    endif()
    add_test(NAME test_hamr_stream_cuda_async COMMAND test_hamr_stream_cuda 5000 async)
    set_tests_properties(test_hamr_stream_cuda_async PROPERTIES ENVIRONMENT HAMR_VERBOSE=1)
    add_test(NAME test_hamr_stream_cuda_default COMMAND test_hamr_stream_cuda 5000 default)
    set_tests_properties(test_hamr_stream_cuda_default PROPERTIES ENVIRONMENT HAMR_VERBOSE=1)
endif()

if (HAMR_ENABLE_HIP)
    add_executable(test_hamr_pipeline_hip test_hamr_pipeline_hip.cpp)
    target_link_libraries(test_hamr_pipeline_hip hamr)
    add_test(NAME test_hamr_pipeline_hip COMMAND test_hamr_pipeline_hip)
    set_tests_properties(test_hamr_pipeline_hip PROPERTIES ENVIRONMENT HAMR_VERBOSE=1)
endif()

if (HAMR_ENABLE_OPENMP)
    add_executable(test_hamr_pipeline_openmp test_hamr_pipeline_openmp.cpp)
    if (HAMR_ENABLE_CUDA AND NOT HAMR_NVHPC_CUDA)
        set_target_properties(test_hamr_pipeline_openmp PROPERTIES CUDA_ARCHITECTURES "${HAMR_CUDA_ARCHITECTURES}")
    endif()
    target_link_libraries(test_hamr_pipeline_openmp hamr)
    add_test(NAME test_hamr_pipeline_openmp COMMAND test_hamr_pipeline_openmp)
    set_tests_properties(test_hamr_pipeline_openmp PROPERTIES ENVIRONMENT HAMR_VERBOSE=1)
endif()

if (HAMR_ENABLE_OPENMP AND HAMR_ENABLE_CUDA AND HAMR_SEPARATE_IMPL)
    if (NOT HAMR_NVHPC_CUDA)
       set_source_files_properties(test_hamr_pipeline_cuda_openmp_cu.cpp PROPERTIES LANGUAGE CUDA)
    endif()
    add_executable(test_hamr_pipeline_cuda_openmp test_hamr_pipeline_cuda_openmp.cpp test_hamr_pipeline_cuda_openmp_cu.cpp test_hamr_pipeline_cuda_openmp_mp.cpp)
    if (NOT HAMR_NVHPC_CUDA)
        set_target_properties(test_hamr_pipeline_cuda_openmp PROPERTIES CUDA_ARCHITECTURES "${HAMR_CUDA_ARCHITECTURES}")
    endif()
    target_link_libraries(test_hamr_pipeline_cuda_openmp hamr)
    add_test(NAME test_hamr_pipeline_cuda_openmp COMMAND test_hamr_pipeline_cuda_openmp)
    set_tests_properties(test_hamr_pipeline_cuda_openmp PROPERTIES ENVIRONMENT HAMR_VERBOSE=1)
endif()


if (HAMR_ENABLE_CUDA AND NOT HAMR_NVHPC_CUDA AND NOT HAMR_SEPARATE_IMPL)
    set_source_files_properties(test_hamr_pipeline_host.cpp PROPERTIES LANGUAGE CUDA)
endif()
add_executable(test_hamr_pipeline_host test_hamr_pipeline_host.cpp)
if (HAMR_ENABLE_CUDA AND NOT HAMR_NVHPC_CUDA)
    set_target_properties(test_hamr_pipeline_host PROPERTIES CUDA_ARCHITECTURES "${HAMR_CUDA_ARCHITECTURES}")
endif()
target_link_libraries(test_hamr_pipeline_host hamr)
add_test(NAME test_hamr_pipeline_host COMMAND test_hamr_pipeline_host)
set_tests_properties(test_hamr_pipeline_host PROPERTIES ENVIRONMENT HAMR_VERBOSE=1)

if (HAMR_ENABLE_CUDA)
    if (NOT HAMR_NVHPC_CUDA AND NOT HAMR_SEPARATE_IMPL)
       set_source_files_properties(test_hamr_multi_gpu_cuda.cpp PROPERTIES LANGUAGE CUDA)
    endif()
    add_executable(test_hamr_multi_gpu_cuda test_hamr_multi_gpu_cuda.cpp)
    target_link_libraries(test_hamr_multi_gpu_cuda hamr)
    if (NOT HAMR_NVHPC_CUDA)
        set_target_properties(test_hamr_multi_gpu_cuda PROPERTIES CUDA_ARCHITECTURES "${HAMR_CUDA_ARCHITECTURES}")
    endif()
    add_test(NAME test_hamr_multi_gpu_cuda COMMAND test_hamr_multi_gpu_cuda)
    set_tests_properties(test_hamr_multi_gpu_cuda PROPERTIES ENVIRONMENT HAMR_VERBOSE=1)
endif()

if (HAMR_ENABLE_HIP)
    add_executable(test_hamr_multi_gpu_hip test_hamr_multi_gpu_hip.cpp)
    target_link_libraries(test_hamr_multi_gpu_hip hamr)
    add_test(NAME test_hamr_multi_gpu_hip COMMAND test_hamr_multi_gpu_hip)
    set_tests_properties(test_hamr_multi_gpu_hip PROPERTIES ENVIRONMENT HAMR_VERBOSE=1)
endif()

if (HAMR_ENABLE_OPENMP)
    add_executable(test_hamr_openmp_allocator test_hamr_openmp_allocator.cpp)
    if (HAMR_ENABLE_CUDA AND NOT HAMR_NVHPC_CUDA)
        set_target_properties(test_hamr_openmp_allocator PROPERTIES CUDA_ARCHITECTURES "${HAMR_CUDA_ARCHITECTURES}")
    endif()
    target_link_libraries(test_hamr_openmp_allocator hamr)
    add_test(NAME test_hamr_openmp_allocator COMMAND test_hamr_openmp_allocator)
    set_tests_properties(test_hamr_openmp_allocator PROPERTIES ENVIRONMENT HAMR_VERBOSE=1)
endif()

if (HAMR_ENABLE_PYTHON)

    add_test(NAME test_hamr_buffer_numpy_host
        COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/test_hamr_buffer_numpy_host.py)

    if (HAMR_ENABLE_CUDA)

        add_test(NAME test_hamr_buffer_numpy_cuda
            COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/test_hamr_buffer_numpy_cuda.py)

        add_test(NAME test_hamr_buffer_cupy_host
            COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/test_hamr_buffer_cupy_host.py)

        add_test(NAME test_hamr_buffer_cupy_cuda
            COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/test_hamr_buffer_cupy_cuda.py)

    endif()

endif()
