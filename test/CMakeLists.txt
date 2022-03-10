
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/test)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    $<TARGET_PROPERTY:hamr,INTERFACE_INCLUDE_DIRECTORIES>
    )

if (HAMR_ENABLE_CUDA)
    set_source_files_properties(test_hamr_pipeline_cuda.cpp PROPERTIES LANGUAGE CUDA)
    add_executable(test_hamr_pipeline_cuda test_hamr_pipeline_cuda.cpp)
    target_link_libraries(test_hamr_pipeline_cuda hamr)
    add_test(NAME test_hamr_pipeline_cuda COMMAND test_hamr_pipeline_cuda)
    set_tests_properties(test_hamr_pipeline_cuda PROPERTIES ENVIRONMENT HAMR_VERBOSE=1)
endif()

if (HAMR_ENABLE_HIP)
    add_executable(test_hamr_pipeline_hip test_hamr_pipeline_hip.cpp)
    target_link_libraries(test_hamr_pipeline_hip hamr)
    add_test(NAME test_hamr_pipeline_hip COMMAND test_hamr_pipeline_hip)
    set_tests_properties(test_hamr_pipeline_hip PROPERTIES ENVIRONMENT HAMR_VERBOSE=1)
endif()

if (HAMR_ENABLE_CUDA)
    set_source_files_properties(test_hamr_pipeline_cpu.cpp PROPERTIES LANGUAGE CUDA)
endif()
add_executable(test_hamr_pipeline_cpu test_hamr_pipeline_cpu.cpp)
target_link_libraries(test_hamr_pipeline_cpu hamr)
add_test(NAME test_hamr_pipeline_cpu COMMAND test_hamr_pipeline_cpu)
set_tests_properties(test_hamr_pipeline_cpu PROPERTIES ENVIRONMENT HAMR_VERBOSE=1)

if (HAMR_ENABLE_CUDA)
    set_source_files_properties(test_hamr_multi_gpu_cuda.cpp PROPERTIES LANGUAGE CUDA)
    add_executable(test_hamr_multi_gpu_cuda test_hamr_multi_gpu_cuda.cpp)
    target_link_libraries(test_hamr_multi_gpu_cuda hamr)
    add_test(NAME test_hamr_multi_gpu_cuda COMMAND test_hamr_multi_gpu_cuda)
    set_tests_properties(test_hamr_multi_gpu_cuda PROPERTIES ENVIRONMENT HAMR_VERBOSE=1)
endif()

if (HAMR_ENABLE_HIP)
    add_executable(test_hamr_multi_gpu_hip test_hamr_multi_gpu_hip.cpp)
    target_link_libraries(test_hamr_multi_gpu_hip hamr)
    add_test(NAME test_hamr_multi_gpu_hip COMMAND test_hamr_multi_gpu_hip)
    set_tests_properties(test_hamr_multi_gpu_hip PROPERTIES ENVIRONMENT HAMR_VERBOSE=1)
endif()

if (HAMR_ENABLE_PYTHON)

    add_test(NAME test_hamr_buffer_numpy_cpu
        COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/test_hamr_buffer_numpy_cpu.py)

    if (HAMR_ENABLE_CUDA)

        add_test(NAME test_hamr_buffer_numpy_cuda
            COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/test_hamr_buffer_numpy_cuda.py)

        add_test(NAME test_hamr_buffer_cupy_cpu
            COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/test_hamr_buffer_cupy_cpu.py)

        add_test(NAME test_hamr_buffer_cupy_cuda
            COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/test_hamr_buffer_cupy_cuda.py)

    endif()

endif()

