#ifndef add_cuda_dispatch_h
#define add_cuda_dispatch_h

#include "add_cuda.h"

#include <hamr_buffer.h>
#include <hamr_cuda_launch.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>

using hamr::buffer;
using hamr::p_buffer;
using allocator = hamr::buffer_allocator;

// **************************************************************************
template <typename T, typename U>
p_buffer<T> add_cuda(const p_buffer<T> &a1, const p_buffer<U> &a2)
{
    // get the inputs
    auto spa1 = a1->get_cuda_accessible();
    const T *pa1 = spa1.get();

    auto spa2 = a2->get_cuda_accessible();
    const U *pa2 = spa2.get();

    // allocate the memory
    size_t n_vals = a1->size();
    p_buffer<T> ao = std::make_shared<buffer<T>>(allocator::cuda);
    ao->resize(n_vals, T(0));

    auto spao = ao->get_cuda_accessible();
    T *pao = spao.get();

    // determine kernel launch parameters
    int n_blocks = 0;
    dim3 block_grid;
    dim3 thread_grid;
    if (hamr::partition_thread_blocks(0, n_vals,
        8, block_grid, n_blocks, thread_grid))
    {
        std::cerr << "ERROR: Failed to determine launch parameters" << std::endl;
        return nullptr;
    }

    // initialize the data
    cudaError_t ierr = cudaSuccess;
    add_cuda<<<block_grid, thread_grid>>>(pao, pa1, pa2, n_vals);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        std::cerr << "ERROR: Failed to launch the add_cuda kernel. "
            << cudaGetErrorString(ierr) << std::endl;
        return nullptr;
    }

    return ao;
}

#endif
