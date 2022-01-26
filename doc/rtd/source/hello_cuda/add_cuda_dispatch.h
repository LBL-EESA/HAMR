#ifndef add_cuda_dispatch_h
#define add_cuda_dispatch_h

#include "add_cuda.h"

#include <hamr_buffer.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>

using hamr::buffer;
using allocator = hamr::buffer_allocator;

// **************************************************************************
template <typename T, typename U>
buffer<T> add_cuda(const buffer<T> &a1, const buffer<U> &a2)
{
    // get pointers to the input arrays
    auto spa1 = a1.get_cuda_accessible();
    const T *pa1 = spa1.get();

    auto spa2 = a2.get_cuda_accessible();
    const U *pa2 = spa2.get();

    // allocate the memory for the result
    size_t n_vals = a1.size();
    buffer<T> ao(allocator::cuda, n_vals, T(0));

    // get pointer to the output array
    auto spao = ao.get_cuda_accessible();
    T *pao = spao.get();

    // determine kernel launch parameters and launch the kernel to add the arrays
    dim3 thread_grid(128);
    dim3 block_grid(n_vals/128 + (n_vals % 128 ? 1 : 0));

    add_cuda<<<block_grid, thread_grid>>>(pao, pa1, pa2, n_vals);

    return ao;
}

#endif
