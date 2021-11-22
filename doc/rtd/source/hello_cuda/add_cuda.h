#ifndef add_cuda_h
#define add_cuda_h

#include "hamr_cuda_launch.h"

#include <cuda.h>
#include <cuda_runtime.h>

// **************************************************************************
template<typename T, typename U>
__global__
void add_cuda(T *result, const T *array_1, const U *array_2, size_t n_vals)
{
    unsigned long i = hamr::thread_id_to_array_index();

    if (i >= n_vals)
        return;

    result[i] = array_1[i] + array_2[i];
}

#endif
