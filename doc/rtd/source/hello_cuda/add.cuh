#ifndef add_cuda_h
#define add_cuda_h

template<typename T, typename U>
__global__
void add(T *result, const T *array_1, const U *array_2, size_t n_vals)
{
    unsigned long i = blockIdx.x*blockDim.x + threadIdx.x;

    if (i >= n_vals)
        return;

    result[i] = array_1[i] + array_2[i];
}

#endif
