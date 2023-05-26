#include "hamr_config.h"
#include "hamr_buffer.h"
#include "hamr_buffer_util.h"
#include "hamr_cuda_launch.h"

#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

using hamr::buffer;
using allocator = hamr::buffer_allocator;

// **************************************************************************
template<typename T>
__global__
void initialize_cuda(T *data, double val, size_t n_vals)
{
    unsigned long i = hamr::thread_id_to_array_index();

    if (i >= n_vals)
        return;

    data[i] = val;
}

// **************************************************************************
template <typename T>
buffer<T> initialize_cuda(size_t n_vals, const T &val)
{
    // allocate the output
    buffer<T> ao(allocator::cuda, n_vals);
    T *pao = ao.data();

    // determine kernel launch parameters
    int n_blocks = 0;
    dim3 block_grid;
    dim3 thread_grid;
    if (hamr::partition_thread_blocks(0, n_vals,
        8, block_grid, n_blocks, thread_grid))
    {
        std::cerr << "ERROR: Failed to determine launch parameters" << std::endl;
        abort();
    }

    // initialize the data
    cudaError_t ierr = cudaSuccess;
    initialize_cuda<<<block_grid, thread_grid>>>(pao, val, n_vals);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        std::cerr << "ERROR: Failed to launch the initialize_cuda kernel. "
            << cudaGetErrorString(ierr) << std::endl;
        abort();
    }

    std::cerr << "initialized to an array of " << n_vals << " to " << val << std::endl;
    if (n_vals < 33)
    {
        std::cerr << "ao = "; ao.print(); std::cerr << std::endl;
        ao.print();
        std::cerr << std::endl;
    }

    return ao;
}






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

// **************************************************************************
template <typename T, typename U>
buffer<T> add_cuda(const buffer<T> &a1, const buffer<U> &a2)
{
    size_t n_vals = a1.size();

    // get the inputs
    auto [spa1, pa1] = hamr::get_cuda_accessible(a1);
    auto [spa2, pa2] = hamr::get_cuda_accessible(a2);

    // allocate the output
    buffer<T> ao(allocator::cuda, n_vals, T(0));
    T *pao = ao.data();

    // determine kernel launch parameters
    int n_blocks = 0;
    dim3 block_grid;
    dim3 thread_grid;
    if (hamr::partition_thread_blocks(0, n_vals,
        8, block_grid, n_blocks, thread_grid))
    {
        std::cerr << "ERROR: Failed to determine launch parameters" << std::endl;
        abort();
    }

    // initialize the data
    cudaError_t ierr = cudaSuccess;
    add_cuda<<<block_grid, thread_grid>>>(pao, pa1, pa2, n_vals);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        std::cerr << "ERROR: Failed to launch the add_cuda kernel. "
            << cudaGetErrorString(ierr) << std::endl;
        abort();
    }

    std::cerr << "added " << n_vals << " array " << typeid(T).name() << sizeof(T)
         << " to array  " << typeid(U).name() << sizeof(U) << std::endl;
    if (n_vals < 33)
    {
        std::cerr << "a1 = "; a1.print(); std::cerr << std::endl;
        std::cerr << "a2 = "; a2.print(); std::cerr << std::endl;
        std::cerr << "ao = "; ao.print(); std::cerr << std::endl;
    }

    return ao;
}





// **************************************************************************
template<typename T, typename U>
__global__
void multiply_scalar_cuda(T *result, const T *array_in, U scalar, size_t n_vals)
{
    unsigned long i = hamr::thread_id_to_array_index();

    if (i >= n_vals)
        return;

    result[i] = array_in[i] * scalar;
}

// **************************************************************************
template <typename T, typename U>
buffer<T> multiply_scalar_cuda(const buffer<T> &ai, const U &val)
{
    size_t n_vals = ai.size();

    // get the inputs
    auto [spai, pai] = hamr::get_cuda_accessible(ai);

    // allocate the output
    buffer<T> ao(allocator::cuda, n_vals, T(0));
    T *pao = ao.data();

    // determine kernel launch parameters
    int n_blocks = 0;
    dim3 block_grid;
    dim3 thread_grid;
    if (hamr::partition_thread_blocks(0, n_vals,
        8, block_grid, n_blocks, thread_grid))
    {
        std::cerr << "ERROR: Failed to determine launch parameters" << std::endl;
        abort();
    }

    // initialize the data
    cudaError_t ierr = cudaSuccess;
    multiply_scalar_cuda<<<block_grid, thread_grid>>>(pao, pai, val, n_vals);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        std::cerr << "ERROR: Failed to launch the multiply_scalar_cuda kernel. "
            << cudaGetErrorString(ierr) << std::endl;
        abort();
    }

    std::cerr << "multiply_scalar " << val << " " << typeid(U).name() << sizeof(U)
       << " by " << n_vals << " array " << typeid(T).name() << sizeof(T) << std::endl;

    if (n_vals < 33)
    {
        std::cerr << "ai = "; ai.print(); std::cerr << std::endl;
        std::cerr << "ao = "; ao.print(); std::cerr << std::endl;
    }

    return ao;
}

#define instantiate_cuda_kernels_(T,U) \
template buffer<T> add_cuda<T,U>(const buffer<T> &a1, const buffer<U> &a2); \
template buffer<T> multiply_scalar_cuda<T,U>(const buffer<T> &ai, const U &val);

#define instantiate_cuda_kernels(T) \
template buffer<T> initialize_cuda(size_t n_vals, const T &val); \
instantiate_cuda_kernels_(T, float) \
instantiate_cuda_kernels_(T, double)

instantiate_cuda_kernels(double)
instantiate_cuda_kernels(float)

