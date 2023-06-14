#include "hamr_buffer.h"
#include "hamr_buffer_util.h"
#include "hamr_cuda_launch.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>

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



// **************************************************************************
template <typename T>
int compare_int(const buffer<T> &ain, int val)
{
    size_t n_vals = ain.size();
    std::cerr << "comparing array with " << n_vals << " elements to " << val << std::endl;

    buffer<int> ai(ain.get_allocator(), n_vals);
    ain.get(ai);

    auto [spai, pai] = hamr::get_host_accessible(ai);

    if (n_vals < 33)
    {
        ai.print();
    }

    for (size_t i = 0; i < n_vals; ++i)
    {
        if (pai[i] != val)
        {
            std::cerr << "ERROR: pai[" << i << "] = "
                << pai[i] << " != " << val << std::endl;
            return -1;
        }
    }

    std::cerr << "all elements are equal to " << val << std::endl;

    return 0;
}



int main(int, char **)
{
    size_t n_vals = 100000;

    buffer<float>  ao0(allocator::cuda, n_vals, 1.0f);                                // = 1 (CUDA)
    buffer<float>  ao1 = multiply_scalar_cuda(ao0, 2.0f);                             // = 2 (CUDA)
    ao0.free();

    buffer<double> ao2 = initialize_cuda(n_vals, 2.0);                                // = 2 (CUDA)
    buffer<double> ao3 = add_cuda(ao2, ao1);                                          // = 4 (CUDA)
    ao1.free();
    ao2.free();

    buffer<double> ao4 = multiply_scalar_cuda(ao3, 1000.0);                           // = 4000 (CUDA)
    ao3.free();

    buffer<float>  ao5(allocator::malloc, n_vals, 3.0f);                              // = 1 (host)
    buffer<float>  ao6 = multiply_scalar_cuda(ao5, 100.0f);                           // = 300 (CUDA)
    ao5.free();

    buffer<float> ao7(allocator::malloc, n_vals);                                     // = uninit (host)
    ao7.set(ao6);                                                                     // = 300 (host)
    ao6.free();

    buffer<double> ao8 = add_cuda(ao4, ao7);                                          // = 4300 (CUDA)
    ao4.free();
    ao7.free();

    return compare_int(ao8, 4300);
}
