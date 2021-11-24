#include "hamr_buffer.h"
#include "hamr_cuda_launch.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>

using std::make_shared;
using hamr::buffer;
using hamr::p_buffer;
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
p_buffer<T> initialize_cuda(size_t n_vals, const T &val)
{
    // allocate the memory
    p_buffer<T> ao = make_shared<buffer<T>>(allocator::cuda);
    ao->resize(n_vals);

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
    initialize_cuda<<<block_grid, thread_grid>>>(pao, val, n_vals);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        std::cerr << "ERROR: Failed to launch the initialize_cuda kernel. "
            << cudaGetErrorString(ierr) << std::endl;
        return nullptr;
    }

    std::cerr << "initialized to an array of " << n_vals << " to " << val << std::endl;
    if (n_vals < 33)
    {
        std::cerr << "ao = "; ao->print(); std::cerr << std::endl;
        ao->print();
        std::cerr << std::endl;
    }

    //cudaDeviceSynchronize();

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
p_buffer<T> add_cuda(const p_buffer<T> &a1,
    const p_buffer<U> &a2)
{
    // get the inputs
    auto spa1 = a1->get_cuda_accessible();
    const T *pa1 = spa1.get();

    auto spa2 = a2->get_cuda_accessible();
    const U *pa2 = spa2.get();

    // allocate the memory
    size_t n_vals = a1->size();
    p_buffer<T> ao = make_shared<buffer<T>>(allocator::cuda);
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

    std::cerr << "added " << n_vals << " array " << typeid(T).name() << sizeof(T)
         << " to array  " << typeid(U).name() << sizeof(U) << std::endl;
    if (n_vals < 33)
    {
        std::cerr << "a1 = "; a1->print(); std::cerr << std::endl;
        std::cerr << "a2 = "; a2->print(); std::cerr << std::endl;
        std::cerr << "ao = "; ao->print(); std::cerr << std::endl;
    }

    //cudaDeviceSynchronize();

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
p_buffer<T> multiply_scalar_cuda(const p_buffer<T> &ain, const U &val)
{
    // get the inputs
    auto spain = ain->get_cuda_accessible();
    const T *pain = spain.get();

    // allocate the memory
    size_t n_vals = ain->size();
    p_buffer<T> ao = make_shared<buffer<T>>(allocator::cuda);
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
    multiply_scalar_cuda<<<block_grid, thread_grid>>>(pao, pain, val, n_vals);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        std::cerr << "ERROR: Failed to launch the multiply_scalar_cuda kernel. "
            << cudaGetErrorString(ierr) << std::endl;
        return nullptr;
    }

    std::cerr << "multiply_scalar " << val << " " << typeid(U).name() << sizeof(U)
       << " by " << n_vals << " array " << typeid(T).name() << sizeof(T) << std::endl;

    if (n_vals < 33)
    {
        std::cerr << "ain = "; ain->print(); std::cerr << std::endl;
        std::cerr << "ao = "; ao->print(); std::cerr << std::endl;
    }

    //cudaDeviceSynchronize();

    return ao;
}



// **************************************************************************
template <typename T>
int compare_int(const p_buffer<T> &ain, int val)
{
    size_t n_vals = ain->size();
    std::cerr << "comparing array with " << n_vals << " elements to " << val << std::endl;

    p_buffer<int> ai = make_shared<buffer<int>>(ain->get_allocator());
    ai->resize(n_vals);
    ain->get(ref_to(ai));

    if (n_vals < 33)
    {
        ai->print();
    }

    auto spai = ai->get_cpu_accessible();
    int *pai = spai.get();

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

    p_buffer<float>  ao0 = make_shared<buffer<float>>(allocator::cuda, n_vals, 1.0f);   // = 1 (CUDA)
    p_buffer<float>  ao1 = multiply_scalar_cuda(ao0, 2.0f);                             // = 2 (CUDA)
    ao0 = nullptr;

    p_buffer<double> ao2 = initialize_cuda(n_vals, 2.0);                                // = 2 (CUDA)
    p_buffer<double> ao3 = add_cuda(ao2, ao1);                                          // = 4 (CUDA)
    ao1 = nullptr;
    ao2 = nullptr;

    p_buffer<double> ao4 = multiply_scalar_cuda(ao3, 1000.0);                           // = 4000 (CUDA)
    ao3 = nullptr;

    p_buffer<float>  ao5 = make_shared<buffer<float>>(allocator::malloc, n_vals, 3.0f); // = 1 (CPU)
    p_buffer<float>  ao6 = multiply_scalar_cuda(ao5, 100.0f);                           // = 300 (CUDA)
    ao5 = nullptr;

    p_buffer<float> ao7 = make_shared<buffer<float>>(allocator::malloc, n_vals);        // = uninit (CPU)
    ao7->set(ref_to(ao6));                                                              // = 300 (CPU)
    ao6 = nullptr;

    p_buffer<double> ao8 = add_cuda(ao4, ao7);                                          // = 4300 (CUDA)
    ao4 = nullptr;
    ao7 = nullptr;

    return compare_int(ao8, 4300);

    /*
    p_buffer<double> ao9 = make_shared<buffer<double>>(allocator::malloc);                // = empty (CPU)
    ao9->assign(ao8);                                                                        // = 4300 (CPU)
    ao8 = nullptr;

    ao9->print();
    ao9 = nullptr;

    return 0;
    */
}
