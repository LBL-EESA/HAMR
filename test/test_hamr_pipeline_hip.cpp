#include "hamr_buffer.h"
#include "hamr_buffer_util.h"
#include "hamr_hip_launch.h"
#include "hamr_buffer_pointer.h"

#include <hip/hip_runtime.h>


#include <iostream>

using hamr::buffer;
using allocator = hamr::buffer_allocator;

// **************************************************************************
template<typename T>
__global__
void initialize_hip(T *data, double val, size_t n_vals)
{
    unsigned long i = hamr::thread_id_to_array_index();

    if (i >= n_vals)
        return;

    data[i] = val;
}

// **************************************************************************
template <typename T>
buffer<T> initialize_hip(size_t n_vals, const T &val)
{
    // allocate the output
    buffer<T> ao(allocator::hip, n_vals);
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
    hipError_t ierr = hipSuccess;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(initialize_hip), dim3(block_grid), dim3(thread_grid), 0, 0, pao, val, n_vals);
    if ((ierr = hipGetLastError()) != hipSuccess)
    {
        std::cerr << "ERROR: Failed to launch the initialize_hip kernel. "
            << hipGetErrorString(ierr) << std::endl;
        abort();
    }

    std::cerr << "initialized to an array of " << n_vals << " to " << val << std::endl;
    if (n_vals < 33)
    {
        std::cerr << "ao = "; ao.print(); std::cerr << std::endl;
        ao.print();
        std::cerr << std::endl;
    }

    //hipDeviceSynchronize();

    return ao;
}






// **************************************************************************
template<typename T, typename U>
__global__
void add_hip(T *result, const T *array_1, const U *array_2, size_t n_vals)
{
    unsigned long i = hamr::thread_id_to_array_index();

    if (i >= n_vals)
        return;

    result[i] = array_1[i] + array_2[i];
}

// **************************************************************************
template <typename T, typename U>
buffer<T> add_hip(const buffer<T> &a1, const buffer<U> &a2)
{
    size_t n_vals = a1.size();

    // get the inputs
    auto [spa1, pa1] = hamr::get_hip_accessible(a1);
    auto [spa2, pa2] = hamr::get_hip_accessible(a2);

    // allocate the output
    buffer<T> ao(allocator::hip, n_vals, T(0));
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
    hipError_t ierr = hipSuccess;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(add_hip), dim3(block_grid), dim3(thread_grid), 0, 0, pao, pa1, pa2, n_vals);
    if ((ierr = hipGetLastError()) != hipSuccess)
    {
        std::cerr << "ERROR: Failed to launch the add_hip kernel. "
            << hipGetErrorString(ierr) << std::endl;
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

    //hipDeviceSynchronize();

    return ao;
}





// **************************************************************************
template<typename T, typename U>
__global__
void multiply_scalar_hip(T *result, const T *array_in, U scalar, size_t n_vals)
{
    unsigned long i = hamr::thread_id_to_array_index();

    if (i >= n_vals)
        return;

    result[i] = array_in[i] * scalar;
}

// **************************************************************************
template <typename T, typename U>
buffer<T> multiply_scalar_hip(const buffer<T> &ai, const U &val)
{
    size_t n_vals = ai.size();

    // get the inputs
    auto [spai, pai] = hamr::get_hip_accessible(ai);

    // allocate the output
    buffer<T> ao(allocator::hip, n_vals, T(0));
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
    hipError_t ierr = hipSuccess;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(multiply_scalar_hip), dim3(block_grid), dim3(thread_grid), 0, 0, pao, pai, val, n_vals);
    if ((ierr = hipGetLastError()) != hipSuccess)
    {
        std::cerr << "ERROR: Failed to launch the multiply_scalar_hip kernel. "
            << hipGetErrorString(ierr) << std::endl;
        abort();
    }

    std::cerr << "multiply_scalar " << val << " " << typeid(U).name() << sizeof(U)
       << " by " << n_vals << " array " << typeid(T).name() << sizeof(T) << std::endl;

    if (n_vals < 33)
    {
        std::cerr << "ain = "; ai.print(); std::cerr << std::endl;
        std::cerr << "ao = "; ao.print(); std::cerr << std::endl;
    }

    //hipDeviceSynchronize();

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

    if (n_vals < 33)
    {
        ai.print();
    }

    auto [spai, pai] = hamr::get_host_accessible(ai);

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

    buffer<float>  ao0(allocator::hip, n_vals, 1.0f);      // = 1 (HIP)
    buffer<float>  ao1 = multiply_scalar_hip(ao0, 2.0f);   // = 2 (HIP)
    ao0.free();

    buffer<double> ao2 = initialize_hip(n_vals, 2.0);      // = 2 (HIP)
    buffer<double> ao3 = add_hip(ao2, ao1);                // = 4 (HIP)
    ao1.free();
    ao2.free();

    buffer<double> ao4 = multiply_scalar_hip(ao3, 1000.0); // = 4000 (HIP)
    ao3.free();

    buffer<float>  ao5(allocator::malloc, n_vals, 3.0f);   // = 1 (host)
    buffer<float>  ao6 = multiply_scalar_hip(ao5, 100.0f); // = 300 (HIP)
    ao5.free();

    buffer<float> ao7(allocator::malloc, n_vals);          // = uninit (host)
    ao7.set(ao6);                                          // = 300 (host)
    ao6.free();

    buffer<double> ao8 = add_hip(ao4, ao7);                // = 4300 (HIP)
    ao4.free();
    ao7.free();

    return compare_int(ao8, 4300);
}
