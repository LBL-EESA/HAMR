#include "hamr_buffer.h"
#include "hamr_buffer_util.h"

#include <iostream>

using hamr::buffer;
using allocator = hamr::buffer_allocator;

// with LLVM Clang CUDA and OpenMP need to be compiled in separate
// translation units.

//
// CUDA kernels
//
template <typename T>
buffer<T> initialize_cuda(size_t n_vals, const T &val) HAMR_EXPORT;

template <typename T, typename U>
buffer<T> add_cuda(const buffer<T> &a1, const buffer<U> &a2) HAMR_EXPORT;

template <typename T, typename U>
buffer<T> multiply_scalar_cuda(const buffer<T> &ai, const U &val) HAMR_EXPORT;

//
// OpenMP kernels
//
template <typename T>
buffer<T> initialize_openmp(size_t n_vals, const T &val) HAMR_EXPORT;

template <typename T, typename U>
buffer<T> add_openmp(const buffer<T> &a1, const buffer<U> &a2) HAMR_EXPORT;

template <typename T, typename U>
buffer<T> multiply_scalar_openmp(const buffer<T> &ai, const U &val) HAMR_EXPORT;



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

    buffer<float>  ao0(allocator::cuda, n_vals, 1.0f);         // = 1 (CUDA)
    buffer<float>  ao1 = multiply_scalar_cuda(ao0, 2.0f);      // = 2 (CUDA)
    ao0.free();

    buffer<double> ao2 = initialize_openmp(n_vals, 2.0);       // = 2 (OpenMP)
    buffer<double> ao3 = add_openmp(ao2, ao1);                 // = 4 (OpenMP w/ CUDA data)
    ao1.free();
    ao2.free();

    buffer<double> ao4 = multiply_scalar_cuda(ao3, 1000.0);    // = 4000 (CUDA w/ OpenMP data)
    ao3.free();

    buffer<float>  ao5(allocator::malloc, n_vals, 3.0f);       // = 3 (host)
    buffer<float>  ao6 = multiply_scalar_cuda(ao5, 100.0f);    // = 300 (CUDA)
    ao5.free();

    buffer<float> ao7(allocator::openmp, n_vals);              // = uninit (OpenMP)
    ao7.set(ao6);                                              // = 300 (CUDA to OpenMP)
    ao6.free();

    buffer<float> ao8(allocator::cuda, n_vals);                // = uninit (CUDA)
    ao8.set(ao7);                                              // = 300 (OpenMP to CUDA)
    ao7.free();

    buffer<double> ao9 = add_cuda(ao4, ao8);                   // = 4300 (CUDA)
    ao4.free();
    ao8.free();

    return compare_int(ao8, 4300);
}
