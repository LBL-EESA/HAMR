#include "hamr_config.h"
#include "hamr_buffer.h"
#include "hamr_buffer_util.h"

#include <iostream>

using hamr::buffer;
using allocator = hamr::buffer_allocator;

// **************************************************************************
template <typename T>
hamr::buffer<T> initialize_openmp(size_t n_vals, const T &val)
{
    // allocate the memory
    hamr::buffer<T> ao(allocator::openmp, n_vals);
    T *pao = ao.data();

    // initialize using openmp

    #pragma omp target teams HAMR_OPENMP_LOOP is_device_ptr(pao) map(to: val)
    for (size_t i = 0; i < n_vals; ++i)
    {
        pao[i] = val;
    }

    // print the results
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
template <typename T, typename U>
hamr::buffer<T> add_openmp(const hamr::buffer<T> &a1, const hamr::buffer<U> &a2)
{
    size_t n_vals = a1.size();

    // get the inputs
    auto spa1 = a1.get_openmp_accessible();
    auto pa1 = spa1.get();

    auto spa2 = a2.get_openmp_accessible();
    auto pa2 = spa2.get();

    // allocate the memory
    hamr::buffer<T> ao(allocator::openmp, n_vals, T(0));
    T *pao = ao.data();

    // do the calculation
    #pragma omp target teams HAMR_OPENMP_LOOP is_device_ptr(pao, pa1, pa2)
    for (size_t i = 0; i < n_vals; ++i)
    {
        pao[i] = pa1[i] + pa2[i];
    }

    // print the results
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
template <typename T, typename U>
hamr::buffer<T> multiply_scalar_openmp(const hamr::buffer<T> &ai, const U &val)
{
    size_t n_vals = ai.size();

    // get the inputs
    auto spai = ai.get_openmp_accessible();
    auto pai = spai.get();

    // allocate the memory
    hamr::buffer<T> ao(allocator::openmp, n_vals, T(0));
    T *pao = ao.data();

    // do the calculation
    #pragma omp target teams HAMR_OPENMP_LOOP is_device_ptr(pao, pai) map(to: val)
    for (size_t i = 0; i < n_vals; ++i)
    {
        pao[i] = val * pai[i];
    }

    // print the results
    std::cerr << "multiply_scalar " << val << " " << typeid(U).name() << sizeof(U)
       << " by " << n_vals << " array " << typeid(T).name() << sizeof(T) << std::endl;

    if (n_vals < 33)
    {
        std::cerr << "ai = "; ai.print(); std::cerr << std::endl;
        std::cerr << "ao = "; ao.print(); std::cerr << std::endl;
    }

    return ao;
}

#define instantiate_openmp_kernels_(T,U) \
template buffer<T> add_openmp<T,U>(const buffer<T> &a1, const buffer<U> &a2); \
template buffer<T> multiply_scalar_openmp<T,U>(const buffer<T> &ai, const U &val);

#define instantiate_openmp_kernels(T) \
template buffer<T> initialize_openmp(size_t n_vals, const T &val); \
instantiate_openmp_kernels_(T, float) \
instantiate_openmp_kernels_(T, double)

instantiate_openmp_kernels(double)
instantiate_openmp_kernels(float)

