#include "hamr_config.h"
#include "hamr_buffer.h"
#include "hamr_buffer_util.h"
#include "hamr_buffer_allocator.h"

#include <iostream>

using allocator = hamr::buffer_allocator;

// **************************************************************************
template <typename T>
hamr::buffer<T> initialize(size_t n_vals, const T &val)
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
hamr::buffer<T> add(const hamr::buffer<T> &a1, const hamr::buffer<U> &a2)
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
hamr::buffer<T> multiply_scalar(const hamr::buffer<T> &ai, const U &val)
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


// **************************************************************************
template <typename T>
int compare_int(const hamr::buffer<T> &ain, int val)
{
    size_t n_vals = ain.size();
    std::cerr << "comparing array with " << n_vals << " elements to " << val << std::endl;

    hamr::buffer<int> ai(ain.get_allocator(), n_vals);
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
    size_t n_vals = 10000;

    hamr::buffer<float>  ao0(allocator::openmp, n_vals, 1.0f);   // = 1 (device)
    hamr::buffer<float>  ao1 = multiply_scalar(ao0, 2.0f);       // = 2 (device)
    ao0.free();

    hamr::buffer<double> ao2 = initialize(n_vals, 2.0);          // = 2 (device)
    hamr::buffer<double> ao3 = add(ao2, ao1);                    // = 4 (device)
    ao1.free();
    ao2.free();

    hamr::buffer<double> ao4 = multiply_scalar(ao3, 1000.0);     // = 4000 (device)
    ao3.free();

    hamr::buffer<float>  ao5(allocator::malloc, n_vals, 3.0f);   // = 1 (host)
    hamr::buffer<float>  ao6 = multiply_scalar(ao5, 100.0f);     // = 300 (device)
    ao5.free();

    hamr::buffer<float> ao7(allocator::malloc, n_vals);          // = uninit (host)
    ao7.set(ao6);                                                // = 300 (host)
    ao6.free();

    hamr::buffer<double> ao8 = add(ao4, ao7);                    // = 4300 (device)
    ao4.free();
    ao7.free();

    return compare_int(ao8, 4300);
}
