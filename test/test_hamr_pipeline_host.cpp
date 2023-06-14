#include "hamr_buffer.h"
#include "hamr_buffer_util.h"

#include <iostream>

using hamr::buffer;
using allocator = hamr::buffer_allocator;

// **************************************************************************
template<typename T>
void initialize_host(T *data, double val, size_t n_vals)
{
    for (size_t i = 0; i < n_vals; ++i)
    {
        data[i] = val;
    }
}

// **************************************************************************
template <typename T>
buffer<T> initialize_host(size_t n_vals, const T &val)
{
    // allocate the memory
    buffer<T> ao(allocator::malloc, n_vals);
    T *pao = ao.data();

    // initialize the data
    initialize_host(pao, val, n_vals);

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
void add_host(T *result, const T *array_1, const U *array_2, size_t n_vals)
{
    for (size_t i = 0; i < n_vals; ++i)
    {
        result[i] = array_1[i] + array_2[i];
    }
}

// **************************************************************************
template <typename T, typename U>
buffer<T> add_host(const buffer<T> &a1, const buffer<U> &a2)
{
    // get the inputs
    auto [spa1, pa1] = hamr::get_host_accessible(a1);
    auto [spa2, pa2] = hamr::get_host_accessible(a2);

    // allocate the memory
    size_t n_vals = a1.size();
    buffer<T> ao(allocator::malloc, n_vals, T(0));
    T *pao = ao.data();

    // initialize the data
    add_host(pao, pa1, pa2, n_vals);

    std::cerr << "added " << n_vals << " array " << typeid(T).name()
        << sizeof(T) << " to array  " << typeid(U).name() << sizeof(U) << std::endl;
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
void multiply_scalar_host(T *result, const T *array_in, U scalar, size_t n_vals)
{
    for (size_t i = 0; i < n_vals; ++i)
    {
        result[i] = array_in[i] * scalar;
    }
}

// **************************************************************************
template <typename T, typename U>
buffer<T> multiply_scalar_host(const buffer<T> &ai, const U &val)
{
    // get the inputs
    auto [spai, pai] = hamr::get_host_accessible(ai);

    // allocate the memory
    size_t n_vals = ai.size();
    buffer<T> ao(allocator::malloc, n_vals, T(0));
    T *pao = ao.data();

    // initialize the data
    multiply_scalar_host(pao, pai, val, n_vals);

    std::cerr << "multiply_scalar " << val << " " << typeid(U).name() << sizeof(U)
       << " by " << n_vals << " array " << typeid(T).name() << sizeof(T) << std::endl;

    if (n_vals < 33)
    {
        std::cerr << "ain = "; ai.print(); std::cerr << std::endl;
        std::cerr << "ao = "; ao.print(); std::cerr << std::endl;
    }

    return ao;
}

// **************************************************************************
template <typename T>
int compare_int(const buffer<T> &ain, int val)
{
    size_t n_vals = ain.size();
    std::cerr << "comparing array with " << n_vals
        << " elements to " << val << std::endl;

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
            std::cerr << "ERROR: pai[" << i << "] = " << pai[i]
                << " != " << val << std::endl;
            return -1;
        }
    }

    std::cerr << "all elements are equal to " << val << std::endl;

    return 0;
}



int main(int, char **)
{
    size_t n_vals = 100000;

    buffer<float>  ao0(allocator::malloc, n_vals, 1.0f);    // = 1 (host)
    buffer<float>  ao1 = multiply_scalar_host(ao0, 2.0f);    // = 2 (host)
    ao0.free();

    buffer<double> ao2 = initialize_host(n_vals, 2.0);       // = 2 (host)
    buffer<double> ao3 = add_host(ao2, ao1);                 // = 4 (host)
    ao1.free();
    ao2.free();

    buffer<double> ao4 = multiply_scalar_host(ao3, 1000.0);  // = 4000 (host)
    ao3.free();

    buffer<float>  ao5(allocator::malloc, n_vals, 3.0f);    // = 1 (host)
    buffer<float>  ao6 = multiply_scalar_host(ao5, 100.0f);  // = 300 (host)
    ao5.free();

    buffer<float> ao7(allocator::malloc, n_vals);           // = uninit (host)
    ao7.set(ao6);                                           // = 300 (host)

    buffer<double> ao8 = add_host(ao4, ao7);                 // = 4300 (host)
    ao4.free();
    ao7.free();

    int res = compare_int(ao8, 4300);
    ao8.free();

    return res;
}
