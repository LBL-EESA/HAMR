#ifndef hamr_cuda_kernels_h
#define hamr_cuda_kernels_h

#include "hamr_config.h"
#include "hamr_env.h"
#include "hamr_cuda_launch.h"

namespace hamr
{

namespace cuda_kernels
{

/// helpers to get the printf code given a POD type
template <typename T> struct printf_tt {};

#define declare_printf_tt(cpp_t, print_t, code, len)\
/** printf code wrapper for cpp_t */                \
template <> struct printf_tt<cpp_t>                 \
{                                                   \
    /** cast from cpp_t to print_t */               \
    __device__                                      \
    static print_t get_value(cpp_t v)               \
    { return v; }                                   \
                                                    \
    /** returns the printf code for cpp_t */        \
    __device__                                      \
    static const char *get_code()                   \
    { return code; }                                \
                                                    \
    /** copies the printf code */                   \
    __device__                                      \
    static void copy_code(char *dest)               \
    {                                               \
        for (int i = 0; i < len; ++i)               \
            dest[i] = get_code()[i];                \
    }                                               \
                                                    \
    /** returns the length of the printf code */    \
    __device__                                      \
    static int get_code_len()                       \
    { return len; }                                 \
};

declare_printf_tt(char, int, "%d", 2)
declare_printf_tt(signed char, int, "%d", 2)
declare_printf_tt(unsigned char, unsigned int, "%u", 2)
declare_printf_tt(short, short, "%hd", 3)
declare_printf_tt(unsigned short, unsigned short, "%hu", 3)
declare_printf_tt(int, int, "%d", 2)
declare_printf_tt(unsigned int, unsigned int, "%u", 2)
declare_printf_tt(long, long, "%ld", 3)
declare_printf_tt(unsigned long, unsigned long, "%lu", 3)
declare_printf_tt(long long, long long, "%lld", 4)
declare_printf_tt(unsigned long long, unsigned long long, "%llu", 4)
declare_printf_tt(float, float, "%g", 2)
declare_printf_tt(double, double, "%g", 2)


/// send an array to the stderr stream on the GPU using CUDA
template <typename T>
__global__
void print(const T *vals, size_t n_elem)
{
    unsigned long i = hamr::thread_id_to_array_index();

    if (i >= n_elem)
        return;

    int cl = printf_tt<T>::get_code_len();
    char fmt[] = "vals[%lu] = XXXXXXXXX"; // <-- 20
    printf_tt<T>::copy_code(fmt + 12);
    fmt[12 + cl] = '\n';
    fmt[13 + cl] = '\0';

    printf(fmt, i, printf_tt<T>::get_value(vals[i]));
}

/// copy an array on the GPU using CUDA
template <typename T, typename U>
__global__
void copy(T *dest, const U *src, size_t n_elem)
{
    unsigned long i = hamr::thread_id_to_array_index();

    if (i >= n_elem)
        return;

    dest[i] = static_cast<T>(src[i]);
}

/// default construct on the GPU
template <typename T>
__global__
void construct(T *dest, size_t n_elem)
{
    unsigned long i = hamr::thread_id_to_array_index();

    if (i >= n_elem)
        return;

    new (&dest[i]) T();
}

/// copy construct on the GPU
template <typename T, typename U>
__global__
void construct(T *dest, size_t n_elem, U val)
{
    unsigned long i = hamr::thread_id_to_array_index();

    if (i >= n_elem)
        return;

    new (&dest[i]) T(val);
}

/// copy construct on the GPU
template <typename T, typename U>
__global__
void construct(T *dest, size_t n_elem, const U *vals)
{
    unsigned long i = hamr::thread_id_to_array_index();

    if (i >= n_elem)
        return;

    new (&dest[i]) T(vals[i]);
}

/// destruct on the GPU
template <typename T>
__global__
void destruct(T *dest, size_t n_elem)
{
    unsigned long i = hamr::thread_id_to_array_index();

    if (i >= n_elem)
        return;

    dest[i].~T();
}

/// initialize an array on the GPU
template <typename T, typename U>
__global__
void fill(T *dest, size_t n_elem, U val)
{
    unsigned long i = hamr::thread_id_to_array_index();

    if (i >= n_elem)
        return;

    dest[i] = val;
}

/// initialize an array on the GPU
template <typename T, typename U>
__global__
void fill(T *dest, size_t n_elem, const U *vals)
{
    unsigned long i = hamr::thread_id_to_array_index();

    if (i >= n_elem)
        return;

    dest[i] = vals[i];
}

}

}

#endif
