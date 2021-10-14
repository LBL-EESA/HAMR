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

#define declare_printf_tt(cpp_t, code, len)     \
template <> struct printf_tt<cpp_t>             \
{                                               \
    __device__                                  \
    static const char *get_code()               \
    { return code; }                            \
                                                \
    __device__                                  \
    static void copy_code(char *dest)           \
    {                                           \
        for (int i = 0; i < len; ++i)           \
            dest[i] = get_code()[i];            \
    }                                           \
                                                \
    __device__                                  \
    static int get_code_len()                   \
    { return len; }                             \
};

declare_printf_tt(char, "%hhd", 4)
declare_printf_tt(unsigned char, "%hhu", 4)
declare_printf_tt(short, "%hd", 3)
declare_printf_tt(unsigned short, "%hu", 3)
declare_printf_tt(int, "%d", 2)
declare_printf_tt(unsigned int, "%u", 2)
declare_printf_tt(long, "%ld", 3)
declare_printf_tt(unsigned long, "%lu", 3)
declare_printf_tt(long long, "%lld", 4)
declare_printf_tt(unsigned long long, "%llu", 4)
declare_printf_tt(float, "%g", 2)
declare_printf_tt(double, "%g", 2)


/// send an array to the stderr stream on the GPU using CUDA
template <typename T>
__global__
void print(T *vals, size_t n_elem)
{
    unsigned long i = hamr::thread_id_to_array_index();

    if (i >= n_elem)
        return;

    int cl = printf_tt<double>::get_code_len();
    char fmt[] = "vals[%d] = XXXXXXXXX"; // <-- 20
    printf_tt<double>::copy_code(fmt + 12);
    fmt[12 + cl] = '\n';
    fmt[13 + cl] = '\0';

    printf(fmt, i, vals[i]);
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
template <typename T>
__global__
void construct(T *dest, size_t n_elem, T val)
{
    unsigned long i = hamr::thread_id_to_array_index();

    if (i >= n_elem)
        return;

    new (&dest[i]) T(val);
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
template <typename T>
__global__
void fill(T *dest, size_t n_elem, T val)
{
    unsigned long i = hamr::thread_id_to_array_index();

    if (i >= n_elem)
        return;

    dest[i] = val;
}

}

}

#endif
