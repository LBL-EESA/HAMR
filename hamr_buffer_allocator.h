#ifndef hamr_buffer_allocator_h
#define hamr_buffer_allocator_h

///@file

#include "hamr_config.h"
#include <cassert>

namespace hamr
{

/// allocator types that may be used with hamr::buffer
enum class buffer_allocator
{
    same = -2,     ///< propagate the current allocator
    none = -1,     ///< no allocator specified
    cpp = 0,       ///< allocates memory with new
    malloc = 1,    ///< allocates memory with malloc
    cuda = 2,      ///< allocates memory with cudaMalloc
    cuda_uva = 3,  ///< allocates memory with cudaMallocManaged
    hip = 4,       ///< allocates memory with hipMalloc
    hip_uva = 5,   ///< allocates memory with hipMallocManaged
    openmp = 6     ///< allocates memory with OpenMP device offload API
};

/// return the human readable name of the allocator
HAMR_EXPORT
const char *get_allocator_name(buffer_allocator alloc);

/// @returns true if the allocator creates CPU accessible memory
inline
HAMR_EXPORT
int cpu_accessible(buffer_allocator alloc)
{
    return (alloc == buffer_allocator::cpp) ||
        (alloc == buffer_allocator::malloc) ||
        (alloc == buffer_allocator::cuda_uva) ||
        (alloc == buffer_allocator::hip_uva);
}

/// @returns true if the allocator creates CUDA accessible memory
inline
HAMR_EXPORT
int cuda_accessible(buffer_allocator alloc)
{
    return (alloc == buffer_allocator::cuda) ||
        (alloc == buffer_allocator::cuda_uva) ||
        (alloc == buffer_allocator::hip) ||
        (alloc == buffer_allocator::hip_uva) ||
        (alloc == buffer_allocator::openmp);
}

/// @returns true if the allocator creates HIP accessible memory
inline
HAMR_EXPORT
int hip_accessible(buffer_allocator alloc)
{
    return (alloc == buffer_allocator::cuda) ||
        (alloc == buffer_allocator::cuda_uva) ||
        (alloc == buffer_allocator::hip) ||
        (alloc == buffer_allocator::hip_uva);
}

/// @returns true if the allocator creates OPENMP accessible memory
inline
HAMR_EXPORT
int openmp_accessible(buffer_allocator alloc)
{
    return (alloc == buffer_allocator::cuda) ||
        (alloc == buffer_allocator::cuda_uva) ||
        (alloc == buffer_allocator::openmp);
}

/// asserts that the passed value is one of the known allocators
inline
HAMR_EXPORT
void assert_valid_allocator(buffer_allocator alloc)
{
    (void) alloc;
    assert((alloc == buffer_allocator::cpp)
        || (alloc == buffer_allocator::malloc)
#if defined(HAMR_ENABLE_CUDA)
        || (alloc == buffer_allocator::cuda)
        || (alloc == buffer_allocator::cuda_uva)
#endif
#if defined(HAMR_ENABLE_HIP)
        || (alloc == buffer_allocator::hip)
        || (alloc == buffer_allocator::hip_uva)
#endif
#if defined(HAMR_ENABLE_OPENMP)
        || (alloc == buffer_allocator::openmp)
#endif
        );
}

/// get the allocator type most suitable for the current build configuration
inline HAMR_EXPORT buffer_allocator get_device_allocator()
{
#if defined(HAMR_ENABLE_CUDA)
    return buffer_allocator::cuda;
#endif
#if defined(HAMR_ENABLE_HIP)
    return buffer_allocator::hip;
#endif
#if defined(HAMR_ENABLE_OPENMP)
    return buffer_allocator::openmp;
#endif
    return buffer_allocator::malloc;
}

}

#endif
