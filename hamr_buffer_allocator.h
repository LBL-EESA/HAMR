#ifndef hamr_buffer_allocator_h
#define hamr_buffer_allocator_h

///@file

#include "hamr_config.h"

namespace hamr
{

/// allocator types that may be used with hamr::buffer
enum class buffer_allocator
{
    same = -2,   ///< propagate the current allocator
    none = -1,   ///< no allocator specified
    cpp = 0,     ///< allocates memory with new
    malloc = 1,  ///< allocates memory with malloc
    cuda = 2,    ///< allocates memory with cudaMalloc
    cuda_uva = 3 ///< allocates memory with cudaMallocManaged
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
        (alloc == buffer_allocator::cuda_uva);
}

/// @returns true if the allocator creates CUDA accessible memory
inline
HAMR_EXPORT
int cuda_accessible(buffer_allocator alloc)
{
    return (alloc == buffer_allocator::cuda) ||
        (alloc == buffer_allocator::cuda_uva);
}

}

#endif
