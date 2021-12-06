#ifndef hamr_buffer_allocator_h
#define hamr_buffer_allocator_h

///@file

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

}

#endif
