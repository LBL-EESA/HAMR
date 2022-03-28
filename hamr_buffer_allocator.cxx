#include "hamr_buffer_allocator.h"

namespace hamr
{

// **************************************************************************
const char *get_allocator_name(buffer_allocator alloc)
{
    if (alloc == buffer_allocator::cpp)
    {
        return "cpp";
    }
    else if (alloc == buffer_allocator::malloc)
    {
        return "malloc";
    }
    else if (alloc == buffer_allocator::cuda)
    {
        return "cuda_malloc_allocator";
    }
    else if (alloc == buffer_allocator::cuda_uva)
    {
        return "cuda_malloc_uva_allocator";
    }
    else if (alloc == buffer_allocator::hip)
    {
        return "hip_malloc_allocator";
    }
    else if (alloc == buffer_allocator::hip_uva)
    {
        return "hip_malloc_uva_allocator";
    }
    else if (alloc == buffer_allocator::openmp)
    {
        return "openmp_allocator";
    }

    return "the allocator name is not known";
}

}
