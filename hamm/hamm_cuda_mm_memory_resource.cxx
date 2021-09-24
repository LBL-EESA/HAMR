#include "hamm_cuda_mm_memory_resource.h"
#include <iostream>

#if !defined(HAMM_ENABLE_CUDA)
// --------------------------------------------------------------------------
p_hamm_cuda_mm_memory_resource hamm_cuda_mm_memory_resource::New()
{
    std::cerr << "[" << __FILE__ << ":" << __LINE__
        << "] Error:" << "Failed to allocate memory because CUDA is not available"
        << std::endl;

    return nullptr;
}

// --------------------------------------------------------------------------
p_hamm_memory_resource hamm_cuda_mm_memory_resource::new_instance() const
{
    std::cerr << "[" << __FILE__ << ":" << __LINE__
        << "] Error:" << "Failed to allocate memory because CUDA is not available"
        << std::endl;

    return nullptr;
}

// --------------------------------------------------------------------------
void *hamm_cuda_mm_memory_resource::do_allocate(std::size_t n_bytes, std::size_t align)
{
    (void) n_bytes;
    (void) align;

    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] Error: "
        << "Failed to allocate memory because CUDA is not available"
        << std::endl;

    return nullptr;
}

// --------------------------------------------------------------------------
void hamm_cuda_mm_memory_resource::do_deallocate(void *ptr, std::size_t n_bytes,
    std::size_t align)
{
    (void) ptr;
    (void) n_bytes;
    (void) align;

    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] Error: "
        << "Failed to de-allocate memory because CUDA is not available"
        << std::endl;
}

// --------------------------------------------------------------------------
bool hamm_cuda_mm_memory_resource::do_is_equal(const hamm_pmr_memory_resource& other) const noexcept
{
    (void) other;
    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] Error: "
        << "Failed to compare resource because CUDA is not available"
        << std::endl;

    return false;
}
#endif
