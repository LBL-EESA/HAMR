#include "hamr_cuda_memory_resource.h"

#include <iostream>

namespace hamr
{

// --------------------------------------------------------------------------
p_cuda_memory_resource cuda_memory_resource::New()
{
    return std::shared_ptr<cuda_memory_resource>(new cuda_memory_resource);
}

// --------------------------------------------------------------------------
p_memory_resource cuda_memory_resource::new_instance() const
{
    return std::shared_ptr<cuda_memory_resource>(new cuda_memory_resource);
}

// --------------------------------------------------------------------------
void *cuda_memory_resource::do_allocate(std::size_t n_bytes, std::size_t align)
{
    (void) align;

    void *ptr = nullptr;

    cudaError_t ierr = cudaMalloc(&ptr, n_bytes, cudaMemAttachGlobal);
    if (ierr != cudaSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] Error:"
            << "Failed to allocate " << n_bytes << " of CUDA managed memory. "
            << cudaGetErrorString(ierr) << std::endl;

        throw std::bad_alloc();
    }

    if (this->verbose > 1)
    {
        std::cerr << "cuda_memory_resource(" << this << ") allocated " << n_bytes
            << " alligned to " << align << " byte boundary at " << ptr << std::endl;
    }

    return ptr;
}

// --------------------------------------------------------------------------
void cuda_memory_resource::do_deallocate(void *ptr, std::size_t n_bytes,
    std::size_t align)
{
    (void) n_bytes;
    (void) align;

    cudaError_t ierr = cudaFree(ptr);

    if (ierr != cudaSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] Error:"
            << "Failed to free " << n_bytes << " of CUDA managed memory at "
            << ptr << ". " << cudaGetErrorString(ierr) << std::endl;
    }

    if (this->verbose > 1)
    {
        std::cerr << "cuda_memory_resource(" << this << ") deallocated " << n_bytes
            << " alligned to " << align << " byte boundary  at " << ptr << std::endl;
    }
}

// --------------------------------------------------------------------------
bool cuda_memory_resource::do_is_equal(const pmr_memory_resource& other) const noexcept
{
    return dynamic_cast<const cuda_memory_resource*>(&other) != nullptr;
}

}
