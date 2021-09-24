#include "hamm_cpu_memory_resource.h"

#include <iostream>

// --------------------------------------------------------------------------
p_hamm_cpu_memory_resource hamm_cpu_memory_resource::New()
{
    return std::shared_ptr<hamm_cpu_memory_resource>(new hamm_cpu_memory_resource);
}

// --------------------------------------------------------------------------
p_hamm_memory_resource hamm_cpu_memory_resource::new_instance() const
{
    return std::shared_ptr<hamm_cpu_memory_resource>(new hamm_cpu_memory_resource);
}

// --------------------------------------------------------------------------
void *hamm_cpu_memory_resource::do_allocate(std::size_t n_bytes, std::size_t align)
{
    void *ptr = aligned_alloc(align, n_bytes);
    if (ptr == 0)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] Error: "
            << "Failed to allocate " << n_bytes << " aligned to "
            << align << " bytes" << std::endl;

        throw std::bad_alloc();
    }

    if (this->verbose > 1)
    {
        std::cerr << "hamm_cpu_memory_resource(" << this << ") allocated " << n_bytes
            << " alligned to " << align << " bytes at " << ptr << std::endl;
    }

    return ptr;
}

// --------------------------------------------------------------------------
void hamm_cpu_memory_resource::do_deallocate(void *ptr, std::size_t n_bytes,
    std::size_t align)
{
    (void) n_bytes;
    (void) align;

    free(ptr);

    if (this->verbose > 1)
    {
        std::cerr << "hamm_cpu_memory_resource(" << this << ") deallocated "
            << n_bytes << std::endl;
    }
}

// --------------------------------------------------------------------------
bool hamm_cpu_memory_resource::do_is_equal(const hamm_pmr_memory_resource& other) const noexcept
{
    return dynamic_cast<const hamm_cpu_memory_resource*>(&other) != nullptr;
}
