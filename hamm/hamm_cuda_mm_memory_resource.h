#ifndef hamm_cuda_mm_memory_resource_h
#define hamm_cuda_mm_memory_resource_h

#include "hamm_config.h"
#include "hamm_memory_resource.h"

#include <memory>

class hamm_cuda_mm_memory_resource;

/// shared poointer to an instance of hmm::hamm_cuda_mm_memory_resource
using p_hamm_cuda_mm_memory_resource = std::shared_ptr<hamm_cuda_mm_memory_resource>;

/// A memory resource that manages memory for use both on the CPU and from CUDA
/** This resource uses CUDA UVA managed memory technology. The memory is
 * accessible from code running on the CPU and in CUDA. Implements
 * std::pmr::memory_resource.
 */
class hamm_cuda_mm_memory_resource : public hamm_memory_resource
{
public:
    /// return a new CUDA managed memory resource object
    static p_hamm_cuda_mm_memory_resource New();

    /// return a new instance of the same type of resource
    p_hamm_memory_resource new_instance() const override;

    /// returns true if the memory can be use on the CPU
    bool cpu_accessible() const override { return true; }

    /// returns true if the memory can be use on the GPU from CUDA
    bool cuda_accessible() const override { return true; }

    /// return the name of the class
    const char *get_class_name() const override
    { return "hmm::hamm_cuda_mm_memory_resource"; }

private:
    /// allocate memory for use on the CPU
    void *do_allocate(std::size_t n_bytes, std::size_t alignment) override;

    /// deallocate memory allocated for use on the CPU
    void do_deallocate(void *ptr, std::size_t n_bytes, std::size_t alignment) override;

    /// check for equality (eqaul if one can delete the memory allocated by the other)
    bool do_is_equal(const hamm_pmr_memory_resource& other) const noexcept override;
};

#endif
