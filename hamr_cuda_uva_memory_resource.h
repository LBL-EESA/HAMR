#ifndef hamr_cuda_uva_memory_resource_h
#define hamr_cuda_uva_memory_resource_h

#include "hamr_config.h"
#include "hamr_memory_resource.h"

#include <memory>

/// heterogeneous accelerator memory resource
namespace hamr
{

class cuda_uva_memory_resource;

/// shared poointer to an instance of hmm::cuda_uva_memory_resource
using p_cuda_uva_memory_resource =
    std::shared_ptr<cuda_uva_memory_resource>;

/// A memory resource that manages memory for use both on the CPU and from CUDA
/** This resource uses CUDA UVA managed memory technology. The memory is
 * accessible from code running on the CPU and in CUDA. Implements
 * std::pmr::memory_resource.
 */
class HAMR_EXPORT cuda_uva_memory_resource : public hamr::memory_resource
{
public:
    /// return a new CUDA managed memory resource object
    static p_cuda_uva_memory_resource New();

    /// return a new instance of the same type of resource
    p_memory_resource new_instance() const override;

    /// returns true if the memory can be use on the CPU
    bool cpu_accessible() const override { return true; }

    /// returns true if the memory can be use on the GPU from CUDA
    bool cuda_accessible() const override { return true; }

    /// return the name of the class
    const char *get_class_name() const override
    { return "hmm::cuda_uva_memory_resource"; }

private:
    /// allocate memory for use on the CPU
    void *do_allocate(std::size_t n_bytes, std::size_t alignment) override;

    /// deallocate memory allocated for use on the CPU
    void do_deallocate(void *ptr, std::size_t n_bytes, std::size_t alignment) override;

    /// check for equality (eqaul if one can delete the memory allocated by the other)
    bool do_is_equal(const pmr_memory_resource& other) const noexcept override;
};

}
#endif
