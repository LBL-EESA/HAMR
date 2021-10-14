#ifndef hamr_memory_resource_h
#define hamr_memory_resource_h

#include "hamr_config.h"
#include "hamr_pmr_memory_resource.h"

#include <memory>

/// heterogeneous accelerator memory resource
namespace hamr
{

class memory_resource;

/// A shared pointer to an hmm::memory_resource instance
using p_memory_resource = std::shared_ptr<hamr::memory_resource>;

/// The base class for heterogeneous memory model resources
/** Defines the APIs for * determining the technologies compatible with this
 * resource. Each supported technology has an accisible method that can be
 * used to verify if the memory is accessible from a supported technology. The
 * list of supported technologies are: CUDA, CPU. Derived classes must implement
 * std::pmr::memory_resource.
 */
class memory_resource : public
    hamr::pmr_memory_resource,
    std::enable_shared_from_this<memory_resource>
{
public:
    memory_resource() : verbose(0) {}
    virtual ~memory_resource() {}

    /// return a new instance of the same type of resource
    virtual p_memory_resource new_instance() const = 0;

    /// returns true if the memory can be used by code running on the CPU
    virtual bool cpu_accessible() const { return false; }

    /// returns true if the memory can be use on the GPU from CUDA
    virtual bool cuda_accessible() const { return false; }

    /// return the name of the class
    virtual const char *get_class_name() const = 0;

    /// return the current verbosity level
    int get_verbose() { return this->verbose; }

    /// set the verbosity level
    void set_verbose(int val) { this->verbose = val; }

protected:
    int verbose;
};

}

#endif
