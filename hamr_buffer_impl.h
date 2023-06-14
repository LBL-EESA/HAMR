#ifndef buffer_impl_h
#define buffer_impl_h

#include "hamr_config.h"
#include "hamr_env.h"
#include "hamr_malloc_allocator.h"
#include "hamr_new_allocator.h"
#include "hamr_host_copy.h"
#if defined(HAMR_ENABLE_CUDA)
#include "hamr_cuda_device.h"
#include "hamr_cuda_malloc_allocator.h"
#include "hamr_cuda_malloc_async_allocator.h"
#include "hamr_cuda_malloc_uva_allocator.h"
#include "hamr_cuda_malloc_host_allocator.h"
#include "hamr_cuda_print.h"
//#include "hamr_cuda_copy.h"
#include "hamr_cuda_copy_async.h"
#endif
#if defined(HAMR_ENABLE_HIP)
#include "hamr_hip_device.h"
#include "hamr_hip_malloc_allocator.h"
#include "hamr_hip_malloc_uva_allocator.h"
#include "hamr_hip_print.h"
#include "hamr_hip_copy.h"
#endif
#if defined(HAMR_ENABLE_OPENMP)
#include "hamr_openmp_device.h"
#include "hamr_openmp_allocator.h"
#include "hamr_openmp_print.h"
#include "hamr_openmp_copy.h"
#endif
#include "hamr_buffer_allocator.h"
#include "hamr_buffer_transfer.h"
#include "hamr_stream.h"

#include <memory>
#include <iostream>

/// heterogeneous accelerator memory resource
namespace hamr
{

// --------------------------------------------------------------------------
template <typename T>
int buffer<T>::set_owner()
{
    // host backed memory
    m_owner = -1;

#if defined(HAMR_ENABLE_CUDA)
    if (((m_alloc == allocator::cuda) ||
        (m_alloc == allocator::cuda_async) || (m_alloc == allocator::cuda_uva))
        && hamr::get_active_cuda_device(m_owner))
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to get the active CUDA device." << std::endl;
        return -1;
    }
#endif
#if defined(HAMR_ENABLE_HIP)
    if (((m_alloc == allocator::hip) || (m_alloc == allocator::hip_uva))
        && hamr::get_active_hip_device(m_owner))
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to get the active HIP device." << std::endl;
        return -1;
    }
#endif
#if defined(HAMR_ENABLE_OPENMP)
    if ((m_alloc == allocator::openmp)
        && hamr::get_active_openmp_device(m_owner))
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to get the active OpenMP device." << std::endl;
        return -1;
    }
#endif

    return 0;
}

// --------------------------------------------------------------------------
template <typename T>
int buffer<T>::set_owner(const T *ptr)
{
    (void) ptr;

    // host backed memory
    m_owner = -1;

#if defined(HAMR_ENABLE_CUDA)
    if ((m_alloc == allocator::cuda) ||
        (m_alloc == allocator::cuda_async) || (m_alloc == allocator::cuda_uva))
    {
        if (get_cuda_device(ptr, m_owner))
        {
            std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
                " Failed to determine device ownership for " << ptr << std::endl;
            return -1;
        }
    }
#endif
#if defined(HAMR_ENABLE_HIP)
    if ((m_alloc == allocator::hip) || (m_alloc == allocator::hip_uva))
    {
        if (get_hip_device(ptr, m_owner))
        {
            std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
                " Failed to determine device ownership for " << ptr << std::endl;
            return -1;
        }
    }
#endif
#if defined(HAMR_ENABLE_OPENMP)
    if (m_alloc == allocator::openmp)
    {
        // TODO -- is it possible to look up the device on which the
        // pointer resides?
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to determine device ownership for " << ptr << std::endl;
        return -1;
    }
#endif

    return 0;
}

// --------------------------------------------------------------------------
template <typename T>
buffer<T>::buffer(allocator alloc, const hamr::stream &strm, transfer sync) :
    m_alloc(alloc), m_data(nullptr), m_size(0), m_capacity(0), m_owner(-1),
    m_stream(strm), m_sync(sync)
{
    assert_valid_allocator(alloc);
    this->set_owner();
}

// --------------------------------------------------------------------------
template <typename T>
buffer<T>::buffer(allocator alloc, const hamr::stream &strm,
    transfer sync, size_t n_elem) : buffer<T>(alloc, strm, sync)
{
    m_data = this->allocate(n_elem);
    m_size = n_elem;
    m_capacity = n_elem;
}

// --------------------------------------------------------------------------
template <typename T>
buffer<T>::buffer(allocator alloc, const hamr::stream &strm,
    transfer sync, size_t n_elem, const T &val) : buffer<T>(alloc, strm, sync)
{
    m_data = this->allocate(n_elem, val);
    m_size = n_elem;
    m_capacity = n_elem;
}

// --------------------------------------------------------------------------
template <typename T>
buffer<T>::buffer(allocator alloc, const hamr::stream &strm,
    transfer sync, size_t n_elem, const T *vals) : buffer<T>(alloc, strm, sync)
{
    m_data = this->allocate(n_elem, vals);
    m_size = n_elem;
    m_capacity = n_elem;
}

// --------------------------------------------------------------------------
template <typename T>
buffer<T>::buffer(allocator alloc, const hamr::stream &strm, transfer sync,
    size_t size, int owner, const std::shared_ptr<T> &data) : m_alloc(alloc),
    m_data(data), m_size(size), m_capacity(size), m_owner(owner),
    m_stream(strm), m_sync(sync)

{
    assert_valid_allocator(alloc);

    // query the driver api to determine the owner
#if defined(HAMR_ENABLE_CUDA)
    if (((alloc == allocator::cuda) || (m_alloc == allocator::cuda_async) ||
        (alloc == allocator::cuda_uva)) && (m_owner < 0))
    {
        this->set_owner(data.get());
    }
#endif
#if defined(HAMR_ENABLE_HIP)
    if (((alloc == allocator::hip) ||
        (alloc == allocator::hip_uva)) && (m_owner < 0))
    {
        this->set_owner(data.get());
    }
#endif
#if defined(HAMR_ENABLE_OPENMP)
    if ((alloc == allocator::openmp) && (m_owner < 0))
    {
        //this->set_owner(data.get());
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " The owner must be set explicitly for OpenMP device memory"
            << std::endl;
        abort();
    }
#endif
}

// --------------------------------------------------------------------------
template <typename T>
template <typename delete_func_t>
buffer<T>::buffer(allocator alloc, const hamr::stream &strm, transfer sync,
    size_t size, int owner, T *ptr, delete_func_t df) : m_alloc(alloc),
    m_data(std::shared_ptr<T>(ptr, df)), m_size(size), m_capacity(size),
    m_owner(owner), m_stream(strm), m_sync(sync)
{
    assert_valid_allocator(alloc);

    // query the driver api to determine the owner
#if defined(HAMR_ENABLE_CUDA)
    if (((alloc == allocator::cuda) || (m_alloc == allocator::cuda_async) ||
        (alloc == allocator::cuda_uva)) && (m_owner < 0))
    {
        this->set_owner(ptr);
    }
#endif
#if defined(HAMR_ENABLE_HIP)
    if (((alloc == allocator::hip) ||
        (alloc == allocator::hip_uva)) && (m_owner < 0))
    {
        this->set_owner(ptr);
    }
#endif
#if defined(HAMR_ENABLE_OPENMP)
    if ((alloc == allocator::openmp) && (m_owner < 0))
    {
        //this->set_owner(data.get());
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " The owner must be set explicitly for OpenMP device memory"
            << std::endl;
        abort();
    }
#endif
}

// --------------------------------------------------------------------------
template <typename T>
buffer<T>::buffer(allocator alloc, const hamr::stream &strm, transfer sync,
    size_t size, int owner, T *ptr) : m_alloc(alloc), m_data(nullptr),
    m_size(size), m_capacity(size), m_owner(owner), m_stream(strm),
    m_sync(sync)
{
    assert_valid_allocator(alloc);

    // create the deleter for the passed allocator
    if (alloc == allocator::cpp)
    {
        m_data = std::shared_ptr<T>(ptr, new_deleter<T>(ptr, m_size));
    }
    else if (alloc == allocator::malloc)
    {
        m_data = std::shared_ptr<T>(ptr, malloc_deleter<T>(ptr, m_size));
    }
#if defined(HAMR_ENABLE_CUDA)
    else if ((alloc == allocator::cuda_async) ||
        ((alloc == allocator::cuda) && (m_stream != cudaStreamDefault) &&
        (m_stream != cudaStreamLegacy) && (m_stream != cudaStreamPerThread)))
    {
        // using a stream with cuda_malloc_allocator should forward to the
        // cuda_malloc_async_allocator
        m_data = std::shared_ptr<T>(ptr,
            cuda_malloc_async_deleter<T>(m_stream, ptr, m_size));
    }
    else if (alloc == allocator::cuda)
    {
        m_data = std::shared_ptr<T>(ptr,
            cuda_malloc_deleter<T>(ptr, m_size));
    }
    else if (alloc == allocator::cuda_uva)
    {
        m_data = std::shared_ptr<T>(ptr,
            cuda_malloc_uva_deleter<T>(m_stream, ptr, m_size));
    }
    else if (alloc == allocator::cuda_host)
    {
        m_data = std::shared_ptr<T>(ptr,
            cuda_malloc_host_deleter<T>(ptr, m_size));
    }
#endif
#if defined(HAMR_ENABLE_HIP)
    else if (alloc == allocator::hip)
    {
        m_data = std::shared_ptr<T>(ptr, hip_malloc_deleter<T>(ptr, m_size));
    }
    else if (alloc == allocator::hip_uva)
    {
        m_data = std::shared_ptr<T>(ptr, hip_malloc_uva_deleter<T>(ptr, m_size));
    }
#endif
#if defined(HAMR_ENABLE_OPENMP)
    else if (alloc == allocator::openmp)
    {
        m_data = std::shared_ptr<T>(ptr, openmp_deleter<T>(ptr, m_size, owner));
    }
#endif
    else
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Invalid allocator type " << get_allocator_name(m_alloc)
            << std::endl;
    }

    // set the owner
#if defined(HAMR_ENABLE_CUDA)
    if (((alloc == allocator::cuda) ||
        (alloc == allocator::cuda_uva)) && (m_owner < 0))
    {
        this->set_owner(ptr);
    }
#endif
#if defined(HAMR_ENABLE_HIP)
    if (((alloc == allocator::hip) ||
        (alloc == allocator::hip_uva)) && (m_owner < 0))
    {
        this->set_owner(ptr);
    }
#endif
#if defined(HAMR_ENABLE_OPENMP)
    if ((alloc == allocator::openmp) && (m_owner < 0))
    {
        //this->set_owner(data.get());
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " The owner must be set explicitly for OpenMP device memory"
            << std::endl;
        abort();
    }
#endif
}

// --------------------------------------------------------------------------
template <typename T>
buffer<T>::buffer(const buffer<T> &other) :
    buffer<T>(other.m_alloc, other.m_stream, other.m_sync, other)
{
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
buffer<T>::buffer(const buffer<U> &other) :
    buffer<T>(other.m_alloc, other.m_stream, other.m_sync, other)
{
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
buffer<T>::buffer(allocator alloc, const hamr::stream &strm, transfer sync,
    const buffer<U> &other) : buffer<T>(alloc, strm, sync, other.m_size)
{
    if (this->set(0, other, 0, m_size))
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Copy constructor failed to copy data from the other object."
            << std::endl;
        abort();
    }
}

// --------------------------------------------------------------------------
template <typename T>
buffer<T>::buffer(buffer<T> &&other) : buffer<T>(other.m_alloc)
{
    this->swap(other);
}

// --------------------------------------------------------------------------
template <typename T>
buffer<T>::buffer(allocator alloc, const hamr::stream &strm, transfer sync,
    buffer<T> &&other) : buffer<T>(alloc, strm, sync)
{
    if ((m_alloc == other.m_alloc) && (m_owner == other.m_owner))
    {
        std::swap(m_data, other.m_data);
        std::swap(m_size, other.m_size);
        std::swap(m_capacity, other.m_capacity);
    }
    else
    {
        this->assign(other);
    }
}

// --------------------------------------------------------------------------
template <typename T>
void buffer<T>::operator=(buffer<T> &&other)
{
    if ((m_alloc == other.m_alloc) && (m_owner == other.m_owner))
    {
        std::swap(m_data, other.m_data);
        std::swap(m_size, other.m_size);
        std::swap(m_capacity, other.m_capacity);
    }
    else
    {
        this->assign(other);
    }
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
void buffer<T>::operator=(const buffer<U> &other)
{
    this->assign(other);
}

// --------------------------------------------------------------------------
template <typename T>
void buffer<T>::operator=(const buffer<T> &other)
{
    this->assign(other);
}

// --------------------------------------------------------------------------
template <typename T>
void buffer<T>::swap(buffer<T> &other)
{
    std::swap(m_alloc, other.m_alloc);
    std::swap(m_data, other.m_data);
    std::swap(m_size, other.m_size);
    std::swap(m_capacity, other.m_capacity);
    std::swap(m_owner, other.m_owner);
    std::swap(m_stream, other.m_stream);
    std::swap(m_sync, other.m_sync);
}

// --------------------------------------------------------------------------
template <typename T>
int buffer<T>::get_active_device(int &dev_id)
{
    if ((m_alloc == allocator::malloc) ||
        (m_alloc == allocator::cpp) || (m_alloc == allocator::cuda_host))
    {
        dev_id = -1;
        return 0;
    }
#if defined(HAMR_ENABLE_CUDA)
    else if ((m_alloc == allocator::cuda) ||
        (m_alloc == allocator::cuda_async) || (m_alloc == allocator::cuda_uva))
    {
        return hamr::get_active_cuda_device(dev_id);
    }
#endif
#if defined(HAMR_ENABLE_HIP)
    else if ((m_alloc == allocator::hip) || (m_alloc == allocator::hip_uva))
    {
        return hamr::get_active_hip_device(dev_id);
    }
#endif
#if defined(HAMR_ENABLE_OPENMP)
    else if (m_alloc == allocator::openmp)
    {
        return hamr::get_active_openmp_device(dev_id);
    }
#endif

    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " Invalid allocator type " << get_allocator_name(m_alloc)
        << std::endl;

    dev_id = 0;
    return -1;
}

// --------------------------------------------------------------------------
template <typename T>
int buffer<T>::move(allocator alloc)
{
    // get the active device, this is the new owner
    int owner = -1;
    if (this->get_active_device(owner))
        return -1;

    // we don't need to do anything if both the new allocator
    // and the new owner match the current allocator and owner
    if ((alloc == m_alloc) && (owner == m_owner))
        return 0;

    // construct a temporary using the new allocator
    buffer<T> tmp(alloc, m_stream, m_sync, m_size);

    // copy the data to the temporary
    if (tmp.set(0, *this, 0, m_size))
        return -1;

    // swap internals
    this->swap(tmp);

    return 0;
}

// --------------------------------------------------------------------------
template <typename T>
int buffer<T>::host_accessible() const
{
    return hamr::host_accessible(m_alloc);
}

// --------------------------------------------------------------------------
template <typename T>
int buffer<T>::cuda_accessible() const
{
    return hamr::cuda_accessible(m_alloc);
}

// --------------------------------------------------------------------------
template <typename T>
int buffer<T>::hip_accessible() const
{
    return hamr::hip_accessible(m_alloc);
}

// --------------------------------------------------------------------------
template <typename T>
int buffer<T>::openmp_accessible() const
{
    return hamr::openmp_accessible(m_alloc);
}

// --------------------------------------------------------------------------
template <typename T>
int buffer<T>::device_accessible() const
{
#if defined(HAMR_ENABLE_CUDA)
    return hamr::cuda_accessible(m_alloc);
#elif defined(HAMR_ENABLE_HIP)
    return hamr::hip_accessible(m_alloc);
#elif defined(HAMR_ENABLE_OPENMP)
    return hamr::openmp_accessible(m_alloc);
#else
    return false;
#endif
}

// --------------------------------------------------------------------------
template <typename T>
std::shared_ptr<T> buffer<T>::allocate(size_t n_elem, const T &val)
{
    if (m_alloc == allocator::cpp)
    {
        return new_allocator<T>::allocate(n_elem, val);
    }
    else if (m_alloc == allocator::malloc)
    {
        return malloc_allocator<T>::allocate(n_elem, val);
    }
#if defined(HAMR_ENABLE_CUDA)
    else if (m_alloc == allocator::cuda)
    {
        return cuda_malloc_allocator<T>::allocate(m_stream, n_elem, val);
    }
    else if (m_alloc == allocator::cuda_async)
    {
        return cuda_malloc_async_allocator<T>::allocate(m_stream, n_elem, val);
    }
    else if (m_alloc == allocator::cuda_uva)
    {
        return cuda_malloc_uva_allocator<T>::allocate(m_stream, n_elem, val);
    }
    else if (m_alloc == allocator::cuda_host)
    {
        return cuda_malloc_host_allocator<T>::allocate(n_elem, val);
    }
#endif
#if defined(HAMR_ENABLE_HIP)
    else if (m_alloc == allocator::hip)
    {
        return hip_malloc_allocator<T>::allocate(n_elem, val);
    }
    else if (m_alloc == allocator::hip_uva)
    {
        return hip_malloc_uva_allocator<T>::allocate(n_elem, val);
    }
#endif
#if defined(HAMR_ENABLE_OPENMP)
    else if (m_alloc == allocator::openmp)
    {
        return openmp_allocator<T>::allocate(n_elem, val);
    }
#endif

    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " Invalid allocator type " << get_allocator_name(m_alloc)
        << std::endl;

    return nullptr;
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
std::shared_ptr<T> buffer<T>::allocate(size_t n_elem, const U *vals)
{
    if (m_alloc == allocator::cpp)
    {
        return new_allocator<T>::allocate(n_elem, vals);
    }
    else if (m_alloc == allocator::malloc)
    {
        return malloc_allocator<T>::allocate(n_elem, vals);
    }
#if defined(HAMR_ENABLE_CUDA)
    else if (m_alloc == allocator::cuda)
    {
        activate_cuda_device dev(m_owner);
        return cuda_malloc_allocator<T>::allocate(
            m_stream, n_elem, vals);
    }
    else if (m_alloc == allocator::cuda_async)
    {
        activate_cuda_device dev(m_owner);
        return cuda_malloc_async_allocator<T>::allocate(
            m_stream, n_elem, vals);
    }
    else if (m_alloc == allocator::cuda_uva)
    {
        activate_cuda_device dev(m_owner);
        return cuda_malloc_uva_allocator<T>::allocate(
            m_stream, n_elem, vals);
    }
    else if (m_alloc == allocator::cuda_host)
    {
        return cuda_malloc_host_allocator<T>::allocate(n_elem, vals);
    }
#endif
#if defined(HAMR_ENABLE_HIP)
    else if (m_alloc == allocator::hip)
    {
        activate_hip_device dev(m_owner);
        return hip_malloc_allocator<T>::allocate(n_elem, vals);
    }
    else if (m_alloc == allocator::hip_uva)
    {
        activate_hip_device dev(m_owner);
        return hip_malloc_uva_allocator<T>::allocate(n_elem, vals);
    }
#endif
#if defined(HAMR_ENABLE_OPENMP)
    else if (m_alloc == allocator::openmp)
    {
        activate_openmp_device dev(m_owner);
        return openmp_allocator<T>::allocate(n_elem, vals);
    }
#endif

    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " Invalid allocator type " << get_allocator_name(m_alloc)
        << std::endl;

    return nullptr;
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
std::shared_ptr<T> buffer<T>::allocate(const buffer<U> &vals)
{
    // TODO -- this implementation fails when the source and dest are on
    // different GPUs.

    size_t n_elem = vals.size();

    if (m_alloc == allocator::cpp)
    {
        std::shared_ptr<const U> pvals = vals.get_host_accessible();

        // a deep copy was made, return the pointer to the copy
        if (std::is_same<T,U>::value && !vals.host_accessible())
            return std::const_pointer_cast<T>(pvals);

        return new_allocator<T>::allocate(n_elem, pvals.get());
    }
    else if (m_alloc == allocator::malloc)
    {
        std::shared_ptr<const U> pvals = vals.get_host_accessible();

        // a deep copy was made, return the pointer to the copy
        if (std::is_same<T,U>::value && !vals.host_accessible())
            return std::const_pointer_cast<T>(pvals);

        return malloc_allocator<T>::allocate(n_elem, pvals.get());
    }
#if defined(HAMR_ENABLE_CUDA)
    else if (m_alloc == allocator::cuda)
    {
        activate_cuda_device dev(m_owner);
        std::shared_ptr<const U> pvals = vals.get_cuda_accessible();

        // a deep copy was made, return the pointer to the copy
        if (std::is_same<T,U>::value &&
            (!vals.cuda_accessible() || (vals.m_owner != m_owner)))
            return std::const_pointer_cast<T>(pvals);

        return cuda_malloc_allocator<T>::allocate(
            m_stream, n_elem, pvals.get(), true);
    }
    else if (m_alloc == allocator::cuda_async)
    {
        activate_cuda_device dev(m_owner);
        std::shared_ptr<const U> pvals = vals.get_cuda_accessible();

        // a deep copy was made, return the pointer to the copy
        if (std::is_same<T,U>::value &&
            (!vals.cuda_accessible() || (vals.m_owner != m_owner)))
            return std::const_pointer_cast<T>(pvals);

        return cuda_malloc_async_allocator<T>::allocate(
            m_stream, n_elem, pvals.get(), true);
    }
    else if (m_alloc == allocator::cuda_uva)
    {
        activate_cuda_device dev(m_owner);
        std::shared_ptr<const U> pvals = vals.get_cuda_accessible();

        // a deep copy was made, return the pointer to the copy
        if (std::is_same<T,U>::value &&
            (!vals.cuda_accessible() || (vals.m_owner != m_owner)))
            return  std::const_pointer_cast<T>(pvals);

        return cuda_malloc_uva_allocator<T>::allocate(
            m_stream, n_elem, pvals.get(), true);
    }
    else if (m_alloc == allocator::cuda_host)
    {
        std::shared_ptr<const U> pvals = vals.get_host_accessible();

        // a deep copy was made, return the pointer to the copy
        if (std::is_same<T,U>::value && !vals.host_accessible())
            return std::const_pointer_cast<T>(pvals);

        return cuda_malloc_host_allocator<T>::allocate(n_elem, pvals.get());
    }
#endif
#if defined(HAMR_ENABLE_HIP)
    else if (m_alloc == allocator::hip)
    {
        activate_hip_device dev(m_owner);
        std::shared_ptr<const U> pvals = vals.get_hip_accessible();

        // a deep copy was made, return the pointer to the copy
        if (std::is_same<T,U>::value &&
            (!vals.hip_accessible() || (vals.m_owner != m_owner)))
            return std::const_pointer_cast<T>(pvals);

        return hip_malloc_allocator<T>::allocate(n_elem, pvals.get(), true);
    }
    else if (m_alloc == allocator::hip_uva)
    {
        activate_hip_device dev(m_owner);
        std::shared_ptr<const U> pvals = vals.get_hip_accessible();

        // a deep copy was made, return the pointer to the copy
        if (std::is_same<T,U>::value &&
            (!vals.hip_accessible() || (vals.m_owner != m_owner)))
            return  std::const_pointer_cast<T>(pvals);

        return hip_malloc_uva_allocator<T>::allocate(n_elem, pvals.get(), true);
    }
#endif
#if defined(HAMR_ENABLE_OPENMP)
    else if (m_alloc == allocator::openmp)
    {
        activate_openmp_device dev(m_owner);
        std::shared_ptr<const U> pvals = vals.get_openmp_accessible();

        // a deep copy was made, return the pointer to the copy
        if (std::is_same<T,U>::value &&
            (!vals.openmp_accessible() || (vals.m_owner != m_owner)))
            return std::const_pointer_cast<T>(pvals);

        return openmp_allocator<T>::allocate(n_elem, pvals.get(), true);
    }
#endif

    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " Invalid allocator type "
        << get_allocator_name(m_alloc) << std::endl;

    return nullptr;
}

// --------------------------------------------------------------------------
template <typename T>
std::shared_ptr<T> buffer<T>::allocate(size_t n_elem)
{
    if (m_alloc == allocator::cpp)
    {
        return new_allocator<T>::allocate(n_elem);
    }
    else if (m_alloc == allocator::malloc)
    {
        return malloc_allocator<T>::allocate(n_elem);
    }
#if defined(HAMR_ENABLE_CUDA)
    else if (m_alloc == allocator::cuda)
    {
        activate_cuda_device dev(m_owner);
        return cuda_malloc_allocator<T>::allocate(n_elem);
    }
    else if (m_alloc == allocator::cuda_async)
    {
        activate_cuda_device dev(m_owner);
        return cuda_malloc_async_allocator<T>::allocate(m_stream, n_elem);
    }
    else if (m_alloc == allocator::cuda_uva)
    {
        activate_cuda_device dev(m_owner);
        return cuda_malloc_uva_allocator<T>::allocate(m_stream, n_elem);
    }
    else if (m_alloc == allocator::cuda_host)
    {
        return cuda_malloc_host_allocator<T>::allocate(n_elem);
    }
#endif
#if defined(HAMR_ENABLE_HIP)
    else if (m_alloc == allocator::hip)
    {
        activate_hip_device dev(m_owner);
        return hip_malloc_allocator<T>::allocate(n_elem);
    }
    else if (m_alloc == allocator::hip_uva)
    {
        activate_hip_device dev(m_owner);
        return hip_malloc_uva_allocator<T>::allocate(n_elem);
    }
#endif
#if defined(HAMR_ENABLE_OPENMP)
    else if (m_alloc == allocator::openmp)
    {
        activate_openmp_device dev(m_owner);
        return openmp_allocator<T>::allocate(n_elem);
    }
#endif

    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " Invalid allocator type "
        << get_allocator_name(m_alloc) << std::endl;

    return nullptr;
}

// --------------------------------------------------------------------------
template <typename T>
int buffer<T>::reserve(size_t n_elem)
{
    // already have enough memory
    if ((n_elem  == 0) || (m_capacity >= n_elem))
        return 0;

    // do not have enough memory
    // allocate space
    std::shared_ptr<T> tmp;
    if (!(tmp = this->allocate(n_elem)))
        return -1;

    // copy existing elements
    if (m_size)
    {
        int ierr = 0;
        if ((m_alloc == allocator::cpp) ||
            (m_alloc == allocator::malloc) || (m_alloc == allocator::cuda_host))
        {
            ierr = copy_to_host_from_host(tmp.get(), m_data.get(), m_size);
        }
#if defined(HAMR_ENABLE_CUDA)
        else if ((m_alloc == allocator::cuda) ||
            (m_alloc == allocator::cuda_async) || (m_alloc == allocator::cuda_uva))
        {
            activate_cuda_device dev(m_owner);
            ierr = copy_to_cuda_from_cuda(m_stream, tmp.get(), m_data.get(), m_size);
        }
#endif
#if defined(HAMR_ENABLE_HIP)
        else if ((m_alloc == allocator::hip) || (m_alloc == allocator::hip_uva))
        {
            activate_hip_device dev(m_owner);
            ierr = copy_to_hip_from_hip(tmp.get(), m_data.get(), m_size);
        }
#endif
#if defined(HAMR_ENABLE_OPENMP)
        else if (m_alloc == allocator::openmp)
        {
            activate_openmp_device dev(m_owner);
            ierr = copy_to_openmp_from_openmp(tmp.get(), m_data.get(), m_size);
        }
#endif
        else
        {
            std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
                " Invalid allocator type "
                << get_allocator_name(m_alloc) << std::endl;
        }

        // check for errors
        if (ierr)
            return -1;
    }

    // update state
    m_capacity = n_elem;
    m_data = tmp;

    return 0;
}

// --------------------------------------------------------------------------
template <typename T>
int buffer<T>::reserve(size_t n_elem, const T &val)
{
    // already have enough memory
    if ((n_elem  == 0) || (m_capacity >= n_elem))
        return 0;

    // do not have enough memory
    // allocate space
    std::shared_ptr<T> tmp;
    if (!(tmp = this->allocate(n_elem, val)))
        return -1;

    // copy existing elements
    if (m_size)
    {
        int ierr = 0;
        if ((m_alloc == allocator::cpp) ||
            (m_alloc == allocator::malloc) || (m_alloc == allocator::cuda_host))
        {
            ierr = copy_to_host_from_host(tmp.get(), m_data.get(), m_size);
        }
#if defined(HAMR_ENABLE_CUDA)
        else if ((m_alloc == allocator::cuda) ||
            (m_alloc == allocator::cuda_async) ||(m_alloc == allocator::cuda_uva))
        {
            activate_cuda_device dev(m_owner);
            ierr = copy_to_cuda_from_cuda(m_stream,
                tmp.get(), m_data.get(), m_size);
        }
#endif
#if defined(HAMR_ENABLE_HIP)
        else if ((m_alloc == allocator::hip) || (m_alloc == allocator::hip_uva))
        {
            activate_hip_device dev(m_owner);
            ierr = copy_to_hip_from_hip(tmp.get(), m_data.get(), m_size);
        }
#endif
#if defined(HAMR_ENABLE_OPENMP)
        else if (m_alloc == allocator::openmp)
        {
            activate_openmp_device dev(m_owner);
            ierr = copy_to_openmp_from_openmp(tmp.get(), m_data.get(), m_size);
        }
#endif
        else
        {
            std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
                " Invalid allocator type " << get_allocator_name(m_alloc)
                << std::endl;
        }

        // check for errors
        if (ierr)
            return -1;
    }

    // update state
    m_capacity = n_elem;
    m_data = tmp;

    return 0;
}

// --------------------------------------------------------------------------
template <typename T>
int buffer<T>::resize(size_t n_elem)
{
    // allocate space
    if (this->reserve(n_elem))
        return -1;

    // update the size
    m_size = n_elem;

    return 0;
}

// --------------------------------------------------------------------------
template <typename T>
int buffer<T>::resize(size_t n_elem, const T &val)
{
    // allocate space
    if (this->reserve(n_elem, val))
        return -1;

    // update the size
    m_size = n_elem;

    return 0;
}

// --------------------------------------------------------------------------
template <typename T>
int buffer<T>::free()
{
    m_data = nullptr;
    m_size = 0;
    m_capacity = 0;
    m_owner = -1;
    return 0;
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
int buffer<T>::assign(const buffer<U> &src)
{
    size_t n_vals = src.size();

    // allocate space if needed
    if (this->resize(n_vals))
        return -1;

    // copy the values
    if (this->set(0, src, 0, n_vals))
        return -1;

    return 0;
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
int buffer<T>::assign(const buffer<U> &src, size_t src_start, size_t n_vals)
{
    // allocate space if needed
    if (this->resize(n_vals))
        return -1;

    // copy the values
    if (this->set(0, src, src_start, n_vals))
        return -1;

    return 0;
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
int buffer<T>::assign(const U *src, size_t src_start, size_t n_vals)
{
    // allocate space if needed
    if (this->resize(n_vals))
        return -1;

    // copy the values
    if (this->set(0, src, src_start, n_vals))
        return -1;

    return 0;
}

// --------------------------------------------------------------------------
template <typename T>
int buffer<T>::reserve_for_append(size_t n_vals)
{
    if (n_vals)
    {
        size_t new_size = m_size + n_vals;
        size_t new_capacity = m_capacity;
        if (new_size > new_capacity)
        {

            if (new_capacity == 0)
                new_capacity = 8;

            while (new_size > new_capacity)
                new_capacity *= 2;

            if (this->reserve(new_capacity))
                return -1;

            m_capacity = new_capacity;
        }
    }
    return 0;
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
int buffer<T>::append(const U *src, size_t src_start, size_t n_vals)
{
    // source is always on the host
    if (n_vals)
    {
        // allocate space if needed
        if (this->reserve_for_append(n_vals))
            return -1;

        // get the append location
        size_t back = m_size;

        // update state
        m_size += n_vals;

        // copy the value to the back
        if (this->set(back, src, src_start, n_vals))
            return -1;
    }
    return 0;
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
int buffer<T>::append(const buffer<U> &src, size_t src_start, size_t n_vals)
{
    if (n_vals)
    {
        // allocate space if needed
        if (this->reserve_for_append(n_vals))
            return -1;

        // get the append location
        size_t back = m_size;

        // update state
        m_size += n_vals;

        // copy the value to the back.
        if (this->set(back, src, src_start, n_vals))
            return -1;
    }
    return 0;
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
int buffer<T>::append(const buffer<U> &src)
{
    if (this->append(src, 0, src.size()))
        return -1;

    return 0;
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
int buffer<T>::set(size_t dest_start, const U *src,
    size_t src_start, size_t n_vals)
{
    if (n_vals)
    {
        // bounds check
        assert(m_size >= (dest_start + n_vals));

        // copy the values (src is always on the host)
        int ierr = 0;
        if ((m_alloc == allocator::cpp) ||
            (m_alloc == allocator::malloc) || (m_alloc == allocator::cuda_host))
        {
            ierr = copy_to_host_from_host(m_data.get() + dest_start,
                src + src_start, n_vals);
        }
#if defined(HAMR_ENABLE_CUDA)
        else if ((m_alloc == allocator::cuda) ||
            (m_alloc == allocator::cuda_async) || (m_alloc == allocator::cuda_uva))
        {
            activate_cuda_device dev(m_owner);

            ierr = copy_to_cuda_from_host(m_stream, m_data.get() + dest_start,
                src + src_start, n_vals);
        }
#endif
#if defined(HAMR_ENABLE_HIP)
        else if ((m_alloc == allocator::hip) || (m_alloc == allocator::hip_uva))
        {

            activate_hip_device dev(m_owner);

            ierr = copy_to_hip_from_host(m_data.get() + dest_start,
                src + src_start, n_vals);
        }
#endif
#if defined(HAMR_ENABLE_OPENMP)
        else if (m_alloc == allocator::openmp)
        {

            activate_openmp_device dev(m_owner);

            ierr = copy_to_openmp_from_host(m_data.get() + dest_start,
                src + src_start, n_vals);
        }
#endif
        else
        {
            std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
                " Invalid allocator type " << get_allocator_name(m_alloc)
                << std::endl;
        }

        // synchronize
        if (m_sync == transfer::sync)
            m_stream.synchronize();

        // check for errors
        if (ierr)
            return -1;
    }

    return 0;
}

// ---------------------------------------------------------------------------
template <typename T>
template <typename U>
int buffer<T>::set(size_t dest_start, const buffer<U> &src,
    size_t src_start, size_t n_vals)
{
    if (n_vals)
    {
        // bounds check
        assert(m_size >= (dest_start + n_vals));
        assert(src.size() >= (src_start + n_vals));

        // copy the value to the back. buffers can either be on the host or GPU
        // and use different technologies so all permutations must be realized.
        int ierr = 0;
        if ((m_alloc == allocator::cpp) ||
            (m_alloc == allocator::malloc) || (m_alloc == allocator::cuda_host))
        {
            // destination is on the host

            if ((src.m_alloc == allocator::cpp) ||
                (src.m_alloc == allocator::malloc) ||
                (src.m_alloc == allocator::cuda_host))
            {
                // source is on the host
                ierr = copy_to_host_from_host(m_data.get() + dest_start,
                    src.m_data.get() + src_start, n_vals);
            }
#if defined(HAMR_ENABLE_CUDA)
            else if ((src.m_alloc == allocator::cuda) ||
                (src.m_alloc == allocator::cuda_async) || (src.m_alloc == allocator::cuda_uva))
            {
                // source is on the GPU
                activate_cuda_device dev(src.m_owner);

                ierr = copy_to_host_from_cuda(m_stream,
                    m_data.get() + dest_start, src.m_data.get() + src_start,
                    n_vals);

                // synchronize
                if ((m_sync == transfer::sync_host) || (m_sync == transfer::sync))
                    m_stream.synchronize();
            }
#endif
#if defined(HAMR_ENABLE_HIP)
            else if ((src.m_alloc == allocator::hip) ||
                (src.m_alloc == allocator::hip_uva))
            {
                // source is on the GPU
                activate_hip_device dev(src.m_owner);

                ierr = copy_to_host_from_hip(m_data.get() + dest_start,
                    src.m_data.get() + src_start, n_vals);


                // synchronize
                if ((m_sync == transfer::sync_host) || (m_sync == transfer::sync))
                    m_stream.synchronize();
            }
#endif
#if defined(HAMR_ENABLE_OPENMP)
            else if (src.m_alloc == allocator::openmp)
            {
                // source is on the GPU
                activate_openmp_device dev(src.m_owner);

                ierr = copy_to_host_from_openmp(m_data.get() + dest_start,
                    src.m_data.get() + src_start, n_vals);

                // synchronize
                if ((m_sync == transfer::sync_host) || (m_sync == transfer::sync))
                    m_stream.synchronize();
            }
#endif
            else
            {
                std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
                    " Invalid allocator type in the source "
                    << get_allocator_name(src.m_alloc) << std::endl;
            }
        }
#if defined(HAMR_ENABLE_CUDA)
        else if ((m_alloc == allocator::cuda) ||
            (m_alloc == allocator::cuda_async) || (m_alloc == allocator::cuda_uva))
        {
            // destination is on the GPU
            activate_cuda_device dev(m_owner);

            if ((src.m_alloc == allocator::cpp) ||
                (src.m_alloc == allocator::malloc) ||
                (src.m_alloc == allocator::cuda_host))
            {
                // source is on the host
                ierr = copy_to_cuda_from_host(m_stream,
                    m_data.get() + dest_start, src.m_data.get() + src_start, n_vals);
            }
            else if (src.cuda_accessible())
            {
                if (m_owner == src.m_owner)
                {
                    // source is on this GPU
                    ierr = copy_to_cuda_from_cuda(m_stream,
                        m_data.get() + dest_start, src.m_data.get() + src_start,
                        n_vals);
                }
                else
                {
                    // source is on another GPU
                    ierr = copy_to_cuda_from_cuda(m_stream,
                        m_data.get() + dest_start, src.m_data.get() + src_start,
                        src.m_owner, n_vals);
                }
            }
            else
            {
                std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
                    " Invalid allocator type in the source "
                    << get_allocator_name(src.m_alloc) << std::endl;
            }

            // synchronize
            if (m_sync == transfer::sync)
                m_stream.synchronize();
        }
#endif
#if defined(HAMR_ENABLE_HIP)
        else if ((m_alloc == allocator::hip) || (m_alloc == allocator::hip_uva))
        {
            // destination is on the GPU
            activate_hip_device dev(m_owner);

            if ((src.m_alloc == allocator::cpp) ||
                (src.m_alloc == allocator::malloc) ||
                (src.m_alloc == allocator::cuda_host))

            {
                // source is on the host
                ierr = copy_to_hip_from_host(m_data.get() + dest_start,
                    src.m_data.get() + src_start, n_vals);
            }
            else if (src.hip_accessible())
            {
                if (m_owner == src.m_owner)
                {
                    // source is on this GPU
                    ierr = copy_to_hip_from_hip(m_data.get() + dest_start,
                        src.m_data.get() + src_start, n_vals);
                }
                else
                {
                    // source is on another GPU
                    ierr = copy_to_hip_from_hip(m_data.get() + dest_start,
                        src.m_data.get() + src_start, src.m_owner, n_vals);
                }
            }
            else
            {
                std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
                    " Invalid allocator type in the source "
                    << get_allocator_name(src.m_alloc) << std::endl;
            }

            // synchronize
            if (m_sync == transfer::sync)
                m_stream.synchronize();
        }
#endif
#if defined(HAMR_ENABLE_OPENMP)
        else if (m_alloc == allocator::openmp)
        {
            // destination is on the GPU
            activate_openmp_device dev(m_owner);

            if ((src.m_alloc == allocator::cpp) ||
                (src.m_alloc == allocator::malloc) ||
                (src.m_alloc == allocator::cuda_host))
            {
                // source is on the host
                ierr = copy_to_openmp_from_host(m_data.get() + dest_start,
                    src.m_data.get() + src_start, n_vals);
            }
            else if (src.openmp_accessible())
            {
                if (m_owner == src.m_owner)
                {
                    // source is on this GPU
                    ierr = copy_to_openmp_from_openmp(m_data.get() + dest_start,
                        src.m_data.get() + src_start, n_vals);
                }
                else
                {
                    // source is on another GPU
                    ierr = copy_to_openmp_from_openmp(m_data.get() + dest_start,
                        src.m_data.get() + src_start, src.m_owner, n_vals);
                }
            }
            else
            {
                std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
                    " Invalid allocator type in the source "
                    << get_allocator_name(src.m_alloc) << std::endl;
            }

            // synchronize
            if (m_sync == transfer::sync)
                m_stream.synchronize();
        }
#endif
        else
        {
            std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
                " Invalid allocator type "
                << get_allocator_name(m_alloc) << std::endl;
        }

        // check for errors
        if (ierr)
            return -1;
    }

    return 0;
}

// ---------------------------------------------------------------------------
template <typename T>
template <typename U>
int buffer<T>::get(size_t src_start, U *dest,
    size_t dest_start, size_t n_vals) const
{
    if (n_vals)
    {
        // bounds check
        assert(m_size >= (src_start + n_vals));

        // copy the values (dest is always on the host)
        int ierr = 0;
        if ((m_alloc == allocator::cpp) ||
            (m_alloc == allocator::malloc) || (m_alloc == allocator::cuda_host))
        {
            ierr = copy_to_host_from_host(dest + dest_start,
                m_data.get() + src_start, n_vals);
        }
#if defined(HAMR_ENABLE_CUDA)
        else if ((m_alloc == allocator::cuda) ||
            (m_alloc == allocator::cuda_async) || (m_alloc == allocator::cuda_uva))
        {
            activate_cuda_device dev(m_owner);

            ierr = copy_to_host_from_cuda(m_stream,
                dest + dest_start, m_data.get() + src_start, n_vals);

            // synchronize
            if ((m_sync == transfer::sync_host) || (m_sync == transfer::sync))
                m_stream.synchronize();
        }
#endif
#if defined(HAMR_ENABLE_HIP)
        else if ((m_alloc == allocator::hip) || (m_alloc == allocator::hip_uva))
        {
            activate_hip_device dev(m_owner);

            ierr = copy_to_host_from_hip(dest + dest_start,
                m_data.get() + src_start, n_vals);

            // synchronize
            if ((m_sync == transfer::sync_host) || (m_sync == transfer::sync))
                m_stream.synchronize();
        }
#endif
#if defined(HAMR_ENABLE_OPENMP)
        else if (m_alloc == allocator::openmp)
        {
            activate_openmp_device dev(m_owner);

            ierr = copy_to_host_from_openmp(dest + dest_start,
                m_data.get() + src_start, n_vals);

            // synchronize
            if ((m_sync == transfer::sync_host) || (m_sync == transfer::sync))
                m_stream.synchronize();
        }
#endif
        else
        {
            std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
                " Invalid allocator type "
                << get_allocator_name(m_alloc) << std::endl;
        }

        // check for errors
        if (ierr)
            return -1;
    }

    return 0;
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
int buffer<T>::get(size_t src_start,
    buffer<U> &dest, size_t dest_start, size_t n_vals) const
{
    if (n_vals)
    {
        // bounds check
        assert(m_size >= (src_start + n_vals));
        assert(dest.size() >= (dest_start + n_vals));

        // copy the value to the back. buffers can either be on the host or GPU
        // and use different technologies so all permutations must be realized.
        int ierr = 0;
        if ((m_alloc == allocator::cpp) ||
            (m_alloc == allocator::malloc) || (m_alloc == allocator::malloc))
        {
            // destination is on the host

            if ((dest.m_alloc == allocator::cpp) ||
                (dest.m_alloc == allocator::malloc) ||
                (dest.m_alloc == allocator::cuda_host))
            {
                // source is on the host
                ierr = copy_to_host_from_host(dest.m_data.get() + dest_start,
                    m_data.get() + src_start, n_vals);
            }
#if defined(HAMR_ENABLE_CUDA)
            else if ((dest.m_alloc == allocator::cuda) ||
                (dest.m_alloc == allocator::cuda_async) || (dest.m_alloc == allocator::cuda_uva))
            {
                // source is on the GPU
                activate_cuda_device dev(m_owner);

                ierr = copy_to_host_from_cuda(m_stream,
                    dest.m_data.get() + dest_start, m_data.get() + src_start,
                    n_vals);

                // synchronize
                if ((m_sync == transfer::sync_host) || (m_sync == transfer::sync))
                    m_stream.synchronize();
            }
#endif
#if defined(HAMR_ENABLE_HIP)
            else if ((dest.m_alloc == allocator::hip) ||
                (dest.m_alloc == allocator::hip_uva))
            {
                // source is on the GPU
                activate_hip_device dev(m_owner);

                ierr = copy_to_host_from_hip(dest.m_data.get() + dest_start,
                    m_data.get() + src_start, n_vals);

                // synchronize
                if ((m_sync == transfer::sync_host) || (m_sync == transfer::sync))
                    m_stream.synchronize();
            }
#endif
#if defined(HAMR_ENABLE_OPENMP)
            else if (dest.m_alloc == allocator::openmp)
            {
                // source is on the GPU
                activate_openmp_device dev(m_owner);

                ierr = copy_to_host_from_openmp(dest.m_data.get() + dest_start,
                    m_data.get() + src_start, n_vals);

                // synchronize
                if ((m_sync == transfer::sync_host) || (m_sync == transfer::sync))
                    m_stream.synchronize();
            }
#endif
            else
            {
                std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
                    " Invalid allocator type in the source "
                    << get_allocator_name(dest.m_alloc) << std::endl;
            }
        }
#if defined(HAMR_ENABLE_CUDA)
        else if ((m_alloc == allocator::cuda) ||
            (m_alloc == allocator::cuda_async) || (m_alloc == allocator::cuda_uva))
        {
            // destination is on the GPU
            activate_cuda_device dev(dest.m_owner);

            if ((dest.m_alloc == allocator::cpp) ||
                (dest.m_alloc == allocator::malloc) ||
                (dest.m_alloc == allocator::cuda_host))
            {
                // source is on the host
                ierr = copy_to_cuda_from_host(m_stream,
                    dest.m_data.get() + dest_start, m_data.get() + src_start,
                    n_vals);
            }
            else if ((dest.m_alloc == allocator::cuda) ||
                (dest.m_alloc == allocator::cuda_async) || (dest.m_alloc == allocator::cuda_uva))
            {
                if (m_owner == dest.m_owner)
                {
                    // source is on this GPU
                    ierr = copy_to_cuda_from_cuda(m_stream,
                        dest.m_data.get() + dest_start, m_data.get() + src_start,
                        n_vals);
                }
                else
                {
                    // source is on another GPU
                    ierr = copy_to_cuda_from_cuda(m_stream,
                        dest.m_data.get() + dest_start,
                        m_data.get() + src_start, m_owner, n_vals);
                }
            }
            else
            {
                std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
                    " Transfers from " << get_allocator_name(m_alloc) << " to "
                    << get_allocator_name(dest.m_alloc) << " not yet implemented."
                    << std::endl;
            }

            // synchronize
            if (m_sync == transfer::sync)
                m_stream.synchronize();
        }
#endif
#if defined(HAMR_ENABLE_HIP)
        else if ((m_alloc == allocator::hip) ||
            (m_alloc == allocator::hip_uva))
        {
            // destination is on the GPU
            activate_hip_device dev(dest.m_owner);

            if ((dest.m_alloc == allocator::cpp) ||
                (dest.m_alloc == allocator::malloc) ||
                (dest.m_alloc == allocator::cuda_host))
            {
                // source is on the host
                ierr = copy_to_hip_from_host(dest.m_data.get() + dest_start,
                    m_data.get() + src_start, n_vals);
            }
            else if ((dest.m_alloc == allocator::hip) ||
                (dest.m_alloc == allocator::hip_uva))
            {
                if (m_owner == dest.m_owner)
                {
                    // source is on this GPU
                    ierr = copy_to_hip_from_hip(dest.m_data.get() + dest_start,
                        m_data.get() + src_start, n_vals);
                }
                else
                {
                    // source is on another GPU
                    ierr = copy_to_hip_from_hip(dest.m_data.get() + dest_start,
                        m_data.get() + src_start, m_owner, n_vals);
                }
            }
            else
            {
                std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
                    " Transfers from " << get_allocator_name(m_alloc) << " to "
                    << get_allocator_name(dest.m_alloc) << " not yet implemented."
                    << std::endl;
            }

            // synchronize
            if (m_sync == transfer::sync)
                m_stream.synchronize();
        }
#endif
#if defined(HAMR_ENABLE_OPENMP)
        else if (m_alloc == allocator::openmp)
        {
            // destination is on the GPU
            activate_openmp_device dev(dest.m_owner);

            if ((dest.m_alloc == allocator::cpp) ||
                (dest.m_alloc == allocator::malloc) ||
                (dest.m_alloc == allocator::cuda_host))
            {
                // source is on the host
                ierr = copy_to_openmp_from_host(dest.m_data.get() + dest_start,
                    m_data.get() + src_start, n_vals);
            }
            else if (dest.m_alloc == allocator::openmp)
            {
                if (m_owner == dest.m_owner)
                {
                    // source is on this GPU
                    ierr = copy_to_openmp_from_openmp(dest.m_data.get() + dest_start,
                        m_data.get() + src_start, n_vals);
                }
                else
                {
                    // source is on another GPU
                    ierr = copy_to_openmp_from_openmp(dest.m_data.get() + dest_start,
                        m_data.get() + src_start, m_owner, n_vals);
                }
            }
            else
            {
                std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
                    " Transfers from " << get_allocator_name(m_alloc) << " to "
                    << get_allocator_name(dest.m_alloc) << " not yet implemented."
                    << std::endl;
            }

            // synchronize
            if (m_sync == transfer::sync)
                m_stream.synchronize();
        }
#endif
        else
        {
            std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
                " Invalid allocator type "
                << get_allocator_name(m_alloc) << std::endl;
        }

        // check for errors
        if (ierr)
            return -1;
    }

    return 0;
}

// ---------------------------------------------------------------------------
template <typename T>
std::shared_ptr<const T> buffer<T>::get_host_accessible() const
{
    if ((m_alloc == allocator::cpp) || (m_alloc == allocator::malloc) ||
        (m_alloc == allocator::cuda_uva) || (m_alloc == allocator::cuda_host) ||
        (m_alloc == allocator::hip_uva))
    {
        // already on the host
        return m_data;
    }
#if defined(HAMR_ENABLE_CUDA)
    else if ((m_alloc == allocator::cuda) || (m_alloc == allocator::cuda_async))
    {
        // make a copy on the host
        std::shared_ptr<T> tmp = cuda_malloc_host_allocator<T>::allocate(m_size);
        if (!tmp)
        {
            std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
                " CUDA failed to allocate host pinned memory, falling back"
                " to the default system allocator." << std::endl;
            tmp = malloc_allocator<T>::allocate(m_size);
        }

        activate_cuda_device dev(m_owner);

        if (copy_to_host_from_cuda(m_stream, tmp.get(), m_data.get(), m_size))
            return nullptr;

        // synchronize
        if ((m_sync == transfer::sync_host) || (m_sync == transfer::sync))
            m_stream.synchronize();

        return tmp;
    }
#endif
#if defined(HAMR_ENABLE_HIP)
    else if (m_alloc == allocator::hip)
    {
        // make a copy on the host
        std::shared_ptr<T> tmp = malloc_allocator<T>::allocate(m_size);

        activate_hip_device dev(m_owner);

        if (copy_to_host_from_hip(tmp.get(), m_data.get(), m_size))
            return nullptr;

        // synchronize
        if ((m_sync == transfer::sync_host) || (m_sync == transfer::sync))
            m_stream.synchronize();

        return tmp;
    }
#endif
#if defined(HAMR_ENABLE_OPENMP)
    else if (m_alloc == allocator::openmp)
    {
        // make a copy on the host
        std::shared_ptr<T> tmp = malloc_allocator<T>::allocate(m_size);

        activate_openmp_device dev(m_owner);

        if (copy_to_host_from_openmp(tmp.get(), m_data.get(), m_size))
            return nullptr;

        // synchronize
        if ((m_sync == transfer::sync_host) || (m_sync == transfer::sync))
            m_stream.synchronize();

        return tmp;
    }
#endif
    else
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Invalid allocator type " << get_allocator_name(m_alloc)
            << std::endl;
    }

    return nullptr;
}

// ---------------------------------------------------------------------------
template <typename T>
std::shared_ptr<const T> buffer<T>::get_cuda_accessible() const
{
#if !defined(HAMR_ENABLE_CUDA)
    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " get_cuda_accessible failed, CUDA is not available."
        << std::endl;
    return nullptr;
#else
    if ((m_alloc == allocator::cpp) ||
        (m_alloc == allocator::malloc) || (m_alloc == allocator::cuda_host))
    {
        // make a copy on the GPU
        std::shared_ptr<T> tmp = cuda_malloc_async_allocator<T>::
            allocate(m_stream, m_size);

        if (copy_to_cuda_from_host(m_stream,
            tmp.get(), m_data.get(), m_size))
            return nullptr;

        // synchronize
        if (m_sync == transfer::sync)
            m_stream.synchronize();

        return tmp;
    }
    else if ((m_alloc == allocator::cuda) ||
        (m_alloc == allocator::cuda_async) || (m_alloc == allocator::cuda_uva))
    {
        int dest_device = 0;
        if (hamr::get_active_cuda_device(dest_device))
            return nullptr;

        if (m_owner == dest_device)
        {
            // already on this GPU
            return m_data;
        }
        else
        {
            // on another GPU, move to this one
            std::shared_ptr<T> tmp = cuda_malloc_async_allocator<T>
                ::allocate(m_stream, m_size);

            if (copy_to_cuda_from_cuda(m_stream,
                tmp.get(), m_data.get(), m_owner, m_size))
                return nullptr;

            // synchronize
            if (m_sync == transfer::sync)
                m_stream.synchronize();

            return tmp;
        }
    }
#if defined(HAMR_ENABLE_OPENMP)
    else if (m_alloc == allocator::openmp)
    {
        int dest_device = 0;
        if (hamr::get_active_cuda_device(dest_device))
            return nullptr;

        if (m_owner == dest_device)
        {
            // already on this GPU
            return m_data;
        }
        else
        {
            // on another GPU, move to this one
            std::shared_ptr<T> tmp = openmp_allocator<T>::allocate(m_size);

            if (copy_to_openmp_from_openmp(tmp.get(), m_data.get(), m_owner, m_size))
                return nullptr;

            // synchronize
            if (m_sync == transfer::sync)
                m_stream.synchronize();

            return tmp;
        }
    }
#endif
    else
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Transfers from " << get_allocator_name(m_alloc) << " to "
            << get_allocator_name(allocator::cuda) << " not yet implemented."
            << std::endl;
    }

    return nullptr;
#endif
}

// ---------------------------------------------------------------------------
template <typename T>
std::shared_ptr<const T> buffer<T>::get_hip_accessible() const
{
#if !defined(HAMR_ENABLE_HIP)
    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " get_hip_accessible failed, HIP is not available."
        << std::endl;
    return nullptr;
#else
    if ((m_alloc == allocator::cpp) ||
        (m_alloc == allocator::malloc) || (m_alloc == allocator::cuda_host))
    {
        // make a copy on the GPU
        std::shared_ptr<T> tmp = hip_malloc_allocator<T>::allocate(m_size);

        if (copy_to_hip_from_host(tmp.get(), m_data.get(), m_size))
            return nullptr;

        // synchronize
        if (m_sync == transfer::sync)
            m_stream.synchronize();

        return tmp;
    }
    else if ((m_alloc == allocator::hip) || (m_alloc == allocator::hip_uva))
    {
        int dest_device = 0;
        if (hamr::get_active_hip_device(dest_device))
            return nullptr;

        if (m_owner == dest_device)
        {
            // already on this GPU
            return m_data;
        }
        else
        {
            // on another GPU, move to this one
            std::shared_ptr<T> tmp = hip_malloc_allocator<T>::allocate(m_size);

            if (copy_to_hip_from_hip(tmp.get(), m_data.get(), m_owner, m_size))
                return nullptr;

            // synchronize
            if (m_sync == transfer::sync)
                m_stream.synchronize();

            return tmp;
        }
    }
    else
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Transfers from " << get_allocator_name(m_alloc) << " to "
            << get_allocator_name(allocator::hip) << " not yet implemented."
            << std::endl;
    }

    return nullptr;
#endif
}

// ---------------------------------------------------------------------------
template <typename T>
std::shared_ptr<const T> buffer<T>::get_openmp_accessible() const
{
#if !defined(HAMR_ENABLE_OPENMP)
    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " get_openmp_accessible failed, OpenMP is not available."
        << std::endl;
    return nullptr;
#else
    if ((m_alloc == allocator::cpp) ||
        (m_alloc == allocator::malloc) || (m_alloc == allocator::cuda_host))
    {
        // make a copy on the GPU
        std::shared_ptr<T> tmp = openmp_allocator<T>::allocate(m_size);

        if (copy_to_openmp_from_host(tmp.get(), m_data.get(), m_size))
            return nullptr;

        // synchronize
        if (m_sync == transfer::sync)
            m_stream.synchronize();

        return tmp;
    }
    else if (m_alloc == allocator::openmp)
    {
        int dest_device = 0;
        if (hamr::get_active_openmp_device(dest_device))
            return nullptr;

        if (m_owner == dest_device)
        {
            // already on this GPU
            return m_data;
        }
        else
        {
            // on another GPU, move to this one
            std::shared_ptr<T> tmp = openmp_allocator<T>::allocate(m_size);

            if (copy_to_openmp_from_openmp(tmp.get(), m_data.get(), m_owner, m_size))
                return nullptr;

            // synchronize
            if (m_sync == transfer::sync)
                m_stream.synchronize();

            return tmp;
        }
    }
#if defined(HAMR_ENABLE_CUDA)
    else if ((m_alloc == allocator::cuda) ||
        (m_alloc == allocator::cuda_async) || (m_alloc == allocator::cuda_uva))
    {
        int dest_device = 0;
        if (hamr::get_active_openmp_device(dest_device))
            return nullptr;

        if (m_owner == dest_device)
        {
            // already on this GPU
            return m_data;
        }
        else
        {
            // on another GPU, move to this one
            std::shared_ptr<T> tmp = cuda_malloc_async_allocator<T>
                ::allocate(m_stream, m_size);

            if (copy_to_cuda_from_cuda(m_stream,
                tmp.get(), m_data.get(), m_owner, m_size))
                return nullptr;

            // synchronize
            if (m_sync == transfer::sync)
                m_stream.synchronize();

            return tmp;
        }
    }
#endif
    else
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Transfers from " << get_allocator_name(m_alloc) << " to "
            << get_allocator_name(allocator::openmp) << " not yet implemented."
            << std::endl;
    }

    return nullptr;
#endif
}

// ---------------------------------------------------------------------------
template <typename T>
std::shared_ptr<const T> buffer<T>::get_device_accessible() const
{
#if defined(HAMR_ENABLE_CUDA)
    return get_cuda_accessible();
#elif defined(HAMR_ENABLE_HIP)
    return get_hip_accessible();
#elif defined(HAMR_ENABLE_OPENMP)
    return get_openmp_accessible();
#else
    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " get_device_accessible failed, No device technology is available"
        " in this build." << std::endl;
    return nullptr;
#endif
}

// --------------------------------------------------------------------------
template <typename T>
int buffer<T>::print() const
{
    std::cerr << "m_alloc = " << get_allocator_name(m_alloc)
        << ", m_size = " << m_size << ", m_capacity = " << m_capacity
        << ", m_data = ";

    if (m_size)
    {
        if ((m_alloc == allocator::cpp) || (m_alloc == allocator::malloc) ||
            (m_alloc == allocator::cuda_host) || (m_alloc == allocator::cuda_uva) ||
            (m_alloc == allocator::hip_uva))
        {
            std::cerr << m_data.get()[0];
            for (size_t i = 1; i < m_size; ++i)
                std::cerr << ", " << m_data.get()[i];
            std::cerr << std::endl;
        }
#if defined(HAMR_ENABLE_CUDA)
        else if ((m_alloc == allocator::cuda) || (m_alloc == allocator::cuda_async))
        {
            activate_cuda_device dev(m_owner);
            cuda_print(m_stream, m_data.get(), m_size);
        }
#endif
#if defined(HAMR_ENABLE_HIP)
        else if (m_alloc == allocator::hip)
        {
            activate_hip_device dev(m_owner);
            hip_print(m_data.get(), m_size);
        }
#endif
#if defined(HAMR_ENABLE_OPENMP)
        else if (m_alloc == allocator::openmp)
        {
            activate_openmp_device dev(m_owner);
            openmp_print(m_data.get(), m_size);
        }
#endif
        else
        {
            std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
                " Invalid allocator type " << get_allocator_name(m_alloc)
                << std::endl;
        }
    }

    return 0;
}

}
#endif
