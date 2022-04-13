#ifndef buffer_h
#define buffer_h

#include "hamr_config.h"
#include "hamr_env.h"
#include "hamr_malloc_allocator.h"
#include "hamr_new_allocator.h"
#include "hamr_cpu_copy.h"
#if defined(HAMR_ENABLE_CUDA)
#include "hamr_cuda_device.h"
#include "hamr_cuda_malloc_allocator.h"
#include "hamr_cuda_malloc_uva_allocator.h"
#include "hamr_cuda_print.h"
#include "hamr_cuda_copy.h"
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

#include <memory>
#include <iostream>

/// heterogeneous accelerator memory resource
namespace hamr
{

/** @brief A technology agnostic buffer that manages memory on CPUs, GPUs, and
 * accelerators.
 * @details The buffer mediates between different accelerator and platform
 * portability technologies' memory models. Examples of platform portability
 * technologies are HIP, OpenMP, OpenCL, SYCL, and Kokos, Examples of
 * accelerator technologies are CUDA and roc. Other accelerator and platform
 * portability technologies exist and can be supported. Data can be left in
 * place until it is consumed. The consumer of the data can get a pointer that
 * is accessible in the technology that will be used to process the data. If
 * the data is already accessible in that technology access is a NOOP,
 * otherwise the data will be moved such that it is accessible. Smart pointers
 * take care of destruction of temporary buffers if needed.
 */
template <typename T>
class HAMR_EXPORT buffer
{
public:
    /** An enumeration for the type of allocator to use for memory allocations.
     * See ::buffer_allocator. */
    using allocator = buffer_allocator;

    /// construct an empty buffer that will use the passed allocator type
    buffer(allocator alloc);

    /// construct a buffer with n_elem size using the passed allocator type
    buffer(allocator alloc, size_t n_elem);

    /** construct a buffer with n_elem size initialized to the passed value
     * using the passed allocator type
     */
    buffer(allocator alloc, size_t n_elem, const T &val);

    /** construct a buffer with n_elem size initialized to the passed value
     * using the passed allocator type
     */
    buffer(allocator alloc, size_t n_elem, const T *vals);

    /** construct by directly providing the buffer contents. This can be used
     * for zero-copy transfer of data.  One must also name the allocator type
     * and device owning the data.  In addition for new allocations the
     * allocator type and owner are used internally to know how to
     * automatically move data during inter technology transfers.
     *
     * @param[in] alloc a ::buffer_allocator indicating the technology
     *                  backing the pointer
     * @param[in] size  the number of elements in the array pointed to by ptr
     * @param[in] owner the device owning the memory, -1 for CPU. if the
     *                  allocator is a GPU allocator and -1 is passed the
     *                  driver API is used to determine the device that
     *                  allocated the memory.
     * @param[in] ptr   a pointer to the array
     * @param[in] df    a function `void df(void*ptr)` used to delete the array
     *                  when this instance is finished.
     */
    template <typename delete_func_t>
    buffer(allocator alloc, size_t size, int owner, T *ptr, delete_func_t df);

    /** construct by directly providing the buffer contents. This can be used
     * for zero-copy transfer of data.  One must also name the allocator type
     * and device owning the data.  In addition for new allocations the
     * allocator type and owner are used internally to know how to
     * automatically move data during inter technology transfers.
     * The pass ::buffer_allocator is used to create the deleter that will be
     * called when this instance is finished with the memeory. Use this
     * constructor to transfer ownership of the array.
     *
     * @param[in] alloc a ::buffer_allocator indicating the technology
     *                  backing the pointer
     * @param[in] size  the number of elements in the array pointed to by ptr
     * @param[in] owner the device owning the memory, -1 for CPU. if the
     *                  allocator is a GPU allocator and -1 is passed the
     *                  driver API is used to determine the device that
     *                  allocated the memory.
     * @param[in] ptr   a pointer to the array
     */
    buffer(allocator alloc, size_t size, int owner, T *ptr);

    /** construct by directly providing the buffer contents. This can be used
     * for zero-copy transfer of data.  One must also name the allocator type
     * and device owning the data.  In addition for new allocations the
     * allocator type and owner are used internally to know how to
     * automatically move data during inter technology transfers.
     *
     * @param[in] alloc a ::buffer_allocator indicating the technology
     *                  backing the pointer
     * @param[in] size  the number of elements in the array pointed to by ptr
     * @param[in] owner the device owning the memory, -1 for CPU. if the
     *                  allocator is a GPU allocator and -1 is passed the
     *                  driver API is used to determine the device that
     *                  allocated the memory.
     * @param[in] data  a shared pointer managing the data
     */
    buffer(allocator alloc, size_t size, int owner,
        const std::shared_ptr<T> &data);

    /// copy construct from the passed buffer
    buffer(const buffer<T> &other);

    /// copy construct from the passed buffer, using the passed allocator type.
    buffer(allocator alloc, const buffer<T> &other);

    /// move construct from the passed buffer
    buffer(buffer<T> &&other);

    /** assign from the other buffer. if this and the passed buffer have
     * different allocators this allocator is used and the data will be copied.
     * if this and the passed buffer have different types elements are
     * cast to this type as they are copied.
     */
    template <typename U>
    void operator=(const buffer<U> &other);

    /** move assign from the other buffer. if this and the passed buffer have
     * the same type and allocator the passed buffer is moved. if this and the
     * passed buffer have different allocators this allocator is used and the
     * data will be copied.  if this and the passed buffer have different types
     * elements are cast to this type as they are copied.
     */
    void operator=(buffer<T> &&other);

    /// swap the contents of the two buffers
    void swap(buffer<T> &other);

    /** @name reserve
     * allocates space for n_elems of data
     */
    ///@{
    /// reserve n_elem of memory
    int reserve(size_t n_elem);

    /// reserve n_elem of memory and initialize them to val
    int reserve(size_t n_elem, const T &val);
    ///@}

    /** @name resize
     * resizes storage for n_elems of data
     */
    ///@{
    /// resize the buffer to hold n_elem of memory
    int resize(size_t n_elem);

    /** resize the buffer to hold n_elem of memory and initialize new elements
     * to val */
    int resize(size_t n_elem, const T &val);
    ///@}

    /// free all internal storage
    int free();

    /// returns the number of elements of storage allocated to the buffer
    size_t size() const { return m_size; }

    /** @name assign
     * Copies data into the buffer resizing the buffer.
     */
    ///@{
    /// assign the range from the passed array (src is always on the CPU)
    template<typename U>
    int assign(const U *src, size_t src_start, size_t n_vals);

    /// assign the range from the passed buffer
    template<typename U>
    int assign(const buffer<U> &src, size_t src_start, size_t n_vals);

    /// assign the passed buffer
    template<typename U>
    int assign(const buffer<U> &src);
    ///@}


    /** @name append
     * insert values at the back of the buffer, growing as needed
     */
    ///@{
    /** appends n_vals from src starting at src_start to the end of the buffer,
     * extending the buffer as needed. (src is always on the CPU)
     */
    template <typename U>
    int append(const U *src, size_t src_start, size_t n_vals);

    /** appends n_vals from src starting at src_start to the end of the buffer,
     * extending the buffer as needed.
     */
    template <typename U>
    int append(const buffer<U> &src, size_t src_start, size_t n_vals);

    /** appends to the end of the buffer, extending the buffer as needed.
     */
    template <typename U>
    int append(const buffer<U> &src);
    ///@}


    /** @name set
     * sets a range of elements in the buffer
     */
    ///@{
    /** sets n_vals elements starting at dest_start from the passed buffer's
     * elements starting at src_start (src is always on the CPU)*/
    template <typename U>
    int set(size_t dest_start, const U *src, size_t src_start, size_t n_vals);

    /** sets n_vals elements starting at dest_start from the passed buffer's
     * elements starting at src_start */
    template <typename U>
    int set(const buffer<U> &src)
    {
        return this->set(0, src, 0, src.size());
    }

    /** sets n_vals elements starting at dest_start from the passed buffer's
     * elements starting at src_start */
    template <typename U>
    int set(size_t dest_start, const buffer<U> &src,
        size_t src_start, size_t n_vals);
    ///@}


    /** @name get
     * gets a range of values from the buffer
     */
    ///@{
    /** gets n_vals elements starting at src_start into the passed array
     * elements starting at dest_start (dest is always on the CPU)*/
    template <typename U>
    int get(size_t src_start, U *dest, size_t dest_start, size_t n_vals) const;

    /** gets n_vals elements starting at src_start into the passed buffer's
     * elements starting at dest_start */
    template <typename U>
    int get(size_t src_start, buffer<U> &dest,
        size_t dest_start, size_t n_vals) const;

    /** gets n_vals elements starting at src_start into the passed buffer's
     * elements starting at dest_start */
    template <typename U>
    int get(buffer<U> &dest) const
    {
        return this->get(0, dest, 0, this->size());
    }
    ///@}

#if !defined(SWIG)
    /** @name get_cpu_accessible
     * Returns a pointer to the contents of the buffer accessible on the CPU.
     * If the buffer is currently accessible by codes running on the CPU then
     * this call is a NOOP.  If the buffer is not currently accessible by codes
     * running on the CPU then a temporary buffer is allocated and the data is
     * moved to the CPU.  The returned shared_ptr deals with deallocation of
     * the temporary if needed.
     */
    ///@{
    /// returns a pointer to the contents of the buffer accessible on the CPU.
    std::shared_ptr<T> get_cpu_accessible();

    /// returns a pointer to the contents of the buffer accessible on the CPU.
    std::shared_ptr<const T> get_cpu_accessible() const;
    ///@}
#endif

    /// returns true if the data is accessible from codes running on the CPU
    int cpu_accessible() const;

#if !defined(SWIG)
    /** @name get_cuda_accessible
     * returns a pointer to the contents of the buffer accessible from the
     * active CUDA device.  If the buffer is currently accessible on the named
     * CUDA device then this call is a NOOP.  If the buffer is not currently
     * accessible on the named CUDA device then a temporary buffer is allocated
     * and the data is moved.  The returned shared_ptr deals with deallocation
     * of the temporary if needed.
     */
    ///@{
    ///  returns a pointer to the contents of the buffer accessible from within CUDA
    std::shared_ptr<T> get_cuda_accessible();

    ///  returns a pointer to the contents of the buffer accessible from within CUDA
    std::shared_ptr<const T> get_cuda_accessible() const;
    ///@}
#endif

    /// returns true if the data is accessible from CUDA codes
    int cuda_accessible() const;

#if !defined(SWIG)
    /** @name get_hip_accessible
     * returns a pointer to the contents of the buffer accessible from the
     * active HIP device.  If the buffer is currently accessible on the named
     * HIP device then this call is a NOOP.  If the buffer is not currently
     * accessible on the named HIP device then a temporary buffer is allocated
     * and the data is moved.  The returned shared_ptr deals with deallocation
     * of the temporary if needed.
     */
    ///@{
    ///  returns a pointer to the contents of the buffer accessible from within HIP
    std::shared_ptr<T> get_hip_accessible();

    ///  returns a pointer to the contents of the buffer accessible from within HIP
    std::shared_ptr<const T> get_hip_accessible() const;
    ///@}
#endif

    /// returns true if the data is accessible from HIP codes
    int hip_accessible() const;

#if !defined(SWIG)
    /** @name get_openmp_accessible returns a pointer to the contents of
     * the buffer accessible from the active OpenMP off load device.  If the
     * buffer is currently accessible on the named OpenMP off load device then
     * this call is a NOOP.  If the buffer is not currently accessible on the
     * named OpenMP off load device then a temporary buffer is allocated and
     * the data is moved.  The returned shared_ptr deals with deallocation of
     * the temporary if needed.
     */
    ///@{
    /** returns a pointer to the contents of the buffer accessible from within
     * OpenMP off load
     */
    std::shared_ptr<T> get_openmp_accessible();

    /** returns a pointer to the contents of the buffer accessible from within
     * OpenMP off load
     */
    std::shared_ptr<const T> get_openmp_accessible() const;
    ///@}
#endif

    /// returns true if the data is accessible from OpenMP off load codes
    int openmp_accessible() const;

    /** @name data
     * return the raw pointer to the buffer contents. Use this when you know
     * that the buffer contents are accessible by the code operating on them to
     * save the cost of a std::shared_ptr copy construct.
     */
    ///@{
    /// return a pointer to the buffer contents
    T *data() { return m_data.get(); }

    /// return a const pointer to the buffer contents
    const T *data() const { return m_data.get(); }
    ///@}

    /// returns the allocator type enum
    allocator get_allocator() const { return m_alloc; }

    /// prints the contents to the stderr stream
    int print() const;

protected:
    /// grow the buffer if needed. doubles in size
    int reserve_for_append(size_t n_vals);

    /// allocate space for n_elem
    std::shared_ptr<T> allocate(size_t n_elem);

    /// allocate space for n_elem initialized to val
    std::shared_ptr<T> allocate(size_t n_elem, const T &val);

    /// allocate space for n_elem initialized with an array of values
    template <typename U>
    std::shared_ptr<T> allocate(size_t n_elem, const U *vals);

    /// allocate space for n_elem initialized with an array of values
    template <typename U>
    std::shared_ptr<T> allocate(const buffer<U> &vals);

    /** set the device where the buffer is located to the active device or the
     * CPU. The allocator is used to determine which. @returns 0 if successful.
     */
    int set_owner();

    /** set the device where the buffer is located by querying the driver API or the
     * CPU. The allocator is used to determine which. @returns 0 if successful.
     */
    int set_owner(const T *ptr);


private:
    allocator m_alloc;
    std::shared_ptr<T> m_data;
    size_t m_size;
    size_t m_capacity;
    int m_owner;

    template<typename U> friend class buffer;
};


// --------------------------------------------------------------------------
template <typename T>
int buffer<T>::set_owner()
{
    // CPU backed memory
    m_owner = -1;

#if defined(HAMR_ENABLE_CUDA)
    if (((m_alloc == allocator::cuda) || (m_alloc == allocator::cuda_uva))
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

    // CPU backed memory
    m_owner = -1;

#if defined(HAMR_ENABLE_CUDA)
    if ((m_alloc == allocator::cuda) || (m_alloc == allocator::cuda_uva))
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
buffer<T>::buffer(allocator alloc) : m_alloc(alloc),
    m_data(nullptr), m_size(0), m_capacity(0), m_owner(-1)
{
    assert_valid_allocator(alloc);
    this->set_owner();
}

// --------------------------------------------------------------------------
template <typename T>
buffer<T>::buffer(allocator alloc, size_t n_elem) : buffer<T>(alloc)
{
    m_data = this->allocate(n_elem);
    m_size = n_elem;
    m_capacity = n_elem;
}

// --------------------------------------------------------------------------
template <typename T>
buffer<T>::buffer(allocator alloc, size_t n_elem, const T &val) : buffer<T>(alloc)
{
    m_data = this->allocate(n_elem, val);
    m_size = n_elem;
    m_capacity = n_elem;
}

// --------------------------------------------------------------------------
template <typename T>
buffer<T>::buffer(allocator alloc, size_t n_elem, const T *vals) : buffer<T>(alloc)
{
    m_data = this->allocate(n_elem, vals);
    m_size = n_elem;
    m_capacity = n_elem;
}

// --------------------------------------------------------------------------
template <typename T>
buffer<T>::buffer(allocator alloc, size_t size, int owner,
    const std::shared_ptr<T> &data) : m_alloc(alloc),
    m_data(data), m_size(size), m_capacity(size), m_owner(owner)

{
    assert_valid_allocator(alloc);

    // query the driver api to determine the owner
#if defined(HAMR_ENABLE_CUDA)
    if (((alloc == allocator::cuda) ||
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
buffer<T>::buffer(allocator alloc, size_t size, int owner, T *ptr,
    delete_func_t df) : m_alloc(alloc), m_data(std::shared_ptr<T>(ptr, df)),
    m_size(size), m_capacity(size), m_owner(owner)
{
    assert_valid_allocator(alloc);

    // query the driver api to determine the owner
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
buffer<T>::buffer(allocator alloc, size_t size, int owner, T *ptr)
    : m_alloc(alloc), m_data(nullptr), m_size(size),
    m_capacity(size), m_owner(owner)
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
    else if (alloc == allocator::cuda)
    {
        m_data = std::shared_ptr<T>(ptr, cuda_malloc_deleter<T>(ptr, m_size));
    }
    else if (alloc == allocator::cuda_uva)
    {
        m_data = std::shared_ptr<T>(ptr, cuda_malloc_uva_deleter<T>(ptr, m_size));
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
buffer<T>::buffer(const buffer<T> &other) : buffer<T>(other.m_alloc)
{
    m_data = this->allocate(other);
    m_size = other.m_size;
    m_capacity = other.m_size;
}

// --------------------------------------------------------------------------
template <typename T>
buffer<T>::buffer(allocator alloc, const buffer<T> &other) : buffer<T>(alloc)
{
    m_data = this->allocate(other);
    m_size = other.m_size;
    m_capacity = other.m_size;
}

// --------------------------------------------------------------------------
template <typename T>
buffer<T>::buffer(buffer<T> &&other) : buffer<T>(other.m_alloc)
{
    this->swap(other);
}

// --------------------------------------------------------------------------
template <typename T>
void buffer<T>::operator=(buffer<T> &&other)
{
    this->swap(other);
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
void buffer<T>::swap(buffer<T> &other)
{
    std::swap(m_alloc, other.m_alloc);
    std::swap(m_data, other.m_data);
    std::swap(m_size, other.m_size);
    std::swap(m_capacity, other.m_capacity);
    std::swap(m_owner, other.m_owner);
}

// --------------------------------------------------------------------------
template <typename T>
int buffer<T>::cpu_accessible() const
{
    return (m_alloc == allocator::cpp) || (m_alloc == allocator::malloc) ||
        (m_alloc == allocator::cuda_uva) || (m_alloc == allocator::hip_uva);
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
        return cuda_malloc_allocator<T>::allocate(n_elem, val);
    }
    else if (m_alloc == allocator::cuda_uva)
    {
        return cuda_malloc_uva_allocator<T>::allocate(n_elem, val);
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
        return cuda_malloc_allocator<T>::allocate(n_elem, vals);
    }
    else if (m_alloc == allocator::cuda_uva)
    {
        activate_cuda_device dev(m_owner);
        return cuda_malloc_uva_allocator<T>::allocate(n_elem, vals);
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
    size_t n_elem = vals.size();

    if (m_alloc == allocator::cpp)
    {
        std::shared_ptr<const U> pvals = vals.get_cpu_accessible();

        // a deep copy was made, return the pointer to the copy
        if (std::is_same<T,U>::value && !vals.cpu_accessible())
            return std::const_pointer_cast<T>(pvals);

        return new_allocator<T>::allocate(n_elem, pvals.get());
    }
    else if (m_alloc == allocator::malloc)
    {
        std::shared_ptr<const U> pvals = vals.get_cpu_accessible();

        // a deep copy was made, return the pointer to the copy
        if (std::is_same<T,U>::value && !vals.cpu_accessible())
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

        return cuda_malloc_allocator<T>::allocate(n_elem, pvals.get(), true);
    }
    else if (m_alloc == allocator::cuda_uva)
    {
        activate_cuda_device dev(m_owner);
        std::shared_ptr<const U> pvals = vals.get_cuda_accessible();

        // a deep copy was made, return the pointer to the copy
        if (std::is_same<T,U>::value &&
            (!vals.cuda_accessible() || (vals.m_owner != m_owner)))
            return  std::const_pointer_cast<T>(pvals);

        return cuda_malloc_uva_allocator<T>::allocate(n_elem, pvals.get(), true);
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
    else if (m_alloc == allocator::cuda_uva)
    {
        activate_cuda_device dev(m_owner);
        return cuda_malloc_uva_allocator<T>::allocate(n_elem);
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
        if ((m_alloc == allocator::cpp) || (m_alloc == allocator::malloc))
        {
            ierr = copy_to_cpu_from_cpu(tmp.get(), m_data.get(), m_size);
        }
#if defined(HAMR_ENABLE_CUDA)
        else if ((m_alloc == allocator::cuda) || (m_alloc == allocator::cuda_uva))
        {
            activate_cuda_device dev(m_owner);
            ierr = copy_to_cuda_from_cuda(tmp.get(), m_data.get(), m_size);
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
        if ((m_alloc == allocator::cpp) || (m_alloc == allocator::malloc))
        {
            ierr = copy_to_cpu_from_cpu(tmp.get(), m_data.get(), m_size);
        }
#if defined(HAMR_ENABLE_CUDA)
        else if ((m_alloc == allocator::cuda) || (m_alloc == allocator::cuda_uva))
        {
            activate_cuda_device dev(m_owner);
            ierr = copy_to_cuda_from_cuda(tmp.get(), m_data.get(), m_size);
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

    return 0;
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
int buffer<T>::append(const U *src, size_t src_start, size_t n_vals)
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

    return 0;
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
int buffer<T>::append(const buffer<U> &src, size_t src_start, size_t n_vals)
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

    return 0;
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
int buffer<T>::append(const buffer<U> &src)
{
    return this->append(src, 0, src.size());
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
int buffer<T>::set(size_t dest_start, const U *src,
    size_t src_start, size_t n_vals)
{
    // bounds check
    assert(m_size >= (dest_start + n_vals));

    // copy the values (src is always on the CPU)
    int ierr = 0;
    if ((m_alloc == allocator::cpp) || (m_alloc == allocator::malloc))
    {
        ierr = copy_to_cpu_from_cpu(m_data.get() + dest_start,
            src + src_start, n_vals);
    }
#if defined(HAMR_ENABLE_CUDA)
    else if ((m_alloc == allocator::cuda) || (m_alloc == allocator::cuda_uva))
    {

        activate_cuda_device dev(m_owner);

        ierr = copy_to_cuda_from_cpu(m_data.get() + dest_start,
            src + src_start, n_vals);
    }
#endif
#if defined(HAMR_ENABLE_HIP)
    else if ((m_alloc == allocator::hip) || (m_alloc == allocator::hip_uva))
    {

        activate_hip_device dev(m_owner);

        ierr = copy_to_hip_from_cpu(m_data.get() + dest_start,
            src + src_start, n_vals);
    }
#endif
#if defined(HAMR_ENABLE_OPENMP)
    else if (m_alloc == allocator::openmp)
    {

        activate_openmp_device dev(m_owner);

        ierr = copy_to_openmp_from_cpu(m_data.get() + dest_start,
            src + src_start, n_vals);
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

    return 0;
}

// ---------------------------------------------------------------------------
template <typename T>
template <typename U>
int buffer<T>::set(size_t dest_start, const buffer<U> &src,
    size_t src_start, size_t n_vals)
{
    // bounds check
    assert(m_size >= (dest_start + n_vals));
    assert(src.size() >= (src_start + n_vals));

    // copy the value to the back. buffers can either be on the CPU or GPU
    // and use different technolofies so all permutations must be realized.
    int ierr = 0;
    if ((m_alloc == allocator::cpp) || (m_alloc == allocator::malloc))
    {
        // destination is on the CPU

        if ((src.m_alloc == allocator::cpp) ||
            (src.m_alloc == allocator::malloc))
        {
            // source is on the CPU
            ierr = copy_to_cpu_from_cpu(m_data.get() + dest_start,
                src.m_data.get() + src_start, n_vals);
        }
#if defined(HAMR_ENABLE_CUDA)
        else if ((src.m_alloc == allocator::cuda) ||
            (src.m_alloc == allocator::cuda_uva))
        {
            // source is on the GPU
            activate_cuda_device dev(src.m_owner);

            ierr = copy_to_cpu_from_cuda(m_data.get() + dest_start,
                src.m_data.get() + src_start, n_vals);
        }
#endif
#if defined(HAMR_ENABLE_HIP)
        else if ((src.m_alloc == allocator::hip) ||
            (src.m_alloc == allocator::hip_uva))
        {
            // source is on the GPU
            activate_hip_device dev(src.m_owner);

            ierr = copy_to_cpu_from_hip(m_data.get() + dest_start,
                src.m_data.get() + src_start, n_vals);
        }
#endif
#if defined(HAMR_ENABLE_OPENMP)
        else if (src.m_alloc == allocator::openmp)
        {
            // source is on the GPU
            activate_openmp_device dev(src.m_owner);

            ierr = copy_to_cpu_from_openmp(m_data.get() + dest_start,
                src.m_data.get() + src_start, n_vals);
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
    else if ((m_alloc == allocator::cuda) || (m_alloc == allocator::cuda_uva))
    {
        // destination is on the GPU
        activate_cuda_device dev(m_owner);

        if ((src.m_alloc == allocator::cpp) ||
            (src.m_alloc == allocator::malloc))
        {
            // source is on the CPU
            ierr = copy_to_cuda_from_cpu(m_data.get() + dest_start,
                src.m_data.get() + src_start, n_vals);
        }
        else if (src.cuda_accessible())
        {
            if (m_owner == src.m_owner)
            {
                // source is on this GPU
                ierr = copy_to_cuda_from_cuda(m_data.get() + dest_start,
                    src.m_data.get() + src_start, n_vals);
            }
            else
            {
                // source is on another GPU
                ierr = copy_to_cuda_from_cuda(m_data.get() + dest_start,
                    src.m_data.get() + src_start, src.m_owner, n_vals);
            }
        }
        else
        {
            std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
                " Invalid allocator type in the source "
                << get_allocator_name(src.m_alloc) << std::endl;
        }
    }
#endif
#if defined(HAMR_ENABLE_HIP)
    else if ((m_alloc == allocator::hip) || (m_alloc == allocator::hip_uva))
    {
        // destination is on the GPU
        activate_hip_device dev(m_owner);

        if ((src.m_alloc == allocator::cpp) ||
            (src.m_alloc == allocator::malloc))
        {
            // source is on the CPU
            ierr = copy_to_hip_from_cpu(m_data.get() + dest_start,
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
    }
#endif
#if defined(HAMR_ENABLE_OPENMP)
    else if (m_alloc == allocator::openmp)
    {
        // destination is on the GPU
        activate_openmp_device dev(m_owner);

        if ((src.m_alloc == allocator::cpp) ||
            (src.m_alloc == allocator::malloc))
        {
            // source is on the CPU
            ierr = copy_to_openmp_from_cpu(m_data.get() + dest_start,
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

    return 0;
}

// ---------------------------------------------------------------------------
template <typename T>
template <typename U>
int buffer<T>::get(size_t src_start, U *dest,
    size_t dest_start, size_t n_vals) const
{
    // bounds check
    assert(m_size >= (src_start + n_vals));

    // copy the values (dest is always on the CPU)
    int ierr = 0;
    if ((m_alloc == allocator::cpp) || (m_alloc == allocator::malloc))
    {
        ierr = copy_to_cpu_from_cpu(dest + dest_start,
            m_data.get() + src_start, n_vals);
    }
#if defined(HAMR_ENABLE_CUDA)
    else if ((m_alloc == allocator::cuda) || (m_alloc == allocator::cuda_uva))
    {
        activate_cuda_device dev(m_owner);

        ierr = copy_to_cpu_from_cuda(dest + dest_start,
            m_data.get() + src_start, n_vals);
    }
#endif
#if defined(HAMR_ENABLE_HIP)
    else if ((m_alloc == allocator::hip) || (m_alloc == allocator::hip_uva))
    {
        activate_hip_device dev(m_owner);

        ierr = copy_to_cpu_from_hip(dest + dest_start,
            m_data.get() + src_start, n_vals);
    }
#endif
#if defined(HAMR_ENABLE_OPENMP)
    else if (m_alloc == allocator::openmp)
    {
        activate_openmp_device dev(m_owner);

        ierr = copy_to_cpu_from_openmp(dest + dest_start,
            m_data.get() + src_start, n_vals);
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

    return 0;
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
int buffer<T>::get(size_t src_start,
    buffer<U> &dest, size_t dest_start, size_t n_vals) const
{
    // bounds check
    assert(m_size >= (src_start + n_vals));
    assert(dest.size() >= (dest_start + n_vals));

    // copy the value to the back. buffers can either be on the CPU or GPU
    // and use different technologies so all permutations must be realized.
    int ierr = 0;
    if ((m_alloc == allocator::cpp) || (m_alloc == allocator::malloc))
    {
        // destination is on the CPU

        if ((dest.m_alloc == allocator::cpp) ||
            (dest.m_alloc == allocator::malloc))
        {
            // source is on the CPU
            ierr = copy_to_cpu_from_cpu(dest.m_data.get() + dest_start,
                m_data.get() + src_start, n_vals);
        }
#if defined(HAMR_ENABLE_CUDA)
        else if ((dest.m_alloc == allocator::cuda) ||
            (dest.m_alloc == allocator::cuda_uva))
        {
            // source is on the GPU
            activate_cuda_device dev(m_owner);

            ierr = copy_to_cpu_from_cuda(dest.m_data.get() + dest_start,
                m_data.get() + src_start, n_vals);
        }
#endif
#if defined(HAMR_ENABLE_HIP)
        else if ((dest.m_alloc == allocator::hip) ||
            (dest.m_alloc == allocator::hip_uva))
        {
            // source is on the GPU
            activate_hip_device dev(m_owner);

            ierr = copy_to_cpu_from_hip(dest.m_data.get() + dest_start,
                m_data.get() + src_start, n_vals);
        }
#endif
#if defined(HAMR_ENABLE_OPENMP)
        else if (dest.m_alloc == allocator::openmp)
        {
            // source is on the GPU
            activate_openmp_device dev(m_owner);

            ierr = copy_to_cpu_from_openmp(dest.m_data.get() + dest_start,
                m_data.get() + src_start, n_vals);
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
        (m_alloc == allocator::cuda_uva))
    {
        // destination is on the GPU
        activate_cuda_device dev(dest.m_owner);

        if ((dest.m_alloc == allocator::cpp) ||
            (dest.m_alloc == allocator::malloc))
        {
            // source is on the CPU
            ierr = copy_to_cuda_from_cpu(dest.m_data.get() + dest_start,
                m_data.get() + src_start, n_vals);
        }
        else if ((dest.m_alloc == allocator::cuda) ||
            (dest.m_alloc == allocator::cuda_uva))
        {
            if (m_owner == dest.m_owner)
            {
                // source is on this GPU
                ierr = copy_to_cuda_from_cuda(dest.m_data.get() + dest_start,
                    m_data.get() + src_start, n_vals);
            }
            else
            {
                // source is on another GPU
                ierr = copy_to_cuda_from_cuda(dest.m_data.get() + dest_start,
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
    }
#endif
#if defined(HAMR_ENABLE_HIP)
    else if ((m_alloc == allocator::hip) ||
        (m_alloc == allocator::hip_uva))
    {
        // destination is on the GPU
        activate_hip_device dev(dest.m_owner);

        if ((dest.m_alloc == allocator::cpp) ||
            (dest.m_alloc == allocator::malloc))
        {
            // source is on the CPU
            ierr = copy_to_hip_from_cpu(dest.m_data.get() + dest_start,
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
    }
#endif
#if defined(HAMR_ENABLE_OPENMP)
    else if (m_alloc == allocator::openmp)
    {
        // destination is on the GPU
        activate_openmp_device dev(dest.m_owner);

        if ((dest.m_alloc == allocator::cpp) ||
            (dest.m_alloc == allocator::malloc))
        {
            // source is on the CPU
            ierr = copy_to_openmp_from_cpu(dest.m_data.get() + dest_start,
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

    return 0;
}

// --------------------------------------------------------------------------
template <typename T>
std::shared_ptr<const T> buffer<T>::get_cpu_accessible() const
{
    return const_cast<buffer<T>*>(this)->get_cpu_accessible();
}

// ---------------------------------------------------------------------------
template <typename T>
std::shared_ptr<T> buffer<T>::get_cpu_accessible()
{
    if ((m_alloc == allocator::cpp) || (m_alloc == allocator::malloc))
    {
        // already on the CPU
        return m_data;
    }
#if defined(HAMR_ENABLE_CUDA)
    else if ((m_alloc == allocator::cuda) ||
        (m_alloc == allocator::cuda_uva))
    {
        // make a copy on the CPU
        std::shared_ptr<T> tmp = malloc_allocator<T>::allocate(m_size);

        activate_cuda_device dev(m_owner);

        if (copy_to_cpu_from_cuda(tmp.get(), m_data.get(), m_size))
            return nullptr;

        return tmp;
    }
#endif
#if defined(HAMR_ENABLE_HIP)
    else if ((m_alloc == allocator::hip) ||
        (m_alloc == allocator::hip_uva))
    {
        // make a copy on the CPU
        std::shared_ptr<T> tmp = malloc_allocator<T>::allocate(m_size);

        activate_hip_device dev(m_owner);

        if (copy_to_cpu_from_hip(tmp.get(), m_data.get(), m_size))
            return nullptr;

        return tmp;
    }
#endif
#if defined(HAMR_ENABLE_OPENMP)
    else if (m_alloc == allocator::openmp)
    {
        // make a copy on the CPU
        std::shared_ptr<T> tmp = malloc_allocator<T>::allocate(m_size);

        activate_openmp_device dev(m_owner);

        if (copy_to_cpu_from_openmp(tmp.get(), m_data.get(), m_size))
            return nullptr;

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
    return const_cast<buffer<T>*>(this)->get_cuda_accessible();
}

// ---------------------------------------------------------------------------
template <typename T>
std::shared_ptr<T> buffer<T>::get_cuda_accessible()
{
#if !defined(HAMR_ENABLE_CUDA)
    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " get_cuda_accessible failed, CUDA is not available."
        << std::endl;
    return nullptr;
#else
    if ((m_alloc == allocator::cpp) || (m_alloc == allocator::malloc))
    {
        // make a copy on the GPU
        std::shared_ptr<T> tmp = cuda_malloc_allocator<T>::allocate(m_size);

        if (copy_to_cuda_from_cpu(tmp.get(), m_data.get(), m_size))
            return nullptr;

        return tmp;
    }
    else if ((m_alloc == allocator::cuda) || (m_alloc == allocator::cuda_uva))
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
            std::shared_ptr<T> tmp = cuda_malloc_allocator<T>::allocate(m_size);

            if (copy_to_cuda_from_cuda(tmp.get(), m_data.get(), m_owner, m_size))
                return nullptr;

            return tmp;
        }
    }
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
    return const_cast<buffer<T>*>(this)->get_hip_accessible();
}

// ---------------------------------------------------------------------------
template <typename T>
std::shared_ptr<T> buffer<T>::get_hip_accessible()
{
#if !defined(HAMR_ENABLE_HIP)
    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " get_hip_accessible failed, HIP is not available."
        << std::endl;
    return nullptr;
#else
    if ((m_alloc == allocator::cpp) || (m_alloc == allocator::malloc))
    {
        // make a copy on the GPU
        std::shared_ptr<T> tmp = hip_malloc_allocator<T>::allocate(m_size);

        if (copy_to_hip_from_cpu(tmp.get(), m_data.get(), m_size))
            return nullptr;

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
    return const_cast<buffer<T>*>(this)->get_openmp_accessible();
}

// ---------------------------------------------------------------------------
template <typename T>
std::shared_ptr<T> buffer<T>::get_openmp_accessible()
{
#if !defined(HAMR_ENABLE_OPENMP)
    std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
        " get_openmp_accessible failed, OpenMP is not available."
        << std::endl;
    return nullptr;
#else
    if ((m_alloc == allocator::cpp) || (m_alloc == allocator::malloc))
    {
        // make a copy on the GPU
        std::shared_ptr<T> tmp = openmp_allocator<T>::allocate(m_size);

        if (copy_to_openmp_from_cpu(tmp.get(), m_data.get(), m_size))
            return nullptr;

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

            return tmp;
        }
    }
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


// --------------------------------------------------------------------------
template <typename T>
int buffer<T>::print() const
{
    std::cerr << "m_alloc = " << get_allocator_name(m_alloc)
        << ", m_size = " << m_size << ", m_capacity = " << m_capacity
        << ", m_data = ";

    if (m_size)
    {
        if ((m_alloc == allocator::cpp) || (m_alloc == allocator::malloc))
        {
            std::cerr << m_data.get()[0];
            for (size_t i = 1; i < m_size; ++i)
                std::cerr << ", " << m_data.get()[i];
            std::cerr << std::endl;
        }
#if defined(HAMR_ENABLE_CUDA)
        else if ((m_alloc == allocator::cuda) || (m_alloc == allocator::cuda_uva))
        {
            activate_cuda_device dev(m_owner);
            cuda_print(m_data.get(), m_size);
        }
#endif
#if defined(HAMR_ENABLE_HIP)
        else if ((m_alloc == allocator::hip) || (m_alloc == allocator::hip_uva))
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
