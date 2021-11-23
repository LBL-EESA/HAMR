#ifndef buffer_h
#define buffer_h

#include "hamr_config.h"
#include "hamr_env.h"
#include "hamr_malloc_allocator.h"
#include "hamr_new_allocator.h"
#if defined(HAMR_ENABLE_CUDA)
#include "hamr_cuda_malloc_allocator.h"
#include "hamr_cuda_malloc_uva_allocator.h"
#include "hamr_cuda_print.h"
#endif
#include "hamr_copy.h"

#include <memory>
#include <iostream>

/// heterogeneous accelerator memory resource
namespace hamr
{
template <typename T> class buffer;

///  a shared pointer to an instance of a buffer<T>
template <typename T>
using p_buffer = std::shared_ptr<buffer<T>>;

///  a shared pointer to an instance of a const buffer<T>
template <typename T>
using const_p_buffer = std::shared_ptr<const buffer<T>>;

/// a helper for explicitly casting to a const buffer pointer.
template <typename T>
hamr::const_p_buffer<T> const_ptr(const hamr::p_buffer<T> &v)
{
    return hamr::const_p_buffer<T>(v);
}

/// a helper for getting a reference to pointed to hamr::buffer
template <typename T>
const hamr::buffer<T> &ref_to(const hamr::const_p_buffer<T> &ptr)
{
    return *(ptr.get());
}

/// a helper for getting a reference to pointed to hamr::buffer
template <typename T>
hamr::buffer<T> &ref_to(const hamr::p_buffer<T> &ptr)
{
    return *(ptr.get());
}


/** @brief A technology agnostic  a buffer that manages data on CPUs, GPUs, and
 * accelerators.
 * @details The buffer mediates between different accelerator and platform
 * portability technologies' memory models. Examples of platform portability
 * technologies are HIP, OpenMP, OpenCL, SYCL, and Kokos, Examples of
 * accelerator technologies are CUDA and ROCm. Other accelerator and platform
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

    /// allocator types
    enum {
        none = -1,
        cpp = 0,     /// allocates memory with new
        malloc = 1,  /// allocates memory with malloc
        cuda = 2,    /// allocates memory with cudaMalloc
        cuda_uva = 3 /// allocates memory with cudaMallocManaged
    };


    /// construct an empty buffer that will use the passed allocator type
    buffer(int alloc);

    /// construct a buffer with n_elem size using the passed allocator type
    buffer(int alloc, size_t n_elem);

    /** construct a buffer with n_elem size initialized to the passed value
     * using the passed allocator type
     */
    buffer(int alloc, size_t n_elem, const T &val);

    /** construct a buffer with n_elem size initialized to the passed value
     * using the passed allocator type
     */
    buffer(int alloc, size_t n_elem, const T *vals);

    /// copy construct from the passed buffer
    buffer(const buffer<T> &other);

    /// copy construct from the passed buffer, using the passed allocator type.
    buffer(int alloc, const buffer<T> &other);

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
    template <typename U>
    void operator=(buffer<U> &&other);

    /// swap the contents of the two buffers
    void swap(buffer<T> &other);

    /** @name reserve
     * allocates space for n_elems of data
     */
    ///@{
    int reserve(size_t n_elem);
    int reserve(size_t n_elem, const T &val);
    ///@}

    /** @name resize
     * resizes storage for n_elems of data
     */
    ///@{
    int resize(size_t n_elem);
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

    /** @name get_accessible
     * get a pointer to the data that is accessible in the given technology
     */
    ///@{
    /** returns a pointer to the contents of the buffer accessible on the CPU
     * if the buffer is currently accessible by codes running on the CPU then
     * this call is a NOOP.  If the buffer is not currently accessible by codes
     * running on the CPU then a temporary buffer is allocated and the data is
     * moved to the CPU.  The returned shared_ptr deals with deallocation of
     * the temporary if needed.
     */
    std::shared_ptr<T> get_cpu_accessible();
    std::shared_ptr<const T> get_cpu_accessible() const;

    /** returns a pointer to the contents of the buffer accessible on the CUDA
     * if the buffer is currently accessible by codes running on the CUDA then
     * this call is a NOOP.  If the buffer is not currently accessible by codes
     * running on the CUDA then a temporary buffer is allocated and the data is
     * moved to the CUDA.  The returned shared_ptr deals with deallocation of
     * the temporary if needed.
     */
    std::shared_ptr<T> get_cuda_accessible();
    std::shared_ptr<const T> get_cuda_accessible() const;
    ///@}

    /// returns the allocator type enum
    int get_allocator() const { return m_alloc; }

    /// returns true if the data is accessible from CUDA codes
    int cuda_accessible() const;

    /// returns true if the data is accessible from codes running on the CPU
    int cpu_accessible() const;

    /// prints the contents to the stderr stream
    int print() const;

protected:
    /// return the human readable name of the allocator
    static
    const char *get_allocator_name(int alloc);

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
    std::shared_ptr<T> allocate(const const_p_buffer<U> &vals);

private:
    int m_alloc;
    std::shared_ptr<T> m_data;
    size_t m_size;
    size_t m_capacity;

    template<typename U> friend class buffer;
};



// --------------------------------------------------------------------------
template <typename T>
buffer<T>::buffer(int alloc) : m_alloc(alloc),
    m_data(nullptr), m_size(0), m_capacity(0)
{
    assert((alloc == buffer<T>::cpp) || (alloc == buffer<T>::malloc) ||
        (alloc == buffer<T>::cuda) || (alloc == buffer<T>::cuda_uva));
}

// --------------------------------------------------------------------------
template <typename T>
buffer<T>::buffer(int alloc, size_t n_elem) : buffer<T>(alloc)
{
    this->resize(n_elem);
}

// --------------------------------------------------------------------------
template <typename T>
buffer<T>::buffer(int alloc, size_t n_elem, const T &val) : buffer<T>(alloc)
{
    this->resize(n_elem, val);
}

// --------------------------------------------------------------------------
template <typename T>
buffer<T>::buffer(int alloc, size_t n_elem, const T *vals) : buffer<T>(alloc)
{
    this->resize(n_elem);
    this->set(0, vals, 0, n_elem);
}

// --------------------------------------------------------------------------
template <typename T>
buffer<T>::buffer(const buffer<T> &other) : buffer<T>(other.m_alloc)
{
    this->assign(other);
}

// --------------------------------------------------------------------------
template <typename T>
buffer<T>::buffer(int alloc, const buffer<T> &other) : buffer<T>(alloc)
{
    this->assign(other);
}

// --------------------------------------------------------------------------
template <typename T>
buffer<T>::buffer(buffer<T> &&other) : buffer<T>(other.m_alloc)
{
    this->swap(other);
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
void buffer<T>::operator=(buffer<U> &&other)
{
    if (std::is_same<T,U>::value && (m_alloc == other.m_alloc))
        this->swap(other);
    else
        this->assign(other);
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
}

// --------------------------------------------------------------------------
template <typename T>
const char *buffer<T>::get_allocator_name(int alloc)
{
    if (alloc == buffer<T>::cpp)
    {
        return "cpp";
    }
    else if (alloc == buffer<T>::malloc)
    {
        return "malloc";
    }
    else if (alloc == buffer<T>::cuda)
    {
        return "cuda_malloc_allocator";
    }
    else if (alloc == buffer<T>::cuda_uva)
    {
        return "cuda_malloc_uva_allocator";
    }

    return "the allocator name is not known";
}

// --------------------------------------------------------------------------
template <typename T>
int buffer<T>::cuda_accessible() const
{
    return (m_alloc == buffer<T>::cpp) ||
        (m_alloc == buffer<T>::malloc) || (m_alloc == buffer<T>::cuda_uva);
}

// --------------------------------------------------------------------------
template <typename T>
int buffer<T>::cpu_accessible() const
{
    return (m_alloc == buffer<T>::cuda) || (m_alloc == buffer<T>::cuda_uva);
}

// --------------------------------------------------------------------------
template <typename T>
std::shared_ptr<T> buffer<T>::allocate(size_t n_elem, const T &val)
{
    if (m_alloc == buffer<T>::cpp)
    {
        return new_allocator<T>::allocate(n_elem, val);
    }
    else if (m_alloc == buffer<T>::malloc)
    {
        return malloc_allocator<T>::allocate(n_elem, val);
    }
#if defined(HAMR_ENABLE_CUDA)
    else if (m_alloc == buffer<T>::cuda)
    {
        return cuda_malloc_allocator<T>::allocate(n_elem, val);
    }
    else if (m_alloc == buffer<T>::cuda_uva)
    {
        return cuda_malloc_uva_allocator<T>::allocate(n_elem, val);
    }
#endif

    std::cerr << "ERROR: Invalid allocator type "
        << get_allocator_name(m_alloc) << std::endl;

    return nullptr;
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
std::shared_ptr<T> buffer<T>::allocate(size_t n_elem, const U *vals)
{
    if (m_alloc == buffer<T>::cpp)
    {
        return new_allocator<T>::allocate(n_elem, vals);
    }
    else if (m_alloc == buffer<T>::malloc)
    {
        return malloc_allocator<T>::allocate(n_elem, vals);
    }
#if defined(HAMR_ENABLE_CUDA)
    else if (m_alloc == buffer<T>::cuda)
    {
        return cuda_malloc_allocator<T>::allocate(n_elem, vals);
    }
    else if (m_alloc == buffer<T>::cuda_uva)
    {
        return cuda_malloc_uva_allocator<T>::allocate(n_elem, vals);
    }
#endif

    std::cerr << "ERROR: Invalid allocator type "
        << get_allocator_name(m_alloc) << std::endl;

    return nullptr;
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
std::shared_ptr<T> buffer<T>::allocate(const const_p_buffer<U> &vals)
{
    size_t n_elem = vals->size();

    if (m_alloc == buffer<T>::cpp)
    {
        std::shared_ptr<const U> pvals = vals->get_cpu_accessible();
        return new_allocator<T>::allocate(n_elem, pvals.get());
    }
    else if (m_alloc == buffer<T>::malloc)
    {
        std::shared_ptr<const U> pvals = vals->get_cpu_accessible();
        return malloc_allocator<T>::allocate(n_elem, pvals.get());
    }
#if defined(HAMR_ENABLE_CUDA)
    else if (m_alloc == buffer<T>::cuda)
    {
        std::shared_ptr<const U> pvals = vals->get_cuda_accessible();
        return cuda_malloc_allocator<T>::allocate(n_elem, pvals.get(), true);
    }
    else if (m_alloc == buffer<T>::cuda_uva)
    {
        std::shared_ptr<const U> pvals = vals->get_cuda_accessible();
        return cuda_malloc_uva_allocator<T>::allocate(n_elem, pvals.get(), true);
    }
#endif

    std::cerr << "ERROR: Invalid allocator type "
        << get_allocator_name(m_alloc) << std::endl;

    return nullptr;
}

// --------------------------------------------------------------------------
template <typename T>
std::shared_ptr<T> buffer<T>::allocate(size_t n_elem)
{
    if (m_alloc == buffer<T>::cpp)
    {
        return new_allocator<T>::allocate(n_elem);
    }
    else if (m_alloc == buffer<T>::malloc)
    {
        return malloc_allocator<T>::allocate(n_elem);
    }
#if defined(HAMR_ENABLE_CUDA)
    else if (m_alloc == buffer<T>::cuda)
    {
        return cuda_malloc_allocator<T>::allocate(n_elem);
    }
    else if (m_alloc == buffer<T>::cuda_uva)
    {
        return cuda_malloc_uva_allocator<T>::allocate(n_elem);
    }
#endif

    std::cerr << "ERROR: Invalid allocator type "
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
        if ((m_alloc == buffer<T>::cpp) || (m_alloc == buffer<T>::malloc))
        {
            ierr = copy_to_cpu_from_cpu(tmp.get(), m_data.get(), m_size);
        }
#if defined(HAMR_ENABLE_CUDA)
        else if ((m_alloc == buffer<T>::cuda) || (m_alloc == buffer<T>::cuda_uva))
        {
            ierr = copy_to_cuda_from_cuda(tmp.get(), m_data.get(), m_size);
        }
#endif
        else
        {
            std::cerr << "ERROR: Invalid allocator type "
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
        if ((m_alloc == buffer<T>::cpp) || (m_alloc == buffer<T>::malloc))
        {
            ierr = copy_to_cpu_from_cpu(tmp.get(), m_data.get(), m_size);
        }
#if defined(HAMR_ENABLE_CUDA)
        else if ((m_alloc == buffer<T>::cuda) || (m_alloc == buffer<T>::cuda_uva))
        {
            ierr = copy_to_cuda_from_cuda(tmp.get(), m_data.get(), m_size);
        }
#endif
        else
        {
            std::cerr << "ERROR: Invalid allocator type "
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
    if ((m_alloc == buffer<T>::cpp) || (m_alloc == buffer<T>::malloc))
    {
        ierr = copy_to_cpu_from_cpu(m_data.get() + dest_start,
            src + src_start, n_vals);
    }
#if defined(HAMR_ENABLE_CUDA)
    else if ((m_alloc == buffer<T>::cuda) || (m_alloc == buffer<T>::cuda_uva))
    {
        ierr = copy_to_cuda_from_cpu(m_data.get() + dest_start,
            src + src_start, n_vals);
    }
#endif
    else
    {
        std::cerr << "ERROR: Invalid allocator type "
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
int buffer<T>::set(size_t dest_start, const buffer<U> &src,
    size_t src_start, size_t n_vals)
{
    // bounds check
    assert(m_size >= (dest_start + n_vals));
    assert(src.size() >= (src_start + n_vals));

    // copy the value to the back. buffers can either be on the CPU or GPU
    // and use different technolofies so all permutations must be realized.
    int ierr = 0;
    if ((m_alloc == buffer<T>::cpp) || (m_alloc == buffer<T>::malloc))
    {
        // destination is on the CPU

        if ((src.m_alloc == buffer<T>::cpp) ||
            (src.m_alloc == buffer<T>::malloc))
        {
            // source is on the CPU
            ierr = copy_to_cpu_from_cpu(m_data.get() + dest_start,
                src.m_data.get() + src_start, n_vals);
        }
        else if ((src.m_alloc == buffer<T>::cuda) ||
            (src.m_alloc == buffer<T>::cuda_uva))
        {
            // source is on the GPU
            ierr = copy_to_cpu_from_cuda(m_data.get() + dest_start,
                src.m_data.get() + src_start, n_vals);
        }
        else
        {
            std::cerr << "ERROR: Invalid allocator type in the source "
                << get_allocator_name(src.m_alloc) << std::endl;
        }
    }
#if defined(HAMR_ENABLE_CUDA)
    else if ((m_alloc == buffer<T>::cuda) || (m_alloc == buffer<T>::cuda_uva))
    {
        // destination is on the GPU

        if ((src.m_alloc == buffer<T>::cpp) ||
            (src.m_alloc == buffer<T>::malloc))
        {
            // source is on the CPU
            ierr = copy_to_cuda_from_cpu(m_data.get() + dest_start,
                src.m_data.get() + src_start, n_vals);
        }
        else if ((src.m_alloc == buffer<T>::cuda) ||
            (src.m_alloc == buffer<T>::cuda_uva))
        {
            // source is on the GPU
            ierr = copy_to_cuda_from_cuda(m_data.get() + dest_start,
                src.m_data.get() + src_start, n_vals);
        }
        else
        {
            std::cerr << "ERROR: Invalid allocator type in the source "
                << get_allocator_name(src.m_alloc) << std::endl;
        }
    }
#endif
    else
    {
        std::cerr << "ERROR: Invalid allocator type "
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
    if ((m_alloc == buffer<T>::cpp) || (m_alloc == buffer<T>::malloc))
    {
        ierr = copy_to_cpu_from_cpu(dest + dest_start,
            m_data.get() + src_start, n_vals);
    }
#if defined(HAMR_ENABLE_CUDA)
    else if ((m_alloc == buffer<T>::cuda) || (m_alloc == buffer<T>::cuda_uva))
    {
        ierr = copy_to_cpu_from_cuda(dest + dest_start,
            m_data.get() + src_start, n_vals);
    }
#endif
    else
    {
        std::cerr << "ERROR: Invalid allocator type "
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
    // and use different technolofies so all permutations must be realized.
    int ierr = 0;
    if ((m_alloc == buffer<T>::cpp) || (m_alloc == buffer<T>::malloc))
    {
        // destination is on the CPU

        if ((dest.m_alloc == buffer<T>::cpp) ||
            (dest.m_alloc == buffer<T>::malloc))
        {
            // source is on the CPU
            ierr = copy_to_cpu_from_cpu(dest.m_data.get() + dest_start,
                m_data.get() + src_start, n_vals);
        }
        else if ((dest.m_alloc == buffer<T>::cuda) ||
            (dest.m_alloc == buffer<T>::cuda_uva))
        {
            // source is on the GPU
            ierr = copy_to_cpu_from_cuda(dest.m_data.get() + dest_start,
                m_data.get() + src_start, n_vals);
        }
        else
        {
            std::cerr << "ERROR: Invalid allocator type in the source "
                << get_allocator_name(dest.m_alloc) << std::endl;
        }
    }
#if defined(HAMR_ENABLE_CUDA)
    else if ((m_alloc == buffer<T>::cuda) ||
        (m_alloc == buffer<T>::cuda_uva))
    {
        // destination is on the GPU

        if ((dest.m_alloc == buffer<T>::cpp) ||
            (dest.m_alloc == buffer<T>::malloc))
        {
            // source is on the CPU
            ierr = copy_to_cuda_from_cpu(dest.m_data.get() + dest_start,
                m_data.get() + src_start, n_vals);
        }
        else if ((dest.m_alloc == buffer<T>::cuda) ||
            (dest.m_alloc == buffer<T>::cuda_uva))
        {
            // source is on the GPU
            ierr = copy_to_cuda_from_cuda(dest.m_data.get() + dest_start,
                m_data.get() + src_start, n_vals);
        }
        else
        {
            std::cerr << "ERROR: Invalid allocator type in the source "
                << get_allocator_name(dest.m_alloc) << std::endl;
        }
    }
#endif
    else
    {
        std::cerr << "ERROR: Invalid allocator type "
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
    if ((m_alloc == buffer<T>::cpp) || (m_alloc == buffer<T>::malloc))
    {
        // already on the CPU
        return m_data;
    }
#if defined(HAMR_ENABLE_CUDA)
    else if ((m_alloc == buffer<T>::cuda) ||
        (m_alloc == buffer<T>::cuda_uva))
    {
        // make a copy on the CPU
        std::shared_ptr<T> tmp = malloc_allocator<T>::allocate(m_size);

        if (copy_to_cpu_from_cuda(tmp.get(), m_data.get(), m_size))
            return nullptr;

        return tmp;
    }
#endif
    else
    {
        std::cerr << "ERROR: Invalid allocator type "
            << get_allocator_name(m_alloc) << std::endl;
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
    std::cerr << "ERROR: get_cuda_accessible failed, CUDA is not available."
        << std::endl;
    return nullptr;
#else
    if ((m_alloc == buffer<T>::cpp) || (m_alloc == buffer<T>::malloc))
    {
        // make a copy on the CPU
        std::shared_ptr<T> tmp = cuda_malloc_allocator<T>::allocate(m_size);

        if (copy_to_cuda_from_cpu(tmp.get(), m_data.get(), m_size))
            return nullptr;

        return tmp;
    }
    else if ((m_alloc == buffer<T>::cuda) || (m_alloc == buffer<T>::cuda_uva))
    {
        // already on the GPU
        return m_data;
    }
    else
    {
        std::cerr << "ERROR: Invalid allocator type "
            << get_allocator_name(m_alloc) << std::endl;
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
        if ((m_alloc == buffer<T>::cpp) || (m_alloc == buffer<T>::malloc))
        {
            std::cerr << m_data.get()[0];
            for (size_t i = 1; i < m_size; ++i)
                std::cerr << ", " << m_data.get()[i];
            std::cerr << std::endl;
        }
#if defined(HAMR_ENABLE_CUDA)
        else if ((m_alloc == buffer<T>::cuda) || (m_alloc == buffer<T>::cuda_uva))
        {
            cuda_print(m_data.get(), m_size);
        }
#endif
        else
        {
            std::cerr << "ERROR: Invalid allocator type "
                << get_allocator_name(m_alloc) << std::endl;
        }
    }

    return 0;
}

}
#endif