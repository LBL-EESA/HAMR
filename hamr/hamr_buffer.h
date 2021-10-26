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

template <typename T>
using p_buffer = std::shared_ptr<buffer<T>>;

template <typename T>
using const_p_buffer = std::shared_ptr<const buffer<T>>;

///  a buffer that manages data on the CPU or GPU using any technology
template <typename T>
class buffer : std::enable_shared_from_this<buffer<T>>
{
public:

    /// allocator types
    enum {
        cpp = 0,     /// allocates memory with new
        malloc = 1,  /// allocates memory with malloc
        cuda = 2,    /// allocates memory with cudaMalloc
        cuda_uva = 3 /// allocates memory with cudaMallocManaged
    };

    /** @name New
     * allocates an empty and unitialized buffer that will use the declared
     * allocator. An allocator type must be declared to construct the buffer.
     */
    ///@{
    static p_buffer<T> New(int alloc, size_t n_elem = 0);
    static p_buffer<T> New(int alloc, size_t n_elem, const T &val);

    template <typename U>
    static p_buffer<T> New(int alloc, size_t n_elem, const U *vals);

    template <typename U>
    static p_buffer<T> New(int alloc, const const_p_buffer<U> &vals);
    ///@}

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

    /// free all internall storage
    int free();

    /// returns the number of elelemts of storage allocated to the buffer
    size_t size() const { return m_size; }

    /** @name copy */
    ///@{
    /// copy the range from the passed array (src is always on the CPU)
    template<typename U>
    int copy(const U *src, size_t src_start, size_t n_vals);

    /// copy the range from the passed buffer
    template<typename U>
    int copy(const const_p_buffer<U> &src, size_t src_start, size_t n_vals);

    /// copy the passed buffer
    template<typename U>
    int copy(const const_p_buffer<U> &src);
    ///@}


    /** @anme append
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
    int append(const const_p_buffer<U> &src, size_t src_start, size_t n_vals);

    /** appends to the end of the buffer,
     * extending the buffer as needed.
     */
    template <typename U>
    int append(const const_p_buffer<U> &src);
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
    int set(size_t dest_start, const const_p_buffer<U> &src, size_t src_start, size_t n_vals);
    ///@}


    /** @name get
     * gets a range of values from the buffer
     */
    ///@{
    /** gets n_vals elements starting at src_start into the passed array
     * elements starting at dest_start (dest is always on the CPU)*/
    template <typename U>
    int get(size_t src_start, const U *dest, size_t dest_start, size_t n_vals);

    /** gets n_vals elements starting at src_start into the passed buffer's
     * elements starting at dest_start */
    template <typename U>
    int get(size_t src_start, const const_p_buffer<U> &dest, size_t dest_start, size_t n_vals);

    /** returns a pointer to the contents of the buffer accessible on the CPU
     * if the buffer is currently accessible by codes running on the CPU then
     * this call is a NOOP.  If the buffer is not currently accessible by codes
     * running on the CPU then a temporary buffer is allocated and the data is
     * moved to the CPU.  The returned shared_ptr deals with deallocation of
     * the temporary if needed.
     */
    std::shared_ptr<T> get_cpu_accessible();
    std::shared_ptr<const T> get_cpu_accessible() const;

    /** retruns a pointer to the contents of the buffer accessible on the CUDA
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


    /// prints the contents to the stderr stream
    int print();

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


    buffer(int alloc) : m_alloc(alloc), m_data(nullptr), m_size(0), m_capacity(0) {}

    // these could be implemented if useful
    buffer() = delete;
    buffer(const buffer&) = delete;
    buffer(buffer&&) = delete;
    void operator=(const buffer&) = delete;


private:
    int m_alloc;
    std::shared_ptr<T> m_data;
    size_t m_size;
    size_t m_capacity;
};



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
p_buffer<T> buffer<T>::New(int alloc, size_t n_elem, const T &val)
{
    if (!((alloc == buffer<T>::cpp) || (alloc == buffer<T>::malloc) ||
        (alloc == buffer<T>::cuda) || (alloc == buffer<T>::cuda_uva)))
    {
        std::cerr << "ERROR: Invalid allocator ("
            << get_allocator_name(alloc) << ")" << std::endl;
        return nullptr;
    }

    p_buffer<T> buf(new buffer(alloc));

    if (n_elem && buf->resize(n_elem, val))
        return nullptr;

    return buf;
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
p_buffer<T> buffer<T>::New(int alloc, size_t n_elem, const U *vals)
{
    if (!((alloc == buffer<T>::cpp) || (alloc == buffer<T>::malloc) ||
        (alloc == buffer<T>::cuda) || (alloc == buffer<T>::cuda_uva)))
    {
        std::cerr << "ERROR: Invalid allocator ("
            << get_allocator_name(alloc) << ")" << std::endl;
        return nullptr;
    }

    p_buffer<T> buf(new buffer(alloc));

    if (n_elem)
    {
        // allocate space
        if (!(buf->m_data = buf->allocate(n_elem, vals)))
            return nullptr;

        // update the size
        buf->m_capacity = n_elem;
        buf->m_size = n_elem;
    }

    return buf;
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
p_buffer<T> buffer<T>::New(int alloc, const const_p_buffer<U> &vals)
{
    if (!((alloc == buffer<T>::cpp) || (alloc == buffer<T>::malloc) ||
        (alloc == buffer<T>::cuda) || (alloc == buffer<T>::cuda_uva)))
    {
        std::cerr << "ERROR: Invalid allocator ("
            << get_allocator_name(alloc) << ")" << std::endl;
        return nullptr;
    }

    p_buffer<T> buf(new buffer(alloc));

    size_t n_elem = vals->size();
    if (n_elem)
    {
        // allocate space
        if (!(buf->m_data = buf->allocate(vals)))
            return nullptr;

        // update the size
        buf->m_capacity = n_elem;
        buf->m_size = n_elem;
    }

    return buf;
}

// --------------------------------------------------------------------------
template <typename T>
p_buffer<T> buffer<T>::New(int alloc, size_t n_elem)
{
    if (!((alloc == buffer<T>::cpp) || (alloc == buffer<T>::malloc) ||
        (alloc == buffer<T>::cuda) || (alloc == buffer<T>::cuda_uva)))
    {
        std::cerr << "ERROR: Invalid allocator ("
            << get_allocator_name(alloc) << ")" << std::endl;
        return nullptr;
    }

    p_buffer<T> buf(new buffer<T>(alloc));

    if (n_elem && buf->resize(n_elem))
        return nullptr;

    return buf;
}

// --------------------------------------------------------------------------
template <typename T>
std::shared_ptr<T> buffer<T>::allocate(size_t n_elem, const T &val)
{
    if (m_alloc == buffer<T>::cpp)
    {
        return hamr::new_allocator<T>::allocate(n_elem, val);
    }
    else if (m_alloc == buffer<T>::malloc)
    {
        return hamr::malloc_allocator<T>::allocate(n_elem, val);
    }
#if defined(HAMR_ENABLE_CUDA)
    else if (m_alloc == buffer<T>::cuda)
    {
        return hamr::cuda_malloc_allocator<T>::allocate(n_elem, val);
    }
    else if (m_alloc == buffer<T>::cuda_uva)
    {
        return hamr::cuda_malloc_uva_allocator<T>::allocate(n_elem, val);
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
        return hamr::new_allocator<T>::allocate(n_elem, vals);
    }
    else if (m_alloc == buffer<T>::malloc)
    {
        return hamr::malloc_allocator<T>::allocate(n_elem, vals);
    }
#if defined(HAMR_ENABLE_CUDA)
    else if (m_alloc == buffer<T>::cuda)
    {
        return hamr::cuda_malloc_allocator<T>::allocate(n_elem, vals);
    }
    else if (m_alloc == buffer<T>::cuda_uva)
    {
        return hamr::cuda_malloc_uva_allocator<T>::allocate(n_elem, vals);
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
        return hamr::new_allocator<T>::allocate(n_elem, pvals.get());
    }
    else if (m_alloc == buffer<T>::malloc)
    {
        std::shared_ptr<const U> pvals = vals->get_cpu_accessible();
        return hamr::malloc_allocator<T>::allocate(n_elem, pvals.get());
    }
#if defined(HAMR_ENABLE_CUDA)
    else if (m_alloc == buffer<T>::cuda)
    {
        std::shared_ptr<const U> pvals = vals->get_cuda_accessible();
        return hamr::cuda_malloc_allocator<T>::allocate(n_elem, pvals.get(), true);
    }
    else if (m_alloc == buffer<T>::cuda_uva)
    {
        std::shared_ptr<const U> pvals = vals->get_cuda_accessible();
        return hamr::cuda_malloc_uva_allocator<T>::allocate(n_elem, pvals.get(), true);
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
        return hamr::new_allocator<T>::allocate(n_elem);
    }
    else if (m_alloc == buffer<T>::malloc)
    {
        return hamr::malloc_allocator<T>::allocate(n_elem);
    }
#if defined(HAMR_ENABLE_CUDA)
    else if (m_alloc == buffer<T>::cuda)
    {
        return hamr::cuda_malloc_allocator<T>::allocate(n_elem);
    }
    else if (m_alloc == buffer<T>::cuda_uva)
    {
        return hamr::cuda_malloc_uva_allocator<T>::allocate(n_elem);
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
    int ierr = 0;
    if ((m_alloc == buffer<T>::cpp) || (m_alloc == buffer<T>::malloc))
    {
        ierr = hamr::copy_to_cpu_from_cpu(tmp.get(), m_data.get(), m_size);
    }
#if defined(HAMR_ENABLE_CUDA)
    else if ((m_alloc == buffer<T>::cuda) || (m_alloc == buffer<T>::cuda_uva))
    {
        ierr = hamr::copy_to_cuda_from_cuda(tmp.get(), m_data.get(), m_size);
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
    int ierr = 0;
    if ((m_alloc == buffer<T>::cpp) || (m_alloc == buffer<T>::malloc))
    {
        ierr = hamr::copy_to_cpu_from_cpu(tmp.get(), m_data.get(), m_size);
    }
#if defined(HAMR_ENABLE_CUDA)
    else if ((m_alloc == buffer<T>::cuda) || (m_alloc == buffer<T>::cuda_uva))
    {
        ierr = hamr::copy_to_cuda_from_cuda(tmp.get(), m_data.get(), m_size);
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
int buffer<T>::copy(const const_p_buffer<U> &src)
{
    size_t n_vals = src->size();

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
int buffer<T>::copy(const const_p_buffer<U> &src, size_t src_start, size_t n_vals)
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
int buffer<T>::copy(const U *src, size_t src_start, size_t n_vals)
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
    if (new_size > m_capacity)
    {
        if (m_capacity == 0)
            m_capacity = 8;

        while (new_size > m_capacity)
            m_capacity *= 2;

        if (this->reserve(m_capacity))
            return -1;
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
int buffer<T>::append(const const_p_buffer<U> &src, size_t src_start, size_t n_vals)
{
    // allocate space if needed
    if (this->reserve_for_append(n_vals))
        return -1;

    // get the append location
    size_t back = m_size;

    // update state
    m_size += n_vals;

    // copy the value to the back.
    if (this->set(back, src.get(), src_start, n_vals))
        return -1;

    return 0;
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
int buffer<T>::append(const const_p_buffer<U> &src)
{
    size_t n_vals = src->size();

    // allocate space if needed
    if (this->reserve_for_append(n_vals))
        return -1;

    // get the append location
    size_t back = m_size;

    // update state
    m_size += n_vals;

    // copy the value to the back.
    if (this->set(back, src.get(), 0, n_vals))
        return -1;

    return 0;
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
int buffer<T>::set(size_t dest_start, const U *src, size_t src_start, size_t n_vals)
{
    // bounds check
    assert(m_size >= (dest_start + n_vals));

    // copy the values (src is always on the CPU)
    int ierr = 0;
    if ((m_alloc == buffer<T>::cpp) || (m_alloc == buffer<T>::malloc))
    {
        ierr = hamr::copy_to_cpu_from_cpu(m_data.get() + dest_start, src + src_start, n_vals);
    }
#if defined(HAMR_ENABLE_CUDA)
    else if ((m_alloc == buffer<T>::cuda) || (m_alloc == buffer<T>::cuda_uva))
    {
        ierr = hamr::copy_to_cuda_from_cpu(m_data.get() + dest_start, src + src_start, n_vals);
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
int buffer<T>::set(size_t dest_start, const const_p_buffer<U> &src, size_t src_start, size_t n_vals)
{
    // bounds check
    assert((m_size >= (dest_start + n_vals)) && (src->size() >= (src_start + n_vals)));

    // copy the value to the back. buffers can either be on the CPU or GPU
    // and use different technolofies so all permutations must be realized.
    int ierr = 0;
    if ((m_alloc == buffer<T>::cpp) || (m_alloc == buffer<T>::malloc))
    {
        // destination is on the CPU

        if ((src->m_alloc == buffer<T>::cpp) || (src->m_alloc == buffer<T>::malloc))
        {
            // source is on the CPU
            ierr = hamr::copy_to_cpu_from_cpu(m_data.get() + dest_start, src->m_data.get() + src_start, n_vals);
        }
        else if ((src->m_alloc == buffer<T>::cuda) || (src->m_alloc == buffer<T>::cuda_uva))
        {
            // source is on the GPU
            ierr = hamr::copy_to_cpu_from_cuda(m_data.get() + dest_start, src->m_data.get() + src_start, n_vals);
        }
        else
        {
            std::cerr << "ERROR: Invalid allocator type in the source "
                << get_allocator_name(src->m_alloc) << std::endl;
        }
    }
#if defined(HAMR_ENABLE_CUDA)
    else if ((m_alloc == buffer<T>::cuda) || (m_alloc == buffer<T>::cuda_uva))
    {
        // destination is on the GPU

        if ((src->m_alloc == buffer<T>::cpp) || (src->m_alloc == buffer<T>::malloc))
        {
            // source is on the CPU
            ierr = hamr::copy_to_cuda_from_cpu(m_data.get() + dest_start, src->m_data.get() + src_start, n_vals);
        }
        else if ((src->m_alloc == buffer<T>::cuda) || (src->m_alloc == buffer<T>::cuda_uva))
        {
            // source is on the GPU
            ierr = hamr::copy_to_cuda_from_cuda(m_data.get() + dest_start, src->m_data.get() + src_start, n_vals);
        }
        else
        {
            std::cerr << "ERROR: Invalid allocator type in the source "
                << get_allocator_name(src->m_alloc) << std::endl;
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
template <typename U>
int buffer<T>::get(size_t src_start, const U *dest, size_t dest_start, size_t n_vals)
{
    // bounds check
    assert(m_size >= (dest_start + n_vals));

    // copy the values (dest is always on the CPU)
    int ierr = 0;
    if ((m_alloc == buffer<T>::cpp) || (m_alloc == buffer<T>::malloc))
    {
        ierr = hamr::copy_to_cpu_from_cpu(dest + dest_start, m_data.get() + src_start, n_vals);
    }
#if defined(HAMR_ENABLE_CUDA)
    else if ((m_alloc == buffer<T>::cuda) || (m_alloc == buffer<T>::cuda_uva))
    {
        ierr = hamr::copy_to_cuda_from_cpu(dest + dest_start, m_data.get() + src_start, n_vals);
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
int buffer<T>::get(size_t src_start, const const_p_buffer<U> &dest, size_t dest_start, size_t n_vals)
{
    // bounds check
    assert((m_size >= (src_start + n_vals)) && (dest->size() >= (src_start + n_vals)));

    // copy the value to the back. buffers can either be on the CPU or GPU
    // and use different technolofies so all permutations must be realized.
    int ierr = 0;
    if ((m_alloc == buffer<T>::cpp) || (m_alloc == buffer<T>::malloc))
    {
        // destination is on the CPU

        if ((dest->m_alloc == buffer<T>::cpp) || (dest->m_alloc == buffer<T>::malloc))
        {
            // source is on the CPU
            ierr = hamr::copy_to_cpu_from_cpu(dest->m_data.get() + dest_start, m_data.get() + src_start, n_vals);
        }
        else if ((dest->m_alloc == buffer<T>::cuda) || (dest->m_alloc == buffer<T>::cuda_uva))
        {
            // source is on the GPU
            ierr = hamr::copy_to_cpu_from_cuda(dest->m_data.get() + dest_start, m_data.get() + src_start, n_vals);
        }
        else
        {
            std::cerr << "ERROR: Invalid allocator type in the source "
                << get_allocator_name(dest->m_alloc) << std::endl;
        }
    }
#if defined(HAMR_ENABLE_CUDA)
    else if ((m_alloc == buffer<T>::cuda) || (m_alloc == buffer<T>::cuda_uva))
    {
        // destination is on the GPU

        if ((dest->m_alloc == buffer<T>::cpp) || (dest->m_alloc == buffer<T>::malloc))
        {
            // source is on the CPU
            ierr = hamr::copy_to_cuda_from_cpu(dest->m_data.get() + dest_start, m_data.get() + src_start, n_vals);
        }
        else if ((dest->m_alloc == buffer<T>::cuda) || (dest->m_alloc == buffer<T>::cuda_uva))
        {
            // source is on the GPU
            ierr = hamr::copy_to_cuda_from_cuda(dest->m_data.get() + dest_start, m_data.get() + src_start, n_vals);
        }
        else
        {
            std::cerr << "ERROR: Invalid allocator type in the source "
                << get_allocator_name(dest->m_alloc) << std::endl;
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

// --------------------------------------------------------------------------
template <typename T>
std::shared_ptr<T> buffer<T>::get_cpu_accessible()
{
    if ((m_alloc == buffer<T>::cpp) || (m_alloc == buffer<T>::malloc))
    {
        // already on the CPU
        return m_data;
    }
#if defined(HAMR_ENABLE_CUDA)
    else if ((m_alloc == buffer<T>::cuda) || (m_alloc == buffer<T>::cuda_uva))
    {
        // make a copy on the CPU
        std::shared_ptr<T> tmp = hamr::malloc_allocator<T>::allocate(m_size);

        if (hamr::copy_to_cpu_from_cuda(tmp.get(), m_data.get(), m_size))
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

// --------------------------------------------------------------------------
template <typename T>
std::shared_ptr<const T> buffer<T>::get_cuda_accessible() const
{
    return const_cast<buffer<T>*>(this)->get_cuda_accessible();
}

// --------------------------------------------------------------------------
template <typename T>
std::shared_ptr<T> buffer<T>::get_cuda_accessible()
{
#if !defined(HAMR_ENABLE_CUDA)
    std::cerr << "ERROR: get_cuda_accessible failed, CUDA is not available." << std::endl;
    return nullptr;
#else
    if ((m_alloc == buffer<T>::cpp) || (m_alloc == buffer<T>::malloc))
    {
        // make a copy on the CPU
        std::shared_ptr<T> tmp = hamr::cuda_malloc_allocator<T>::allocate(m_size);

        if (hamr::copy_to_cuda_from_cpu(tmp.get(), m_data.get(), m_size))
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
int buffer<T>::print()
{
    std::cerr << "m_alloc = " << get_allocator_name(m_alloc)
        << ", m_size = " << m_size << ", m_capacity = " << m_capacity
        << ", m_data = ";

    if (m_size)
    {
        if ((m_alloc == buffer<T>::cpp) || (m_alloc == buffer<T>::malloc))
        {
            std::cerr << m_data[0];
            for (size_t i = 1; i < m_size; ++i)
                std::cerr << ", " << m_data[i];
            std::cerr << std::endl;
        }
#if defined(HAMR_ENABLE_CUDA)
        else if ((m_alloc == buffer<T>::cuda) || (m_alloc == buffer<T>::cuda_uva))
        {
            hamr::cuda_print(m_data, m_size);
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


/*
// --------------------------------------------------------------------------
template <typename T>
int buffer<T>::destruct(T *ptr, size_t n_elem)
{
    if ((m_alloc == buffer<T>::cpp) || (m_alloc == buffer<T>::malloc))
    {
        for (size_t i = 0; i < n_elem; ++i)
        {
            T[i]->~T();
        }
    }
#if defined(HAMR_ENABLE_CUDA)
    else if ((m_alloc == buffer<T>::cuda) || (m_alloc == buffer<T>::cuda_uva))
    {
        size_t n_elem = end - start + 1;

        // get launch parameters
        int device_id = -1;
        dim3 block_grid;
        int n_blocks = 0;
        dim3 thread_grid = 0;
        if (hamr::get_launch_props(-1, n_elem, 8, block_grid, n_blocks, thread_grid))
        {
            std::cerr << "ERROR: Failed to determine launch properties. "
                << cudaGetErrorString(ierr) << std::endl;
            return -1;
        }

        // invoke the kernel
        buffer_internals::destruct<<<block_grid, thread_grid>>>(ptr, n_elem)
        if ((ierr = cudaGetLastError()) != cudaSuccess)
        {
            std::cerr << "ERROR: Failed to launch the construct kernel. "
                << cudaGetErrorString(ierr) << std::endl;
            return -1;
        }
    }
#endif
    else
    {
        std::cerr << "ERROR: Invalid allocator type "
            << buffer_internals::get_allocator_name(alloc) << std::endl;
        return -1;
    }

    return 0;
}
*/

/**
// --------------------------------------------------------------------------
template <typename T>
int buffer<T>::construct(T *ptr, size_t n_elem, const T &val)
{
    if ((m_alloc == buffer<T>::cpp) || (m_alloc == buffer<T>::malloc))
    {
        for (size_t i = 0; i < n_elem; ++i)
        {
            new (&T[i]) T(val);
        }
    }
#if defined(HAMR_ENABLE_CUDA)
    else if ((m_alloc == buffer<T>::cuda) || (m_alloc == buffer<T>::cuda_uva))
    {
        // get launch parameters
        int device_id = -1;
        dim3 block_grid;
        int n_blocks = 0;
        dim3 thread_grid = 0;
        if (hamr::get_launch_props(-1, n_elem, 8, block_grid, n_blocks, thread_grid))
        {
            std::cerr << "ERROR: Failed to determine launch properties. "
                << cudaGetErrorString(ierr) << std::endl;
            return -1;
        }

        // invoke the kernel
        buffer_internals::construct<<<block_grid, thread_grid>>>(ptr, n_elem, val)
        if ((ierr = cudaGetLastError()) != cudaSuccess)
        {
            std::cerr << "ERROR: Failed to launch the construct kernel. "
                << cudaGetErrorString(ierr) << std::endl;
            return -1;
        }
    }
#endif
    else
    {
        std::cerr << "ERROR: Invalid allocator type "
            << buffer_internals::get_allocator_name(alloc) << std::endl;
        return -1;
    }

    return 0;
}

// --------------------------------------------------------------------------
template <typename T>
int buffer<T>::construct(T *ptr, size_t n_elem)
{
    if ((m_alloc == buffer<T>::cpp) || (m_alloc == buffer<T>::malloc))
    {
        for (size_t i = 0; i < n_elem; ++i)
        {
            new (&T[i]) T;
        }
    }
#if defined(HAMR_ENABLE_CUDA)
    else if ((m_alloc == buffer<T>::cuda) || (m_alloc == buffer<T>::cuda_uva))
    {
        size_t n_elem = end - start + 1;

        // get launch parameters
        int device_id = -1;
        dim3 block_grid;
        int n_blocks = 0;
        dim3 thread_grid = 0;
        if (hamr::get_launch_props(-1, n_elem, 8, block_grid, n_blocks, thread_grid))
        {
            std::cerr << "ERROR: Failed to determine launch properties. "
                << cudaGetErrorString(ierr) << std::endl;
            return -1;
        }

        // invoke the kernel
        buffer_internals::construct<<<block_grid, thread_grid>>>(ptr + start, n_elem)
        if ((ierr = cudaGetLastError()) != cudaSuccess)
        {
            std::cerr << "ERROR: Failed to launch the construct kernel. "
                << cudaGetErrorString(ierr) << std::endl;
            return -1;
        }
    }
#endif
    else
    {
        std::cerr << "ERROR: Invalid allocator type "
            << buffer_internals::get_allocator_name(alloc) << std::endl;
        return -1;
    }

    return 0;
}
*
*
*/
/*
// --------------------------------------------------------------------------
template <typename T>
int buffer<T>::copy(T *dest, const T *src, size_t n_elem)
{
    if ((m_alloc == buffer<T>::cpp) || (m_alloc == buffer<T>::malloc))
    {
        for (size_t i = 0; i < n_elem; ++i)
        {
            dest[i] = src[i];
        }
    }
#if defined(HAMR_ENABLE_CUDA)
    else if ((m_alloc == buffer<T>::cuda) || (m_alloc == buffer<T>::cuda_uva))
    {
        size_t n_elem = end - start + 1;

        // get launch parameters
        int device_id = -1;
        dim3 block_grid;
        int n_blocks = 0;
        dim3 thread_grid = 0;
        if (hamr::get_launch_props(-1, n_elem, 8, block_grid, n_blocks, thread_grid))
        {
            std::cerr << "ERROR: Failed to determine launch properties. "
                << cudaGetErrorString(ierr) << std::endl;
            return -1;
        }

        // invoke the kernel
        buffer_internals::copy<<<block_grid, thread_grid>>>(dest, src, n_elem)
        if ((ierr = cudaGetLastError()) != cudaSuccess)
        {
            std::cerr << "ERROR: Failed to launch the construct kernel. "
                << cudaGetErrorString(ierr) << std::endl;
            return -1;
        }
    }
#endif
    else
    {
        std::cerr << "ERROR: Invalid allocator type "
            << buffer_internals::get_allocator_name(alloc) << std::endl;
        return -1;
    }

    return 0;
}


/// initialize an array on the GPU using CUDA
template <typename T>
__global__
void fill(T *dest, size_t n_elem, T val)
{
    unsigned long i = hamr::thread_id_to_array_index();

    if (i >= n_elem)
        return;

    dest[i] = val;
}


}


    if (m_alloc == buffer<T>::cpp)
    {

    }
    else if (m_alloc == buffer<T>::malloc)
    {

    }
    else if (m_alloc == buffer<T>::cuda)
    {

    }
    else if (m_alloc == buffer<T>::cuda_uva)
    {

    }
    else
    {
        std::cerr << "ERROR: Invalid allocator "
            << buffer_internals::get_allocator_name(m_alloc) << std::endl;
    }


*/

/*
// --------------------------------------------------------------------------
template <typename T>
template <typename U>
int buffer<T>::get(const const_p_buffer<U> &dest) const;
*/

/*
// --------------------------------------------------------------------------
template <typename T>
template <typename U>
int buffer<T>::get(size_t start, size_t end, const p_buffer<U> &dest) const;
*/

/*
// --------------------------------------------------------------------------
template <typename T>
template <typename U>
int buffer<T>::get(size_t i, const U &val) const
{
    // bounds check
    assert(i < m_size);

    // copy the value from ith element
    int ierr = 0;
    if ((m_alloc == buffer<T>::cpp) || (m_alloc == buffer<T>::malloc))
    {
        ierr = hamr::copy_to_cpu_from_cpu(&val, m_data.get() + i, 1);
    }
#if defined(HAMR_ENABLE_CUDA)
    else if ((m_alloc == buffer<T>::cuda) || (m_alloc == buffer<T>::cuda_uva))
    {
        ierr = hamr::copy_to_cuda_from_cpu(m_data.get() + i, &val, 1);
    }
#endif
    else
    {
        std::cerr << "ERROR: Invalid allocator type "
            << get_allocator_name(alloc) << std::endl;
    }

    // check for errors
    if (ierr)
        return -1;

    return 0;
}

// --------------------------------------------------------------------------
template <typename T>
template <typename U>
int buffer<T>::get(const std::vector<U> &dest) const;
*/

/*
// --------------------------------------------------------------------------
template <typename T>
template <typename U>
int buffer<T>::set(const const_p_buffer<U> &src)
{
    size_t n_vals = src->size();

    // bounds check
    assert(m_size >= n_vals);

    // copy the value to the back. buffers can either be on the CPU or GPU
    // and use different technolofies so all permutations must be realized.
    int ierr = 0;
    if ((m_alloc == buffer<T>::cpp) || (m_alloc == buffer<T>::malloc))
    {
        // destination is on the CPU

        if ((vals->m_alloc == buffer<T>::cpp) || (vals->m_alloc == buffer<T>::malloc))
        {
            // source is on the CPU
            ierr = hamr::copy_to_cpu_from_cpu(m_data.get(), vals->m_data.get(), n_vals);
        }
        else if ((vals->m_alloc == buffer<T>::cuda) || (vals->m_alloc == buffer<T>::cuda_uva))
        {
            // source is on the GPU
            ierr = hamr::copy_to_cpu_from_cuda(m_data.get(), vals->m_data.get(), n_vals);
        }
        else
        {
            std::cerr << "ERROR: Invalid allocator type in the source "
                << get_allocator_name(vals->m_alloc) << std::endl;
        }
    }
#if defined(HAMR_ENABLE_CUDA)
    else if ((m_alloc == buffer<T>::cuda) || (m_alloc == buffer<T>::cuda_uva))
    {
        // destination is on the GPU

        if ((vals->m_alloc == buffer<T>::cpp) || (vals->m_alloc == buffer<T>::malloc))
        {
            // source is on the CPU
            ierr = hamr::copy_to_cuda_from_cpu(m_data.get(), vals->m_data.get(), n_vals);
        }
        else if ((vals->m_alloc == buffer<T>::cuda) || (vals->m_alloc == buffer<T>::cuda_uva))
        {
            // source is on the GPU
            ierr = hamr::copy_to_cuda_from_cuda(m_data.get(), vals->m_data.get(), n_vals);
        }
        else
        {
            std::cerr << "ERROR: Invalid allocator type in the source "
                << get_allocator_name(vals->m_alloc) << std::endl;
        }
    }
#endif
    else
    {
        std::cerr << "ERROR: Invalid allocator type "
            << get_allocator_name(alloc) << std::endl;
    }

    // check for errors
    if (ierr)
        return -1;

    return 0;
}
*/

/*
// --------------------------------------------------------------------------
template <typename T>
template <typename U>
int buffer<T>::append(const U &val)
{
    // allocate space if needed
    if (this->reserve_for_append(1))
        return -1;

    // copy the value to the back
    int ierr = 0;
    if ((m_alloc == buffer<T>::cpp) || (m_alloc == buffer<T>::malloc))
    {
        ierr = hamr::copy_to_cpu_from_cpu(m_data.get() + m_size, &val, 1);
    }
#if defined(HAMR_ENABLE_CUDA)
    else if ((m_alloc == buffer<T>::cuda) || (m_alloc == buffer<T>::cuda_uva))
    {
        ierr = hamr::copy_to_cuda_from_cpu(m_data.get() + m_size, &val, 1);
    }
#endif
    else
    {
        std::cerr << "ERROR: Invalid allocator type "
            << get_allocator_name(alloc) << std::endl;
    }

    // check for errors
    if (ierr)
        return -1;

    // update state
    m_size += 1;

    return 0;
}
 * old append
    int ierr = 0;
    if ((m_alloc == buffer<T>::cpp) || (m_alloc == buffer<T>::malloc))
    {
        ierr = hamr::copy_to_cpu_from_cpu(m_data.get() + m_size, src, n_vals);
    }
#if defined(HAMR_ENABLE_CUDA)
    else if ((m_alloc == buffer<T>::cuda) || (m_alloc == buffer<T>::cuda_uva))
    {
        ierr = hamr::copy_to_cuda_from_cpu(m_data.get() + m_size, src, n_vals);
    }
#endif
    else
    {
        std::cerr << "ERROR: Invalid allocator type "
            << get_allocator_name(alloc) << std::endl;
    }

    // check for errors
    if (ierr)
        return -1;

 *
 *
 * old append
    //
    //
    // buffers can either be on the CPU or GPU
    // and use different technolofies so all permutations must be realized.
    int ierr = 0;
    if ((m_alloc == buffer<T>::cpp) || (m_alloc == buffer<T>::malloc))
    {
        // destination is on the CPU

        if ((vals->m_alloc == buffer<T>::cpp) || (vals->m_alloc == buffer<T>::malloc))
        {
            // source is on the CPU
            ierr = hamr::copy_to_cpu_from_cpu(m_data.get() + m_size, vals->m_data.get(), n_vals);
        }
        else if ((vals->m_alloc == buffer<T>::cuda) || (vals->m_alloc == buffer<T>::cuda_uva))
        {
            // source is on the GPU
            ierr = hamr::copy_to_cpu_from_cuda(m_data.get() + m_size, vals->m_data.get(), n_vals);
        }
        else
        {
            std::cerr << "ERROR: Invalid allocator type in the source "
                << get_allocator_name(vals->m_alloc) << std::endl;
        }
    }
#if defined(HAMR_ENABLE_CUDA)
    else if ((m_alloc == buffer<T>::cuda) || (m_alloc == buffer<T>::cuda_uva))
    {
        // destination is on the GPU

        if ((vals->m_alloc == buffer<T>::cpp) || (vals->m_alloc == buffer<T>::malloc))
        {
            // source is on the CPU
            ierr = hamr::copy_to_cuda_from_cpu(m_data.get() + m_size, vals->m_data.get(), n_vals);
        }
        else if ((vals->m_alloc == buffer<T>::cuda) || (vals->m_alloc == buffer<T>::cuda_uva))
        {
            // source is on the GPU
            ierr = hamr::copy_to_cuda_from_cuda(m_data.get() + m_size, vals->m_data.get(), n_vals);
        }
        else
        {
            std::cerr << "ERROR: Invalid allocator type in the source "
                << get_allocator_name(vals->m_alloc) << std::endl;
        }
    }
#endif
    else
    {
        std::cerr << "ERROR: Invalid allocator type "
            << get_allocator_name(alloc) << std::endl;
    }

    // check for errors
    if (ierr)
        return -1;
// --------------------------------------------------------------------------
template <typename T>
template <typename U>
int buffer<T>::set(size_t i, const U &val)
{
    // bounds check
    assert(i < m_size);

    // copy the value to ith element
    int ierr = 0;
    if ((m_alloc == buffer<T>::cpp) || (m_alloc == buffer<T>::malloc))
    {
        ierr = hamr::copy_to_cpu_from_cpu(m_data.get() + i, &val, 1);
    }
#if defined(HAMR_ENABLE_CUDA)
    else if ((m_alloc == buffer<T>::cuda) || (m_alloc == buffer<T>::cuda_uva))
    {
        ierr = hamr::copy_to_cuda_from_cpu(m_data.get() + i, &val, 1);
    }
#endif
    else
    {
        std::cerr << "ERROR: Invalid allocator type "
            << get_allocator_name(alloc) << std::endl;
    }

    // check for errors
    if (ierr)
        return -1;

    return 0;
}
*/

/*
// --------------------------------------------------------------------------
template <typename T>
template <typename U>
int buffer<T>::set(const std::vector<U> &src)
{
    size_t n_vals = vals.size();

    // bounds check
    assert(m_size >= n_vals);

    // copy the values (src is always on the CPU)
    int ierr = 0;
    if ((m_alloc == buffer<T>::cpp) || (m_alloc == buffer<T>::malloc))
    {
        ierr = hamr::copy_to_cpu_from_cpu(m_data.get(), vals.data(), n_vals);
    }
#if defined(HAMR_ENABLE_CUDA)
    else if ((m_alloc == buffer<T>::cuda) || (m_alloc == buffer<T>::cuda_uva))
    {
        ierr = hamr::copy_to_cuda_from_cpu(m_data.get(), vals.data(), n_vals);
    }
#endif
    else
    {
        std::cerr << "ERROR: Invalid allocator type "
            << get_allocator_name(alloc) << std::endl;
    }

    // check for errors
    if (ierr)
        return -1;

    return 0;
}
*/

}
#endif
