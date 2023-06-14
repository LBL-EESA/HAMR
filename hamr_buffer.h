#ifndef buffer_h
#define buffer_h

#include "hamr_config.h"
#include "hamr_buffer_allocator.h"
#include "hamr_buffer_transfer.h"
#include "hamr_stream.h"

#include <memory>
#include <type_traits>

/// heterogeneous accelerator memory resource
namespace hamr
{

/** @brief A technology agnostic buffer that manages memory on the host, GPUs,
 * and other accelerators.
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
    /** An enumeration for the type of allocator to use for memory allocations.
     * See ::buffer_allocator.
     */
    using allocator = buffer_allocator;

    /** An enumeration for the types of transfer supported. See
     * ::buffer_transfer
     */
    using transfer = buffer_transfer;

    /** Construct an empty buffer.
     *
     * @param[in] alloc   a ::buffer_allocator indicates what technology
     *                    manages the data internally
     * @param[in] strm    a ::stream object used to order operations
     * @param[in] sync    a ::buffer_transfer specifies synchronous or
     *                    asynchronous behavior.
     */
    buffer(allocator alloc, const hamr::stream &strm, transfer sync = transfer::async);

    /** Construct an empty buffer. This constructor will result in the default
     * stream for the chosen technology with transfer::sync_host mode which
     * synchronizes after data movement from a device to the host.
     *
     * @param[in] alloc   a ::buffer_allocator indicates what technology
     *                    manages the data internally
     */
    buffer(allocator alloc) : buffer(alloc, stream(), transfer::sync_host) {}

    /** Construct a buffer with storage allocated but unitialized.
     *
     * @param[in] alloc   a ::buffer_allocator indicates what technology
     *                    manages the data internally
     * @param[in] strm    a ::stream object used to order operations
     * @param[in] sync    a ::buffer_transfer specifies synchronous or
     *                    asynchronous behavior.
     * @param[in] n_elem  the initial size of the new buffer
     */
    buffer(allocator alloc, const hamr::stream &strm, transfer sync, size_t n_elem);

    /** Construct a buffer configured for asynchronous data transfers, with
     * storage allocated, but unitialized.
     *
     * @param[in] alloc   a ::buffer_allocator indicates what technology
     *                    manages the data internally
     * @param[in] strm    a ::stream object used to order operations
     * @param[in] n_elem  the initial size of the new buffer
     */
    buffer(allocator alloc, const hamr::stream &strm, size_t n_elem)
        : buffer(alloc, strm, transfer::async, n_elem) {}

    /** Construct a buffer with storage allocated but unitialized. This
     * constructor will result in the default stream for the chosen technology
     * with transfer::sync_host mode which synchronizes after data movement from
     * a device to the host.
     *
     * @param[in] alloc   a ::buffer_allocator indicates what technology
     *                    manages the data internally
     * @param[in] n_elem  the initial size of the new buffer
     */
    buffer(allocator alloc, size_t n_elem) :
        buffer(alloc, stream(), transfer::sync_host, n_elem) {}

    /** Construct a buffer with storage allocated and initialized to a single
     * value.
     *
     * @param[in] alloc   a ::buffer_allocator indicates what technology
     *                    manages the data internally
     * @param[in] strm    a ::stream object used to order operations
     * @param[in] sync    a ::buffer_transfer specifies synchronous or
     *                    asynchronous behavior.
     * @param[in] n_elem  the initial size of the new buffer
     * @param[in] val     an single value used to initialize the buffer
     *                    contents
     */
    buffer(allocator alloc, const hamr::stream &strm,
        transfer sync, size_t n_elem, const T &val);

    /** Construct a buffer configured for asynchronous data movement, with
     * storage allocated, and initialized to a single value.
     *
     * @param[in] alloc   a ::buffer_allocator indicates what technology
     *                    manages the data internally
     * @param[in] strm    a ::stream object used to order operations
     * @param[in] n_elem  the initial size of the new buffer
     * @param[in] val     an single value used to initialize the buffer
     *                    contents
     */
    buffer(allocator alloc, const hamr::stream &strm, size_t n_elem, const T &val)
        : buffer(alloc, strm, transfer::async, n_elem, val) {}

    /** Construct a buffer with storage allocated and initialized to a single
     * value. This constructor will result in the default stream for the chosen
     * technology with transfer::sync_host mode which synchronizes after data
     * movement from a device to the host. For fully asynchronous data transfers
     * one must explicitly prtovide a stream and specify the asynchronous mode.
     *
     * @param[in] alloc   a ::buffer_allocator indicates what technology
     *                    manages the data internally
     * @param[in] n_elem  the initial size of the new buffer
     * @param[in] val     an single value used to initialize the buffer
     *                    contents
     */
    buffer(allocator alloc, size_t n_elem, const T &val) :
        buffer(alloc, stream(), transfer::sync_host, n_elem, val) {}

    /** Construct a buffer with storage allocated and initialized to the array
     * of values. This array is always assumed to be accessible on the host. Use
     * one of the zero-copy constructors if the data is already accessible on
     * the device.
     *
     * @param[in] alloc   a ::buffer_allocator indicates what technology
     *                    manages the data internally
     * @param[in] strm    a ::stream object used to order operations
     * @param[in] sync    a ::buffer_transfer specifies synchronous or
     *                    asynchronous behavior.
     * @param[in] n_elem  the initial size of the new buffer and number of
     *                    elements in the array pointed to by vals
     * @param[in] vals    an array of values accessible on the host used to
     *                    initialize the buffer contents
     */
    buffer(allocator alloc, const hamr::stream &strm,
        transfer sync, size_t n_elem, const T *vals);

    /** Construct a buffer configured for asynchronous data movement, with
     * storage allocated, and initialized to the array of values. This array is
     * always assumed to be accessible on the host. Use one of the zero-copy
     * constructors if the data is already accessible on the device.
     *
     * @param[in] alloc   a ::buffer_allocator indicates what technology
     *                    manages the data internally
     * @param[in] strm    a ::stream object used to order operations
     * @param[in] n_elem  the initial size of the new buffer and number of
     *                    elements in the array pointed to by vals
     * @param[in] vals    an array of values accessible on the host used to
     *                    initialize the buffer contents
     */
    buffer(allocator alloc, const hamr::stream &strm, size_t n_elem, const T *vals)
        : buffer(alloc, strm, transfer::async, n_elem, vals) {}

    /** Construct a buffer with storage allocated and initialized to the array
     * of values. This array is always assumed to be accessible on the host. Use
     * one of the zero-copy constructors if the data is already accessible on
     * the device. This constructor will result in the default stream for the
     * chosen technology with transfer::sync_host mode which synchronizes after
     * data movement from a device to the host.
     *
     * @param[in] alloc   a ::buffer_allocator indicates what technology
     *                    manages the data internally
     * @param[in] n_elem  the initial size of the new buffer and number of
     *                    elements in the array pointed to by vals
     * @param[in] vals    an array of values accessible on the host used to
     *                    initialize the buffer contents
     */
    buffer(allocator alloc, size_t n_elem, const T *vals) :
        buffer(alloc, stream(), transfer::sync_host, n_elem, vals) {}

    /** Construct by directly providing the buffer contents. This can be used
     * for zero-copy transfer of data.  One must also name the allocator type
     * and device owning the data.  In addition for new allocations the
     * allocator type and owner are used internally to know how to
     * automatically move data during inter technology transfers.
     *
     * @param[in] alloc a ::buffer_allocator indicating the technology
     *                  backing the pointer
     * @param[in] strm  a ::stream object used to order operations
     * @param[in] sync  a ::buffer_transfer specifies synchronous or
     *                  asynchronous behavior.
     * @param[in] size  the number of elements in the array pointed to by ptr
     * @param[in] owner the device owning the memory, -1 for host. if the
     *                  allocator is a GPU allocator and -1 is passed the
     *                  driver API is used to determine the device that
     *                  allocated the memory.
     * @param[in] ptr   a pointer to the array
     * @param[in] df    a function `void df(void*ptr)` used to delete the array
     *                  when this instance is finished.
     */
    template <typename delete_func_t>
    buffer(allocator alloc, const hamr::stream &strm, transfer sync,
        size_t size, int owner, T *ptr, delete_func_t df);

    /** Construct by directly providing the buffer contents. This can be used
     * for zero-copy transfer of data.  One must also name the allocator type
     * and device owning the data.  In addition for new allocations the
     * allocator type and owner are used internally to know how to
     * automatically move data during inter technology transfers. The buffer is
     * configured for asynchronous data transfers.
     *
     * @param[in] alloc a ::buffer_allocator indicating the technology
     *                  backing the pointer
     * @param[in] strm  a ::stream object used to order operations
     * @param[in] size  the number of elements in the array pointed to by ptr
     * @param[in] owner the device owning the memory, -1 for host. if the
     *                  allocator is a GPU allocator and -1 is passed the
     *                  driver API is used to determine the device that
     *                  allocated the memory.
     * @param[in] ptr   a pointer to the array
     * @param[in] df    a function `void df(void*ptr)` used to delete the array
     *                  when this instance is finished.
     */
    template <typename delete_func_t>
    buffer(allocator alloc, const hamr::stream &strm, size_t size,
        int owner, T *ptr, delete_func_t df)
            : buffer(alloc, strm, transfer::async, size, owner, ptr, df) {}

    /** Construct by directly providing the buffer contents. This can be used
     * for zero-copy transfer of data.  One must also name the allocator type
     * and device owning the data.  In addition for new allocations the
     * allocator type and owner are used internally to know how to
     * automatically move data during inter technology transfers. This
     * constructor will result in the default stream for the chosen technology
     * with transfer::sync_host mode which synchronizes after data movement from
     * a device to the host.
     *
     * @param[in] alloc a ::buffer_allocator indicating the technology
     *                  backing the pointer
     * @param[in] size  the number of elements in the array pointed to by ptr
     * @param[in] owner the device owning the memory, -1 for host. if the
     *                  allocator is a GPU allocator and -1 is passed the
     *                  driver API is used to determine the device that
     *                  allocated the memory.
     * @param[in] ptr   a pointer to the array
     * @param[in] df    a function `void df(void*ptr)` used to delete the array
     *                  when this instance is finished.
     */
    template <typename delete_func_t>
    buffer(allocator alloc, size_t size, int owner, T *ptr, delete_func_t df)
        : buffer(alloc, stream(), transfer::sync_host, size, owner, ptr, df) {}

    /** Construct by directly providing the buffer contents. This can be used
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
     * @param[in] strm  a ::stream object used to order operations
     * @param[in] sync  a ::buffer_transfer specifies synchronous or
     *                  asynchronous behavior.
     * @param[in] size  the number of elements in the array pointed to by ptr
     * @param[in] owner the device owning the memory, -1 for host. if the
     *                  allocator is a GPU allocator and -1 is passed the
     *                  driver API is used to determine the device that
     *                  allocated the memory.
     * @param[in] ptr   a pointer to the array
     */
    buffer(allocator alloc, const hamr::stream &strm,
        transfer sync, size_t size, int owner, T *ptr);

    /** Construct by directly providing the buffer contents. This can be used
     * for zero-copy transfer of data.  One must also name the allocator type
     * and device owning the data.  In addition for new allocations the
     * allocator type and owner are used internally to know how to
     * automatically move data during inter technology transfers.
     * The pass ::buffer_allocator is used to create the deleter that will be
     * called when this instance is finished with the memeory. Use this
     * constructor to transfer ownership of the array. The buffer is configured
     * for asynchronous data transfers.
     *
     * @param[in] alloc a ::buffer_allocator indicating the technology
     *                  backing the pointer
     * @param[in] strm  a ::stream object used to order operations
     * @param[in] size  the number of elements in the array pointed to by ptr
     * @param[in] owner the device owning the memory, -1 for host. if the
     *                  allocator is a GPU allocator and -1 is passed the
     *                  driver API is used to determine the device that
     *                  allocated the memory.
     * @param[in] ptr   a pointer to the array
     */
    buffer(allocator alloc, const hamr::stream &strm, size_t size, int owner, T *ptr)
        : buffer(alloc, strm, transfer::async, size, owner, ptr) {}

    /** construct by directly providing the buffer contents. This can be used
     * for zero-copy transfer of data.  One must also name the allocator type
     * and device owning the data.  In addition for new allocations the
     * allocator type and owner are used internally to know how to
     * automatically move data during inter technology transfers.  The pass
     * ::buffer_allocator is used to create the deleter that will be called
     * when this instance is finished with the memeory. Use this constructor to
     * transfer ownership of the array.  This constructor will result in the
     * default stream for the chosen technology with transfer::sync_host mode
     * which synchronizes after data movement from a device to the host.
     *
     * @param[in] alloc a ::buffer_allocator indicating the technology
     *                  backing the pointer
     * @param[in] size  the number of elements in the array pointed to by ptr
     * @param[in] owner the device owning the memory, -1 for host. if the
     *                  allocator is a GPU allocator and -1 is passed the
     *                  driver API is used to determine the device that
     *                  allocated the memory.
     * @param[in] ptr   a pointer to the array
     */
    buffer(allocator alloc, size_t size, int owner, T *ptr) :
        buffer(alloc, stream(), transfer::sync_host, size, owner, ptr) {}

    /** Construct by directly providing the buffer contents. This can be used
     * for zero-copy transfer of data.  One must also name the allocator type
     * and device owning the data.  In addition for new allocations the
     * allocator type and owner are used internally to know how to
     * automatically move data during inter technology transfers.
     *
     * @param[in] alloc a ::buffer_allocator indicating the technology
     *                  backing the pointer
     * @param[in] strm  a ::stream object used to order operations
     * @param[in] sync  a ::buffer_transfer specifies synchronous or
     *                  asynchronous behavior.
     * @param[in] size  the number of elements in the array pointed to by ptr
     * @param[in] owner the device owning the memory, -1 for host. if the
     *                  allocator is a GPU allocator and -1 is passed the
     *                  driver API is used to determine the device that
     *                  allocated the memory.
     * @param[in] data  a shared pointer managing the data
     */
    buffer(allocator alloc, const hamr::stream &strm, transfer sync,
        size_t size, int owner, const std::shared_ptr<T> &data);

    /** Construct by directly providing the buffer contents. This can be used
     * for zero-copy transfer of data.  One must also name the allocator type
     * and device owning the data.  In addition for new allocations the
     * allocator type and owner are used internally to know how to
     * automatically move data during inter technology transfers. The buffer is
     * configured for asynchronous data transfers.
     *
     * @param[in] alloc a ::buffer_allocator indicating the technology
     *                  backing the pointer
     * @param[in] strm  a ::stream object used to order operations
     * @param[in] size  the number of elements in the array pointed to by ptr
     * @param[in] owner the device owning the memory, -1 for host. if the
     *                  allocator is a GPU allocator and -1 is passed the
     *                  driver API is used to determine the device that
     *                  allocated the memory.
     * @param[in] data  a shared pointer managing the data
     */
    buffer(allocator alloc, const hamr::stream &strm,
        size_t size, int owner, const std::shared_ptr<T> &data)
            : buffer(alloc, strm, transfer::async, size, owner, data) {}

    /** Construct by directly providing the buffer contents. This can be used
     * for zero-copy transfer of data.  One must also name the allocator type
     * and device owning the data.  In addition for new allocations the
     * allocator type and owner are used internally to know how to
     * automatically move data during inter technology transfers.  This
     * constructor will result in the default stream for the chosen technology
     * with transfer::sync_host mode which synchronizes after data movement from
     * a device to the host.
     *
     * @param[in] alloc a ::buffer_allocator indicating the technology
     *                  backing the pointer
     * @param[in] size  the number of elements in the array pointed to by ptr
     * @param[in] owner the device owning the memory, -1 for host. if the
     *                  allocator is a GPU allocator and -1 is passed the
     *                  driver API is used to determine the device that
     *                  allocated the memory.
     * @param[in] data  a shared pointer managing the data
     */
    buffer(allocator alloc, size_t size, int owner, const std::shared_ptr<T> &data)
        : buffer(alloc, stream(), transfer::sync_host, size, owner, data) {}

    /// copy construct from the passed buffer
    template <typename U>
    buffer(const buffer<U> &other);

    /// copy construct from the passed buffer
    buffer(const buffer<T> &other);

    /** Copy construct from the passed buffer, while specifying a potentially
     * different allocator, stream, and synchronization behavior.
     *
     * @param[in] alloc a ::buffer_allocator indicates what technology
     *                  manages the data internally
     * @param[in] strm  a ::stream object used to order operations
     * @param[in] sync  a ::buffer_transfer specifies synchronous or
     *                  asynchronous behavior.
     */
    template <typename U>
    buffer(allocator alloc, const hamr::stream &strm,
        transfer sync, const buffer<U> &other);

    /** Copy construct from the passed buffer, while specifying a potentially
     * different allocator, stream, and synchronization behavior. The buffer is
     * configured for asynchronous data transfers.
     *
     * @param[in] alloc a ::buffer_allocator indicates what technology
     *                  manages the data internally
     * @param[in] strm  a ::stream object used to order operations
     */
    template <typename U>
    buffer(allocator alloc, const hamr::stream &strm, const buffer<U> &other)
        : buffer(alloc, strm, transfer::async, other) {}

    /** Copy construct from the passed buffer, while specifying a potentially
     * different allocator, stream, and synchronization behavior. This
     * constructor will result in the default stream for the chosen technology
     * with transfer::sync_host mode which synchronizes after data movement from
     * a device to the host.
     *
     * @param[in] alloc a ::buffer_allocator indicates what technology
     *                  manages the data internally
     * @param[in] strm  a ::stream object used to order operations
     * @param[in] sync  a ::buffer_transfer specifies synchronous or
     *                  asynchronous behavior.
     */
    template <typename U>
    buffer(allocator alloc, const buffer<U> &other) :
        buffer(alloc, other.m_stream, other.m_sync, other) {}

#if !defined(SWIG)
    /// Move construct from the passed buffer.
    buffer(buffer<T> &&other);

    /** Move construct from the passed buffer,  while specifying a potentially
     * different allocator, owner, stream, and synchronization behavior.  The
     * move occurs only if the allocators and owners match, otherwise a copy is
     * made. For non-host allocators, the active device is used to set the owner
     * of the new object prior to the atempted move.
     *
     * @param[in] alloc a ::buffer_allocator indicates what technology
     *                  manages the data internally
     * @param[in] strm  a ::stream object used to order operations
     * @param[in] sync  a ::buffer_transfer specifies synchronous or
     *                  asynchronous behavior.
     */
    buffer(allocator alloc, const hamr::stream &strm, transfer sync, buffer<T> &&other);

    /** Move construct from the passed buffer,  while specifying a potentially
     * different allocator, owner, stream, and synchronization behavior.  The
     * move occurs only if the allocators and owners match, otherwise a copy is
     * made. For non-host allocators, the active device is used to set the owner
     * of the new object prior to the atempted move. The buffer is configured
     * for asynchronous data transfers.
     *
     * @param[in] alloc a ::buffer_allocator indicates what technology
     *                  manages the data internally
     * @param[in] strm  a ::stream object used to order operations
     */
    buffer(allocator alloc, const hamr::stream &strm, buffer<T> &&other)
        : buffer(alloc, strm, transfer::async, std::move(other)) {}

    /** Move construct from the passed buffer,  while specifying a potentially
     * different allocator, owner, stream, and synchronization behavior.  The
     * move occurs only if the allocators and owners match, otherwise a copy is
     * made. For non-host allocators, the active device is used to set the owner
     * of the new object prior to the atempted move.  This constructor will
     * result in the default stream for the chosen technology with
     * transfer::sync_host mode which synchronizes after data movement from a
     * device to the host.
     *
     * @param[in] alloc a ::buffer_allocator indicates what technology
     *                  manages the data internally
     */
    buffer(allocator alloc, buffer<T> &&other) :
        buffer(alloc, other.m_stream, other.m_sync, std::move(other)) {}

    /** move assign from the other buffer.  The target buffer's allocator,
     * stream, and device transfer mode are preserved.  if this and the passed
     * buffer have the same type, allocator, and owner the passed buffer is
     * moved. If this and the passed buffer have different allocators or owners
     * this allocator is used to allocate space and the data will be copied.
     * if this and the passed buffer have different types elements are cast to
     * this type as they are copied.
     */
    void operator=(buffer<T> &&other);
#endif

    /** Allocate space and copy the contents of another buffer. The allocator,
     * owner, stream, and sychronization mode of the receiving object are
     * unmodified by this operation. Thus one may move data around the system
     * using copy assignment.
     */
    template <typename U>
    void operator=(const buffer<U> &other);
    void operator=(const buffer<T> &other);

    /// swap the contents of the two buffers
    void swap(buffer<T> &other);

    /** This is used to change the location of the buffer contents in place.
     * For GPU based allocators, the new allocation is made on the device
     * active at the time the call is made. If the new allocator and owner are
     * the same as the current allocator and owner, then the call is a NOOP.
     * Otherwise the data is reallocated and moved.
     *
     * @param[in] alloc the new allocator
     * @returns zero if the operation was successful
     */
    int move(allocator alloc);

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
    /// assign the range from the passed array (src is always on the host)
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
     * extending the buffer as needed. (src is always on the host)
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
     * elements starting at src_start (src is always on the host)*/
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
    int set(size_t dest_start, const buffer<U> &src, size_t src_start, size_t n_vals);
    ///@}


    /** @name get
     * gets a range of values from the buffer
     */
    ///@{
    /** gets n_vals elements starting at src_start into the passed array
     * elements starting at dest_start (dest is always on the host)*/
    template <typename U>
    int get(size_t src_start, U *dest, size_t dest_start, size_t n_vals) const;

    /** gets n_vals elements starting at src_start into the passed buffer's
     * elements starting at dest_start */
    template <typename U>
    int get(size_t src_start, buffer<U> &dest, size_t dest_start, size_t n_vals) const;

    /** gets n_vals elements starting at src_start into the passed buffer's
     * elements starting at dest_start */
    template <typename U>
    int get(buffer<U> &dest) const
    {
        return this->get(0, dest, 0, this->size());
    }
    ///@}

#if !defined(SWIG)
    /** @returns a read only pointer to the contents of the buffer accessible on
     * the host.  If the buffer is currently accessible by codes running on the
     * host then this call is a NOOP.  If the buffer is not currently accessible
     * by codes running on the host then a temporary buffer is allocated and the
     * data is moved to the host.  The returned shared_ptr deals with
     * deallocation of the temporary if needed.
     */
    std::shared_ptr<const T> get_host_accessible() const;
#endif

    /// returns true if the data is accessible from codes running on the host
    int host_accessible() const;

#if !defined(SWIG)
    /** @returns a read only pointer to the contents of the buffer accessible
     * from the active CUDA device.  If the buffer is currently accessible on
     * the active CUDA device then this call is a NOOP.  If the buffer is not
     * currently accessible on the active CUDA device then a temporary buffer
     * is allocated and the data is moved.  The returned shared_ptr deals with
     * deallocation of the temporary if needed.
     */
    std::shared_ptr<const T> get_cuda_accessible() const;
#endif

    /// returns true if the data is accessible from CUDA codes
    int cuda_accessible() const;

#if !defined(SWIG)
    /** @returns a read only pointer to the contents of the buffer accessible
     * from the active HIP device.  If the buffer is currently accessible on
     * the active HIP device then this call is a NOOP.  If the buffer is not
     * currently accessible on the active HIP device then a temporary buffer is
     * allocated and the data is moved.  The returned shared_ptr deals with
     * deallocation of the temporary if needed.
     */
    std::shared_ptr<const T> get_hip_accessible() const;
#endif

    /// returns true if the data is accessible from HIP codes
    int hip_accessible() const;

#if !defined(SWIG)
    /** @name get_openmp_accessible
     * @returns a read only pointer to the contents of the buffer accessible
     * from the active OpenMP off load device.  If the buffer is currently
     * accessible on the active OpenMP off load device then this call is a
     * NOOP.  If the buffer is not currently accessible on the active OpenMP
     * off load device then a temporary buffer is allocated and the data is
     * moved.  The returned shared_ptr deals with deallocation of the temporary
     * if needed.
     */
    ///@{
    /** returns a pointer to the contents of the buffer accessible from within
     * OpenMP off load
     */
    std::shared_ptr<const T> get_openmp_accessible() const;
    ///@}
#endif

    /// returns true if the data is accessible from OpenMP off load codes
    int openmp_accessible() const;

#if !defined(SWIG)
    /** @returns a read only pointer to the contents of the buffer accessible
     * from the active device using the technology most suitable witht he
     * current build configuration. If the buffer is currently accessible on
     * the active device then this call is a NOOP.  If the buffer is not
     * currently accessible on the active device then a temporary buffer is
     * allocated and the data is moved.  The returned shared_ptr deals with
     * deallocation of the temporary if needed.
     */
    std::shared_ptr<const T> get_device_accessible() const;
#endif

    /** returns true if the data is accessible from device codes using the
    * technology most suitable with the current build configuration.
    */
    int device_accessible() const;

    /** @name data
     * @returns a writable pointer to the buffer contents. Use this to modify
     * the buffer contents or when you know that the buffer contents are
     * accessible by the code operating on them to save the cost of a
     * std::shared_ptr copy construct.
     */
    ///@{
    /// return a pointer to the buffer contents
    T *data() { return m_data.get(); }

    /// return a const pointer to the buffer contents
    const T *data() const { return m_data.get(); }
    ///@}

    /** @name pointer
     * @returns the smart pointer managing the buffer contents. Use this when you
     * know that the buffer contents are accessible by the code operating on
     * them to save the costs of the logic that determines if a temporary is
     * needed
     */
    ///@{
    /// @returns a pointer to the buffer contents
    std::shared_ptr<T> &pointer() { return m_data; }

    /// @returns a const pointer to the buffer contents
    const std::shared_ptr<T> &pointer() const { return m_data; }
    ///@}

    /// @returns the allocator type enum
    allocator get_allocator() const { return m_alloc; }

    /// @returns the device id where the memory was allocated
    int get_owner() const { return m_owner; }

    /// @returns the active stream
    const hamr::stream &get_stream() const { return m_stream; }
    hamr::stream &get_stream() { return m_stream; }

    /** Sets the active stream and data transfer synchrnonization mode. See
     * buffer_transfer.
     *
     * @param[in] strm a ::stream object used to order operations
     * @param[in] sync a ::buffer_transfer specifies synchronous or
     *                 asynchronous behavior.
     */
    void set_stream(const stream &strm, transfer sync = transfer::async)
    {
        m_stream = strm;
        m_sync = sync;
    }

    /** Set the transfer mode to asynchronous. One must manually synchronize
     * before data access when needed. See ::synchronize
     */
    void set_transfer_asynchronous() { m_sync = transfer::async; }

    /** Set the transfer mode to synchronize automatically after data movement
     * from the GPU to the host.
     */
    void set_transfer_sycnhronous_host() { m_sync = transfer::sync_host; }

    /** Set the transfer mode to synchronize every data transfer. This mode
     * should not be used except for debugging.
     */
    void set_transfer_sycnhronous() { m_sync = transfer::sync; }

    /// @returns the current ::buffer_transfer mode
    transfer get_transfer_mode() const { return m_sync; }

    /** synchronizes with the current stream. This ensures that asynchronous
     * data transfers have completed before you access the data.
     */
    int synchronize() const { return m_stream.synchronize(); }

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
     * host. The allocator is used to determine which. @returns 0 if successful.
     */
    int set_owner();

    /** set the device where the buffer is located by querying the driver API or the
     * host. The allocator is used to determine which. @returns 0 if successful.
     */
    int set_owner(const T *ptr);

    /// get the active device id associated with the current allocator
    int get_active_device(int &dev_id);

private:
    allocator m_alloc;
    std::shared_ptr<T> m_data;
    size_t m_size;
    size_t m_capacity;
    int m_owner;
    hamr::stream m_stream;
    transfer m_sync;

    template<typename U> friend class buffer;
};

}

#if !defined(HAMR_SEPARATE_IMPL)
#include "hamr_buffer_impl.h"
#endif

#endif
