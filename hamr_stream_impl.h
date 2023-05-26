#ifndef hamr_stream_h
#define hamr_stream_h

///@file

#include "hamr_config.h"

#include <cstddef>
#include <variant>

#if defined(HAMR_ENABLE_CUDA)
#include <cuda_runtime.h>
#else
using cudaStream_t = void*;
#endif
#if defined(HAMR_ENABLE_HIP)
#include <hip/hip_runtime.h>
#else
using hipStream_t = void*;
#endif

namespace hamr
{

/// A wrapper around technology specific streams.
/** Streams are used to enable and order concurrent operations on accelerator
 * devices. The default stream used in hamr is a stream-per-thread where
 * available.  However, note that libraries built seperately will likely use
 * the default blocking stream and if so explicit specification of the stream
 * when calling into those libraries is necessary. Note that hamr passes stream
 * correctly when interfacing with Python. In most cases the hamr API's
 * requiring a ::stream can be passed the technology specific stream due to
 * implicit conversion operators implemented here.
 */
class HAMR_EXPORT stream
{
public:
    /// constructs a default stream
    stream() :
#if defined(HAMR_ENABLE_CUDA)
        m_stream(std::in_place_index<1>, cudaStreamPerThread)
#elif defined(HAMR_ENABLE_HIP)
        m_stream(std::in_place_index<2>, hipStreamPerThread)
#else
        m_stream(std::in_place_index<0>, '\0')
#endif
   {}

    stream(const stream &) = default;
    stream(stream &&) = default;

    stream &operator=(const stream &) = default;
    stream &operator=(stream &&) = default;

#if defined(HAMR_ENABLE_CUDA)
    /// convert to a CUDA stream
    operator cudaStream_t () const { return this->get_cuda_stream(); }

    /// assign a CUDA stream
    stream &operator=(cudaStream_t strm)
    {
        m_stream = strm;
        return *this;
    }

    /// Constructs or converts from a CUDA stream
    stream(const cudaStream_t &strm) : m_stream(std::in_place_index<1>, strm) {}

    /// Accesses the CUDA stream.
    cudaStream_t get_cuda_stream() const
    {
        const cudaStream_t *cs;
        if ((cs = std::get_if<1>(&m_stream)))
            return *cs;
        return 0; // default stream
    }
#endif
#if defined(HAMR_ENABLE_HIP)
    /// convert to a HIP stream
    operator hipStream_t () const { return this->get_hip_stream(); }

    /// assign a HIP stream
    stream &operator=(hipStream_t strm)
    {
        m_stream = strm;
        return *this;
    }

    /// Constructs or converts from a HIP stream
    stream(hipStream_t &strm) : m_stream(std::in_place_index<2>, strm) {}

    /// Accesses the HIP stream.
    hipStream_t get_hip_stream() const
    {
        const hipStream_t *hs;
        if ((hs = std::get_if<2>(&m_stream)))
            return *hs;
        return 0; // default stream
    }
#endif

    /// synchronize the stream
    int synchronize() const;

    /// evaluates true if a stream has been set
    operator bool() const;

    /// sends the value of the stream to std::cerr
    void print() const;

    /// convert the technology specific stream to an integer
    size_t get_stream();

private:
    std::variant<char, cudaStream_t, hipStream_t> m_stream;
};
}
#endif
