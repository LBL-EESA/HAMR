#ifndef hamr_stream_h
#define hamr_stream_h

///@file

#include "hamr_config.h"

#include <variant>
#if defined(HAMR_ENABLE_CUDA)
#include <cuda_runtime.h>
#else
using cudaStream_t = char;
#endif
#if defined(HAMR_ENABLE_HIP)
#include <hip/hip_runtime.h>
#else
using hipStream_t = char;
#endif


namespace hamr
{

/// A wrapper around technology specific streams.
/** Streams are used to enable concurrent operations on the GPU. The default is
 * for a stream per thread.  However, note that libraries built seperately will
 * likely use the default blocking stream and if so explicit use of streams is
 * necessary.  In most cases API's requiring a hamr::stream can be passed the
 * technology specific stream due to implicit conversions.
 */
class stream
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
    /// assign a CUDA stream
    stream &operator=(cudaStream_t strm)
    {
        m_stream = strm;
        return *this;
    }

    /// Constructs or converts from a CUDA stream
    stream(cudaStream_t &strm) : m_stream(std::in_place_index<1>, strm) {}

    /// Accesses the CUDA stream.
    cudaStream_t cuda_stream() const
    {
        const cudaStream_t *cs;
        if ((cs = std::get_if<1>(&m_stream)))
            return *cs;
        return 0; // default stream
    }
#endif
#if defined(HAMR_ENABLE_HIP)
    /// assign a HIP stream
    stream &operator=(hipStream_t strm)
    {
        m_stream = strm;
        return *this;
    }

    /// Constructs or converts from a HIP stream
    stream(hipStream_t &strm) : m_stream(std::in_place_index<2>, strm) {}

    /// Accesses the HIP stream.
    hipStream_t hip_stream() const
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
    operator bool()
    {
        if (std::get_if<1>(&m_stream))
        {
            return true;
        }
        else if (std::get_if<2>(&m_stream))
        {
            return true;
        }
        return false;
    }

    /// sends the value of the stream to std::cerr
    void print() const;

private:
    std::variant<char, cudaStream_t, hipStream_t> m_stream;
};
}
#endif
