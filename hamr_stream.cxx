#include "hamr_stream.h"

#include <iostream>

namespace hamr
{

// --------------------------------------------------------------------------
int stream::synchronize() const
{
#if defined(HAMR_ENABLE_CUDA)
    if (const cudaStream_t *cs = std::get_if<1>(&m_stream))
    {
        cudaStreamSynchronize(*cs);
    }
#endif
#if defined(HAMR_ENABLE_HIP)
    if (const hipStream_t *hs = std::get_if<2>(&m_stream))
    {
        hipStreamSynchronize(*hs);
    }
#endif
    return 0;
}

// --------------------------------------------------------------------------
stream::operator bool() const
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

// --------------------------------------------------------------------------
size_t stream::get_stream()
{
#if defined(HAMR_ENABLE_CUDA)
    if (const cudaStream_t *cs = std::get_if<1>(&m_stream))
    {
        return (size_t)*cs;
    }
#endif
#if defined(HAMR_ENABLE_HIP)
    if (const hipStream_t *hs = std::get_if<2>(&m_stream))
    {
        return (size_t)*hs;
    }
#endif
    return 2;
}

// --------------------------------------------------------------------------
void stream::print() const
{
#if defined(HAMR_ENABLE_CUDA)
    if (const cudaStream_t *cs = std::get_if<1>(&m_stream))
    {
        std::cerr << "cudaStream_t m_stream = " << *cs << std::endl;
        return;
    }
#endif
#if defined(HAMR_ENABLE_HIP)
    if (const hipStream_t *hs = std::get_if<2>(&m_stream))
    {
        std::cerr << "hipStream_t m_stream = " << *hs << std::endl;
        return;
    }
#endif
    std::cerr << "empty" << std::endl;
}

}
