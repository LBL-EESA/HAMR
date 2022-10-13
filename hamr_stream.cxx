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
void stream::print() const
{
    if (const cudaStream_t *cs = std::get_if<1>(&m_stream))
    {
        std::cerr << "cudaStream_t m_stream = " << *cs << std::endl;
    }
    else
    if (const hipStream_t *hs = std::get_if<2>(&m_stream))
    {
        std::cerr << "hipStream_t m_stream = " << *hs << std::endl;
    }
    else
    {
        std::cerr << "empty" << std::endl;
    }
}

}
