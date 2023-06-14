#include "hamr_cuda_device.h"

#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

namespace hamr
{

// **************************************************************************
int get_active_cuda_device(int &dev_id)
{
    cudaError_t ierr = cudaSuccess;
    if ((ierr = cudaGetDevice(&dev_id)) != cudaSuccess)
    {
        std::cerr << "Failed to get the active CUDA device. "
            << cudaGetErrorString(ierr) << std::endl;
        return -1;
    }

    return 0;
}

// **************************************************************************
int set_active_cuda_device(int dev_id)
{
    cudaError_t ierr = cudaSuccess;
    if ((ierr = cudaSetDevice(dev_id)) != cudaSuccess)
    {
        std::cerr << "Failed to set the active CUDA device. "
            << cudaGetErrorString(ierr) << std::endl;
        return -1;
    }

    return 0;
}

// **************************************************************************
int get_cuda_device(const void *ptr, int &device_id)
{
    cudaError_t ierr = cudaSuccess;
    cudaPointerAttributes ptrAtts;
    ierr = cudaPointerGetAttributes(&ptrAtts, ptr);

    // these types of pointers are NOT accessible on the GPU
    // cudaErrorInValue occurs when the pointer is unknown to CUDA, as is
    // the case with pointers allocated by malloc or new.
    if ((ierr == cudaErrorInvalidValue) ||
        ((ierr == cudaSuccess) && ((ptrAtts.type == cudaMemoryTypeHost) ||
        (ptrAtts.type == cudaMemoryTypeUnregistered))))
    {
        // this is host backed memory not associate with a GPU
        device_id = -1;
    }
    else if (ierr != cudaSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to get pointer attributes for " << ptr << std::endl;
        return -1;
    }
    else
    {
        device_id = ptrAtts.device;
    }

    return 0;
}

// --------------------------------------------------------------------------
activate_cuda_device::activate_cuda_device(int new_dev) : m_device(-1)
{
    int cur_dev = -1;
    if (!get_active_cuda_device(cur_dev) && (cur_dev != new_dev) &&
        !set_active_cuda_device(new_dev))
    {
        m_device = cur_dev;
    }
}

// --------------------------------------------------------------------------
activate_cuda_device::~activate_cuda_device()
{
    if (m_device >= 0)
    {
        set_active_cuda_device(m_device);
    }
}

}
