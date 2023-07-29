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

// **************************************************************************
int access_cuda_peer::enable(int dest_device, int src_device, bool symetric)
{
    int access = 0;
    cudaError_t ierr = cudaSuccess;

    // ensure that the current device is restored if we return due to an error
    // hamr::cuda_device activate_dest(dest_device);

    // enable dest to access the source memory
    if ((ierr = cudaDeviceCanAccessPeer(&access, dest_device, src_device)) != cudaSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to determine peer accessibility between "
            << dest_device << " and " << src_device << ". "
            << cudaGetErrorString(ierr) << std::endl;
        return -1;
    }

    if (!access)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Can't access device " << src_device
            << " from " << dest_device << std::endl;
        return -1;
    }

    if ((ierr = cudaDeviceEnablePeerAccess(src_device, 0)) != cudaSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to enable peer accessibility between "
            << dest_device << " and " << src_device << ". "
            << cudaGetErrorString(ierr) << std::endl;
        return -1;
    }

    // record that p2p was enabled
    m_src_device = src_device;
    m_dest_device = dest_device;
    m_symetric = false;

    // enable the source to access the dest memory
    if (symetric)
    {
        if ((ierr = cudaDeviceCanAccessPeer(&access, src_device, dest_device)) != cudaSuccess)
        {
            std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
                " Failed to determine peer accessibility between "
                << dest_device << " and " << src_device << ". "
                << cudaGetErrorString(ierr) << std::endl;
            return -1;
        }

        if (!access)
        {
            std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
                " Can't access device " << dest_device
                << " from " << src_device << std::endl;
            return -1;
        }

        // ensure that the current device is restored if we return due to an error
        hamr::activate_cuda_device activate_src(src_device);

        if ((ierr = cudaDeviceEnablePeerAccess(dest_device, 0)) != cudaSuccess)
        {
            std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
                " Failed to enable peer accessibility between "
                << src_device << " and " << dest_device << ". "
                << cudaGetErrorString(ierr) << std::endl;
            return -1;
        }

        // record taht p2p was enabled
        m_symetric = true;
    }
    return 0;
}

// **************************************************************************
int access_cuda_peer::disable()
{
    if (m_src_device >= 0)
    {
        // ensure that the current device is restored if we return due to an error
        hamr::activate_cuda_device activate_dest(m_dest_device);

        // disable peer to peer memory map between dest and src
        cudaError_t ierr = cudaSuccess;
        if ((ierr = cudaDeviceDisablePeerAccess(m_src_device)) != cudaSuccess)
        {
            std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
                " Failed to disable peer accessibility between "
                << m_dest_device << " and " << m_src_device << ". "
                << cudaGetErrorString(ierr) << std::endl;
            return -1;
        }

        if (m_symetric)
        {
            // ensure that the current device is restored if we return due to an error
            hamr::activate_cuda_device activate_src(m_src_device);

            // disable peer to peer memory map between dest and src
            if ((ierr = cudaDeviceDisablePeerAccess(m_dest_device)) != cudaSuccess)
            {
                std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
                    " Failed to disable peer accessibility between "
                    << m_src_device << " and " << m_dest_device << ". "
                    << cudaGetErrorString(ierr) << std::endl;
                return -1;
            }
        }

        // record that p2p was disabled
        m_src_device = -1;
    }
    return 0;
}

}
