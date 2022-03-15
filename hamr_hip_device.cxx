#include "hamr_hip_device.h"

#include <iostream>

#include <hip/hip_runtime.h>


namespace hamr
{

// **************************************************************************
int get_active_hip_device(int &dev_id)
{
    hipError_t ierr = hipSuccess;
    if ((ierr = hipGetDevice(&dev_id)) != hipSuccess)
    {
        std::cerr << "Failed to get the active HIP device. "
            << hipGetErrorString(ierr) << std::endl;
        return -1;
    }

    return 0;
}

// **************************************************************************
int set_active_hip_device(int dev_id)
{
    hipError_t ierr = hipSuccess;
    if ((ierr = hipSetDevice(dev_id)) != hipSuccess)
    {
        std::cerr << "Failed to set the active HIP device. "
            << hipGetErrorString(ierr) << std::endl;
        return -1;
    }

    return 0;
}


// --------------------------------------------------------------------------
activate_hip_device::activate_hip_device(int new_dev) : m_device(-1)
{
    int cur_dev = -1;
    if (!get_active_hip_device(cur_dev) && (cur_dev != new_dev) &&
        !set_active_hip_device(new_dev))
    {
        m_device = cur_dev;
    }
}

// --------------------------------------------------------------------------
activate_hip_device::~activate_hip_device()
{
    if (m_device >= 0)
    {
        set_active_hip_device(m_device);
    }
}

}
