#include "hamr_openmp_device.h"

#include <iostream>
#include <omp.h>

namespace hamr
{
// **************************************************************************
int get_openmp_host_identifier(int &dev_id)
{
    dev_id = omp_get_initial_device();
    return 0;
}

// **************************************************************************
int get_active_openmp_device(int &dev_id)
{
    dev_id = omp_get_default_device();
    return 0;
}

// **************************************************************************
int set_active_openmp_device(int dev_id)
{
    omp_set_default_device(dev_id);
    return 0;
}

// **************************************************************************
int HAMR_EXPORT get_openmp_device(const void *ptr, int &device_id)
{
    (void)ptr;
    device_id = 0;
    return -1;
}

// --------------------------------------------------------------------------
activate_openmp_device::activate_openmp_device(int new_dev) : m_device(-1)
{
    int cur_dev = -1;
    if (!get_active_openmp_device(cur_dev) && (cur_dev != new_dev) &&
        !set_active_openmp_device(new_dev))
    {
        m_device = cur_dev;
    }
}

// --------------------------------------------------------------------------
activate_openmp_device::~activate_openmp_device()
{
    if (m_device >= 0)
    {
        set_active_openmp_device(m_device);
    }
}

}
