#ifndef hamr_openmp_device_h
#define hamr_openmp_device_h

#include "hamr_config.h"

///@file

namespace hamr
{

/// gets the currently atcive HIP device. returns zero if successful.
int HAMR_EXPORT get_active_openmp_device(int &dev_id);

/// sets the active HIP device. returns zero if successful.
int HAMR_EXPORT set_active_openmp_device(int dev_id);


/** Activate the specified HIP device, and restore the previously active
 * device when the object is destroyed.
 */
class HAMR_EXPORT activate_openmp_device
{
public:
    activate_openmp_device() = delete;
    activate_openmp_device(const activate_openmp_device &) = delete;
    void operator=(const activate_openmp_device &) = delete;

    activate_openmp_device(int id);
    ~activate_openmp_device();

private:
    int m_device;
};

}




#endif
