#ifndef hamr_hip_device_h
#define hamr_hip_device_h

#include "hamr_config.h"

///@file

namespace hamr
{
/// gets the device identifier for the first GPU. @returns zero if successful.
inline int HAMR_EXPORT get_hip_device_identifier(int &dev_id) { dev_id = 0; return 0; }

/// gets the device identifier for the host. @returns zero if successful.
inline int HAMR_EXPORT get_hip_host_identifier(int &dev_id) { dev_id = -1; return 0; }

/// gets the currently atcive HIP device. returns zero if successful.
int HAMR_EXPORT get_active_hip_device(int &dev_id);

/// sets the active HIP device. returns zero if successful.
int HAMR_EXPORT set_active_hip_device(int dev_id);

/// gets the device that owns the given pointer. @returns zero if successful.
int HAMR_EXPORT get_hip_device(const void *ptr, int &device_id);


/** Activate the specified HIP device, and restore the previously active
 * device when the object is destroyed.
 */
class HAMR_EXPORT activate_hip_device
{
public:
    activate_hip_device() = delete;
    activate_hip_device(const activate_hip_device &) = delete;
    void operator=(const activate_hip_device &) = delete;

    activate_hip_device(int id);
    ~activate_hip_device();

private:
    int m_device;
};

}




#endif
