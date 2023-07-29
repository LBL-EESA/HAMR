#ifndef hamr_cuda_device_h
#define hamr_cuda_device_h

#include "hamr_config.h"

///@file

namespace hamr
{
/// gets the device identifier for the first GPU. @returns zero if successful.
inline int HAMR_EXPORT get_cuda_device_identifier(int &dev_id) { dev_id = 0; return 0; }

/// gets the device identifier for the host. @returns zero if successful.
inline int HAMR_EXPORT get_cuda_host_identifier(int &dev_id) { dev_id = -1; return 0; }

/// gets the currently atcive CUDA device. @returns zero if successful.
int HAMR_EXPORT get_active_cuda_device(int &dev_id);

/// sets the active CUDA device. returns zero if successful.
int HAMR_EXPORT set_active_cuda_device(int dev_id);

/// gets the device that owns the given pointer. @returns zero if successful.
int HAMR_EXPORT get_cuda_device(const void *ptr, int &device_id);

/** Activate the specified CUDA device, and restore the previously active
 * device when the object is destroyed.
 */
class HAMR_EXPORT activate_cuda_device
{
public:
    activate_cuda_device() = delete;
    activate_cuda_device(const activate_cuda_device &) = delete;
    void operator=(const activate_cuda_device &) = delete;

    activate_cuda_device(int id);
    ~activate_cuda_device();

private:
    int m_device;
};


/** Activate peer to peer memory access between two devices, and deactivate when
 * the object goes out of scope.
 */
class access_cuda_peer
{
public:
    access_cuda_peer() : m_dest_device(-1), m_src_device(-1), m_symetric(false) {}
    ~access_cuda_peer() { disable(); }

    /// enable peer to peer access. the dest device must active.
    int enable(int dest_device, int src_device, bool symetric);

    /// disable peer to peer access.
    int disable();

private:
    int m_dest_device;
    int m_src_device;
    int m_symetric;
};


}
#endif
