#ifndef hamr_cuda_device_h
#define hame_cuda_device_h

///@file

namespace hamr
{

/// gets the currently atcive CUDA device. returns zero if successful.
int get_active_cuda_device(int &dev_id);

/// sets the active CUDA device. returns zero if successful.
int set_active_cuda_device(int dev_id);


/** Activate the specified CUDA device, and restore the previously active
 * device when the object is destroyed.
 */
class activate_cuda_device
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

}




#endif
