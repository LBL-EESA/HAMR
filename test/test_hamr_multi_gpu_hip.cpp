#include "hamr_buffer.h"
#include "hamr_buffer_util.h"

#include <hip/hip_runtime.h>


#include <iostream>

template <typename T>
void print(const hamr::buffer<T> &buf)
{
    auto [spbuf, pbuf] = hamr::get_cpu_accessible(buf);

    std::cerr << pbuf[0];
    for (size_t i = 1; i < buf.size(); ++i)
        std::cerr << ", "<< pbuf[i];
    std::cerr << std::endl;
}


int main(int argc, char **argv)
{
    (void) argc;
    (void) argv;

    // get the number of GPUs
    int n_dev = 0;
    if (hipGetDeviceCount(&n_dev) != hipSuccess)
    {
        std::cerr << "ERROR: failed to get number of devices" << std::endl;
        return -1;
    }

    if (n_dev < 2)
    {
        std::cerr << "Can't run the test with " << n_dev << " HIP devices" << std::endl;
        return 0;
    }

    // allocate some data on the CPU
    size_t n_elem = 1000;

    using T = int;
    T val = 31415;

    hamr::buffer<T> *src = new hamr::buffer<T>(hamr::buffer_allocator::malloc, n_elem, val);

    if (n_elem < 33)
        print(*src);

    // move to each GPU
    for (int i = 0; i < n_dev; ++i)
    {
        std::cerr << " ==== move to device " << i << " ==== " << std::endl;

        // move to GPU i
        if (hipSetDevice(i) != hipSuccess)
        {
            std::cerr << "ERROR: failed to set the active device to " << i << std::endl;
            return -1;
        }

        hamr::buffer<T> *dest = new hamr::buffer<T>(hamr::buffer_allocator::hip, *src);

        if (n_elem < 33)
            print(*dest);

        // update the source
        delete src;
        src = dest;
    }

    // move back to the CPU
    std::cerr << " ==== move to CPU ==== " << std::endl;

    hamr::buffer<T> end(hamr::buffer_allocator::malloc, *src);

    if (n_elem < 33)
       print(end);

    // check for 31415
    std::cerr << " ==== validate ==== " << std::endl;

    auto [spsrc, psrc] = hamr::get_cpu_accessible(*src);

    for (size_t i = 0; i < n_elem; ++i)
    {
        if (psrc[i] != val)
        {
            std::cerr << "ERROR: psrc[ " << i << "] == " << psrc[i]
                << " != " << val << std::endl;
            return -1;
        }
    }

    std::cerr << "All values verified to be " << val << std::endl;

    return 0;
}
