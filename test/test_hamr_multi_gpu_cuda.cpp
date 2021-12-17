#include "hamr_buffer.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>

template <typename T>
void print(const hamr::buffer<T> &buf)
{
    auto spbuf = buf.get_cpu_accessible();
    const T *pbuf = spbuf.get();

    std::cerr << pbuf[0];
    for (int i = 1; i < buf.size(); ++i)
        std::cerr << ", "<< pbuf[i];
    std::cerr << std::endl;
}


int main(int argc, char **argv)
{
    // get the number of GPUs
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess)
    {
        std::cerr << "ERROR: failed to get number of devices" << std::endl;
        return -1;
    }

    if (n_dev < 2)
    {
        std::cerr << "Can't run the test with " << n_dev << " CUDA devices" << std::endl;
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
        if (cudaSetDevice(i) != cudaSuccess)
        {
            std::cerr << "ERROR: failed to set the active device to " << i << std::endl;
            return -1;
        }

        hamr::buffer<T> *dest = new hamr::buffer<T>(hamr::buffer_allocator::cuda, *src);

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

    auto spsrc = src->get_cpu_accessible();
    T *psrc = spsrc.get();

    for (int i = 0; i < n_elem; ++i)
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
