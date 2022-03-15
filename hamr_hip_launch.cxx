
#include <hip/hip_runtime.h>
#include "hamr_hip_launch.h"

#include "hamr_env.h"

#include <iostream>

namespace hamr
{
// **************************************************************************
int synchronize()
{
    hipError_t ierr = hipSuccess;
    if ((ierr = hipDeviceSynchronize()) != hipSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to synchronize HIP execution. "
            << hipGetErrorString(ierr) << std::endl;
        return -1;
    }
    return 0;
}

// **************************************************************************
int get_launch_props(int device_id,
    int *block_grid_max, int &warp_size,
    int &warps_per_block_max)
{
    hipError_t ierr = hipSuccess;

    if (device_id < 0)
    {
        if ((ierr = hipGetDevice(&device_id)) != hipSuccess)
        {
            std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
                " Failed to get the active device id. "
                << hipGetErrorString(ierr) << std::endl;
            return -1;
        }
    }

    if (((ierr = hipDeviceGetAttribute(&block_grid_max[0], hipDeviceAttributeMaxGridDimX, device_id)) != hipSuccess)
        || ((ierr = hipDeviceGetAttribute(&block_grid_max[1], hipDeviceAttributeMaxGridDimY, device_id)) != hipSuccess)
        || ((ierr = hipDeviceGetAttribute(&block_grid_max[2], hipDeviceAttributeMaxGridDimZ, device_id)) != hipSuccess))
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to get HIP max grid dim. " << hipGetErrorString(ierr) << std::endl;
        return -1;
    }

    if ((ierr = hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, device_id)) != hipSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to get HIP warp size. " << hipGetErrorString(ierr) << std::endl;
        return -1;
    }

    int threads_per_block_max = 0;

    if ((ierr = hipDeviceGetAttribute(&threads_per_block_max,
        hipDeviceAttributeMaxThreadsPerBlock, device_id)) != hipSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to get HIP max threads per block. " << hipGetErrorString(ierr) << std::endl;
        return -1;
    }

    warps_per_block_max = threads_per_block_max / warp_size;

    return 0;
}

// **************************************************************************
int partition_thread_blocks(size_t array_size,
    int warps_per_block, int warp_size, int *block_grid_max,
    dim3 &block_grid, int &n_blocks, dim3 &thread_grid)
{
    unsigned long threads_per_block = warps_per_block * warp_size;

    thread_grid.x = threads_per_block;
    thread_grid.y = 1;
    thread_grid.z = 1;

    unsigned long block_size = threads_per_block;
    n_blocks = array_size / block_size;

    if (array_size % block_size)
        ++n_blocks;

    if (n_blocks > block_grid_max[0])
    {
        // multi-d decomp required
        block_grid.x = block_grid_max[0];
        block_grid.y = n_blocks / block_grid_max[0];
        if (n_blocks % block_grid_max[0])
        {
            ++block_grid.y;
        }

        if (block_grid.y > ((unsigned int)block_grid_max[1]))
        {
            // 3d decomp
            unsigned long block_grid_max01 = block_grid_max[0] * block_grid_max[1];
            block_grid.y = block_grid_max[1];
            block_grid.z = n_blocks / block_grid_max01;

            if (n_blocks % block_grid_max01)
                ++block_grid.z;

            if (block_grid.z > ((unsigned int)block_grid_max[2]))
            {
                std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
                    " Too many blocks " << n_blocks << " of size " << block_size
                    << " are required for a grid of (" << block_grid_max[0] << ", "
                    << block_grid_max[1] << ", " << block_grid_max[2]
                    << ") blocks. Hint: increase the number of warps per block." << std::endl;
                return -1;
            }
        }
        else
        {
            // 2d decomp
            block_grid.z = 1;
        }
    }
    else
    {
        // 1d decomp
        block_grid.x = n_blocks;
        block_grid.y = 1;
        block_grid.z = 1;
    }

#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "partition_thread_blocks arrays_size = " << array_size
            << " warps_per_block = " << warps_per_block << " warp_size = " << warp_size
            << " block_grid_max = (" << block_grid_max[0] << ", " << block_grid_max[1]
            << ", " << block_grid_max[2] << ") block_grid = (" << block_grid.x << ", "
            << block_grid.y << ", " << block_grid.z << ") n_blocks = " << n_blocks
            << " thread_grid = (" << thread_grid.x << ", " << thread_grid.y << ", "
            << thread_grid.z << ")" << std::endl;
    }
#endif

    return 0;
}

// **************************************************************************
int partition_thread_blocks(int device_id, size_t array_size,
    int warps_per_block, dim3 &block_grid, int &n_blocks,
    dim3 &thread_grid)
{
    int block_grid_max[3] = {0};
    int warp_size = 0;
    int warps_per_block_max = 0;
    if (get_launch_props(device_id, block_grid_max,
        warp_size, warps_per_block_max))
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to get launch properties" << std::endl;
        return -1;
    }

    return partition_thread_blocks(array_size, warps_per_block,
        warp_size, block_grid_max, block_grid, n_blocks,
        thread_grid);
}

}
