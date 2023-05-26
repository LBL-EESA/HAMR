#ifndef hamr_hip_launch_h
#define hamr_hip_launch_h

/// @file

#include "hamr_config.h"

#include <deque>

#include <hip/hip_runtime.h>


/// heterogeneous accelerator memory resource
namespace hamr
{

/** A flat array is broken into blocks of number of threads where each adjacent
 * thread accesses adjacent memory locations. To accomplish this we might need
 * a large number of blocks. If the number of blocks exceeds the max block
 * dimension in the first and or second block grid dimension then we need to
 * use a 2d or 3d block grid.
 *
 * ::partition_thread_blocks - decides on a partitioning of the data based on
 * warps_per_block parameter. The resulting decomposition will be either 1,2,
 * or 3D as needed to accommodate the number of fixed sized blocks. It can
 * happen that max grid dimensions are hit, in which case you'll need to
 * increase the number of warps per block.
 *
 * ::thread_id_to_array_index - given a thread and block id gets the
 * array index to update. _this may be out of bounds so be sure
 * to validate before using it.
 *
 * ::index_is_valid - test an index for validity.
*/
/// @name CUDA indexing scheme
///@{

/// query properties for the named CUDA device. retruns non-zero on error
HAMR_EXPORT
int get_launch_props(int device_id,
    int *block_grid_max, int &warp_size,
    int &max_warps_per_block);


/** convert a CUDA index into a flat array index using the partitioning scheme
 * defined in partition_thread_blocks
 */
inline
__device__
unsigned long thread_id_to_array_index()
{
    return threadIdx.x + blockDim.x*(blockIdx.x + blockIdx.y * gridDim.x
        + blockIdx.z * gridDim.x * gridDim.y);
}

/// bounds check the flat index
inline
__device__
int index_is_valid(unsigned long index, unsigned long max_index)
{
    return index < max_index;
}

/** Calculate CUDA launch parameters for an arbitrarily large flat array.
 *
 * @param[in]  device_id the CUDA device to use. Default values for
 *                       warps_per_block and block_grid_max are determined by
 *                       querying the capabilities of the device. If -1 is
 *                       passed then the currently active device is used.
 * @param[in]  array_size the length of the array being processed
 * @param[in]  warps_per_block number of warps to use per block (your choice).
 *                             Using a larger number here will result in fewer
 *                             blocks being processed concurrently.
 *
 * @param[out] block_grid block dimension kernel launch control
 * @param[out] n_blocks number of blocks
 * @param[out] thread_grid thread dimension kernel launch control
 *
 * @returns zero if successful and non-zero if an error occurred
 */
HAMR_EXPORT
int partition_thread_blocks(int device_id, size_t array_size,
    int warps_per_block, dim3 &block_grid, int &n_blocks,
    dim3 &thread_grid);

/** Calculate CUDA launch parameters for an arbitrarily large flat array. See
 * ::get_launch_props for determining the correct values for warp_size and
 * block_grid_max.
 *
 * @param[in]  array_size      The length of the array being processed
 * @param[in]  warp_size       The number of threads per warp supported on the device
 * @param[in]  warps_per_block The number of warps to use per block (your choice)
 * @param[in]  block_grid_max  The maximum number of blocks, in 3-dimensions,
 *                             supported by the device
 * @param[out] block_grid      The block grid dimension kernel launch control parameter
 * @param[out] n_blocks        The total number of blocks that will be launched
 * @param[out] thread_grid     The thread grid dimension kernel launch control parameter
 *
 * @returns zero if successful and non-zero if an error occurred
 */
HAMR_EXPORT
int partition_thread_blocks(size_t array_size,
    int warps_per_block, int warp_size, int *block_grid_max,
    dim3 &block_grid, int &n_blocks, dim3 &thread_grid);
}

///@}
#endif
