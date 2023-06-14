#ifndef buffer_transfer_h
#define buffer_transfer_h

///@file

/// heterogeneous accelerator memory resource
namespace hamr
{

/** flag used to indicate whether or not a transfer operation should be
 * synchronous or not.
 */
enum class buffer_transfer
{
    async = 0,   ///< all operations are asynchronous
    sync_host = 1,///< operations moving data from GPU to host memory are synchronous
    sync = 2     ///< all operations are synchronous
};

}

#endif
