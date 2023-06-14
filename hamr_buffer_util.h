#ifndef buffer_util_h
#define buffer_util_h

/// @file

/// heterogeneous accelerator memory resource
namespace hamr
{

/// @cond
template <typename TT>
auto get_host_accessible()
{
    return std::make_tuple();
}
/// @endcond

/** Calls hamr::buffer::get_host_accessible on a number of hamr::buffer
 * instances.
 *
 * @tparam TT hamr::buffer<NT>
 * @tparam PP a paramater pack of TT
 * @param b a hamr::buffer<NT> instance
 * @param args zero or more hamr::buffer<NT> instances
 * @returns a tuple of std::shared_ptr<NT> and NT* one for each
 *          hamr::buffer<NT> passed in.
 */
template <typename TT, typename... PP>
auto get_host_accessible(const TT &b, PP &&... args)
{
    auto spb = b.get_host_accessible();
    return std::tuple_cat(std::make_tuple
        (spb, spb.get()), get_host_accessible<TT>(args...));
}

/// @cond
template <typename TT>
auto get_cuda_accessible()
{
    return std::make_tuple();
}
/// @endcond

/** Calls hamr::buffer::get_cuda_accessible on a number of hamr::buffer
 * instances.
 *
 * @tparam TT hamr::buffer<NT>
 * @tparam PP a paramater pack of TT
 * @param b a hamr::buffer<NT> instance
 * @param args zero or more hamr::buffer<NT> instances
 * @returns a tuple of std::shared_ptr<NT> and NT* one for each
 *          hamr::buffer<NT> passed in.
 */
template <typename TT, typename... PP>
auto get_cuda_accessible(const TT &b, PP &&... args)
{
    auto spb = b.get_cuda_accessible();
    return std::tuple_cat(std::make_tuple
        (spb, spb.get()), get_cuda_accessible<TT>(args...));
}

/// @cond
template <typename TT>
auto get_hip_accessible()
{
    return std::make_tuple();
}
/// @endcond

/** Calls hamr::buffer::get_hip_accessible on a number of hamr::buffer
 * instances.
 *
 * @tparam TT hamr::buffer<NT>
 * @tparam PP a paramater pack of TT
 * @param b a hamr::buffer<NT> instance
 * @param args zero or more hamr::buffer<NT> instances
 * @returns a tuple of std::shared_ptr<NT> and NT* one for each
 *          hamr::buffer<NT> passed in.
 */
template <typename TT, typename... PP>
auto get_hip_accessible(const TT &b, PP &&... args)
{
    auto spb = b.get_hip_accessible();
    return std::tuple_cat(std::make_tuple
        (spb, spb.get()), get_hip_accessible<TT>(args...));
}

/// @cond
template <typename TT>
auto get_openmp_accessible()
{
    return std::make_tuple();
}
/// @endcond

/** Calls hamr::buffer::get_openmp_accessible on a number of hamr::buffer
 * instances.
 *
 * @tparam TT hamr::buffer<NT>
 * @tparam PP a paramater pack of TT
 * @param b a hamr::buffer<NT> instance
 * @param args zero or more hamr::buffer<NT> instances
 * @returns a tuple of std::shared_ptr<NT> and NT* one for each
 *          hamr::buffer<NT> passed in.
 */
template <typename TT, typename... PP>
auto get_openmp_accessible(const TT &b, PP &&... args)
{
    auto spb = b.get_openmp_accessible();
    return std::tuple_cat(std::make_tuple
        (spb, spb.get()), get_openmp_accessible<TT>(args...));
}

/// @cond
template <typename TT>
auto get_device_accessible()
{
    return std::make_tuple();
}
/// @endcond

/** Calls hamr::buffer::get_device_accessible on a number of hamr::buffer
 * instances.
 *
 * @tparam TT hamr::buffer<NT>
 * @tparam PP a paramater pack of TT
 * @param b a hamr::buffer<NT> instance
 * @param args zero or more hamr::buffer<NT> instances
 * @returns a tuple of std::shared_ptr<NT> and NT* one for each
 *          hamr::buffer<NT> passed in.
 */
template <typename TT, typename... PP>
auto get_device_accessible(const TT &b, PP &&... args)
{
    auto spb = b.get_device_accessible();
    return std::tuple_cat(std::make_tuple
        (spb, spb.get()), get_device_accessible<TT>(args...));
}


/** Calls hamr::buffer::data on a number of hamr::buffer instances.
 *
 * @tparam PP a paramater pack of hamr::buffer<NT>
 * @param args any number of hamr::buffer<NT> instances
 * @returns a tuple of NT* one for each hamr::buffer<NT> passed in.
 */
template <typename... PP>
auto data(PP &&... args)
{
    return std::make_tuple(args.data()...);
}

/** Calls hamr::buffer::pointer on a number of hamr::buffer instances.
 *
 * @tparam PP a paramater pack of hamr::buffer<NT>
 * @param args any number of hamr::buffer<NT> instances
 * @returns a tuple of std::shared_ptr<NT> one for each hamr::buffer<NT> passed in.
 */
template <typename... PP>
auto pointer(PP &&... args)
{
    return std::make_tuple(args.pointer()...);
}

/** Calls hamr::buffer::synchronize on a number of hamr::buffer<NT> instances.
 * Note however that one typically need not call synchronize on multiple buffer
 * instances that share the same stream. Synchronizing on any one of them will
 * synchronize all.
 *
 * @tparam PP a paramater pack of hamr::buffer<NT>
 * @param args any number of hamr::buffer<NT> instances
 */
template <typename... PP>
void synchronize(PP &&... args)
{
    (args.synchronize(), ...);
}

/** constructs an un-initialized hamr::buffer<NT> with space for n_elem
 * allocated and returns it along with the writable pointer to it's contents.
 *
 * @param[in] alloc the allocator to allocate memory with
 * @param[in] n_elem the initial size of the allocated memory
 * @returns a std::tuple with the newly constructed buffer in the first slot
 *          and a writable pointer to its internal memory in the second
 */
template <typename NT>
auto make_buffer(buffer_allocator alloc, size_t n_elem)
{
    hamr::buffer<NT> buf(alloc, n_elem);
    return std::make_tuple(buf, buf.data());
}

/** constructs an hamr:buffer<NT> with space for n_elem allocated and
 * initialized and returns it along with the writable pointer to it's contents.
 *
 * @param[in] alloc the allocator to allocate memory with
 * @param[in] n_elem the initial size of the allocated memory
 * @param[in] ival the value used to initialize the allocated memory
 * @returns a std::tuple with the newly constructed buffer in the first slot
 *          and a writable pointer to its internal memory in the second
 */
template <typename NT>
auto make_buffer(buffer_allocator alloc, size_t n_elem, const NT &ival)
{
    hamr::buffer<NT> buf(alloc, n_elem, ival);
    return std::make_tuple(buf, buf.data());
}

}
#endif
