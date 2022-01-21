#ifndef buffer_pointer_h
#define buffer_pointer_h

#include <memory>

/// heterogeneous accelerator memory resource
namespace hamr
{

template <typename T> class buffer;

///  a shared pointer to an instance of a buffer<T>
template <typename T>
using p_buffer = std::shared_ptr<buffer<T>>;

///  a shared pointer to an instance of a const buffer<T>
template <typename T>
using const_p_buffer = std::shared_ptr<const buffer<T>>;

/// a helper for explicitly casting to a const buffer pointer.
template <typename T>
hamr::const_p_buffer<T> const_ptr(const hamr::p_buffer<T> &v)
{
    return hamr::const_p_buffer<T>(v);
}

/// a helper for getting a reference to pointed to hamr::buffer
template <typename T>
const hamr::buffer<T> &ref_to(const hamr::const_p_buffer<T> &ptr)
{
    return *(ptr.get());
}

/// a helper for getting a reference to pointed to hamr::buffer
template <typename T>
hamr::buffer<T> &ref_to(const hamr::p_buffer<T> &ptr)
{
    return *(ptr.get());
}

}

#endif
