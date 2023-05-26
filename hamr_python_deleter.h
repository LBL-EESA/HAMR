#ifndef hamr_python_deleter_h
#define hamr_python_deleter_h

#include "hamr_config.h"
#include <Python.h>

namespace hamr
{

/// a deleter for memory managed from within Python
/** This class manages an array allocated by a Python code. In the functor's
 * constructor a refrence to a user provdied Python object is stolen. When the
 * functor is invoked, a reference to this Python object is released. It is up
 * to the Python object to free the memory. One may use a PyCapsule to
 * implement custom delete methods if they are needed.
 */
template <typename T>
class HAMR_EXPORT python_deleter
{
public:
    /** constructs the deleter. A reference to obj is stolen by this constructor.
     * @param[in] ptr a pointer to shared data
     * @param[in] n_elem the number of elements of type T shared
     * @param[in] obj a PyObject who's reference count will be decremented when
     *                the data shared from Python is no longer needed.
     */
    python_deleter(T *ptr, size_t n_elem, PyObject *obj);

    /** deletes the array
     * @param[in] ptr the pointer to the array to delete. must be the same as
     *                that passed during construction.
     */
    void operator()(T *ptr);

private:
    T *m_ptr;
    size_t m_elem;
    PyObject *m_object;
};

}

#if !defined(HAMR_SEPARATE_IMPL)
#include "hamr_python_deleter_impl.h"
#endif

#endif
