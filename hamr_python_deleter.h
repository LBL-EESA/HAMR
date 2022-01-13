#ifndef hamr_python_deleter_h
#define hamr_python_deleter_h

#include "hamr_config.h"
#include "hamr_gil_state.h"

#include <Python.h>
#include <iostream>

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

// --------------------------------------------------------------------------
template <typename T>
python_deleter<T>::python_deleter(T *ptr, size_t n, PyObject *obj)
    : m_ptr(ptr), m_elem(n), m_object(obj)
{
#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "created python_deleter for array of " << n
            << " objects of type " << typeid(T).name() << sizeof(T)
            << " holding a reference to " << m_object << std::endl;
    }
#endif
    hamr::gil_state gil;
    Py_INCREF(obj);
}

// --------------------------------------------------------------------------
template <typename T>
void python_deleter<T>::operator()(T *ptr)
{
    (void)ptr;
    assert(ptr == m_ptr);
#if defined(HAMR_VERBOSE)
    if (hamr::get_verbose())
    {
        std::cerr << "python_deleter deleting array of " << m_elem
            << " objects of type " << typeid(T).name() << sizeof(T)
            << " release reference to " << m_object << std::endl;
    }
#endif
    hamr::gil_state gil;
    Py_DECREF(m_object);
}

}
#endif
