#include "hamr_gil_state.h"
#include <Python.h>
#include <iostream>

namespace hamr
{

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
