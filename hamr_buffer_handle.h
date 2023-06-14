#ifndef hamr_buffer_handle_h
#define hamr_buffer_handle_h

#include "hamr_config.h"
#include "hamr_env.h"
#include "hamr_buffer_allocator.h"
#include "hamr_gil_state.h"

#include <Python.h>
#include <memory>
#include <iostream>

namespace hamr
{
/// traits for Numpy's array interface protocol
template <typename cpp_t> struct array_interface_tt
{};

#define array_interface_tt_declare(ENDIAN, KIND, SIZE, CPP_T) \
/** CPP_T traits for Numpy's array interface protocol */      \
template <> struct array_interface_tt<CPP_T>                  \
{                                                             \
    /** number of bytes per element (SIZE) */                 \
    static constexpr size_t elemsize()                        \
    {                                                         \
        return SIZE;                                          \
    }                                                         \
    /** char code for the type (KIND) */                      \
    static constexpr char typekind()                          \
    {                                                         \
        return (#KIND) [0];                                   \
    }                                                         \
    /** true if the machine order is little endian */         \
    static constexpr bool little_endian()                     \
    {                                                         \
        return (#ENDIAN) [0] == '<';                          \
    }                                                         \
    /** Numpy type descriptor string (ENDIAN KIND SIZE) */    \
    static constexpr const char *descr()                      \
    {                                                         \
        return #ENDIAN #KIND #SIZE;                           \
    }                                                         \
};
array_interface_tt_declare(<, i, 1, char)
array_interface_tt_declare(<, i, 2, short)
array_interface_tt_declare(<, i, 4, int)
array_interface_tt_declare(<, i, 8, long)
array_interface_tt_declare(<, i, 8, long long)
array_interface_tt_declare(<, u, 1, unsigned char)
array_interface_tt_declare(<, u, 2, unsigned short)
array_interface_tt_declare(<, u, 4, unsigned int)
array_interface_tt_declare(<, u, 8, unsigned long)
array_interface_tt_declare(<, u, 8, unsigned long long)
array_interface_tt_declare(<, f, 4, float)
array_interface_tt_declare(<, f, 8, double)


/// type traits for constructing SWIG wrapped objects
template <typename cpp_t> struct buffer_handle_tt
{};

#define buffer_handle_tt_declare(CPP_T, SWIG_T) \
template <> struct buffer_handle_tt<CPP_T>      \
{                                               \
    static swig_type_info *swig_type()          \
    {                                           \
        return SWIG_T;                          \
    };                                          \
};

buffer_handle_tt_declare(float, SWIGTYPE_p_hamr__buffer_handleT_float_t)
buffer_handle_tt_declare(double, SWIGTYPE_p_hamr__buffer_handleT_double_t)
buffer_handle_tt_declare(char, SWIGTYPE_p_hamr__buffer_handleT_char_t)
buffer_handle_tt_declare(short, SWIGTYPE_p_hamr__buffer_handleT_short_t)
buffer_handle_tt_declare(int, SWIGTYPE_p_hamr__buffer_handleT_int_t)
buffer_handle_tt_declare(long, SWIGTYPE_p_hamr__buffer_handleT_long_t)
buffer_handle_tt_declare(long long, SWIGTYPE_p_hamr__buffer_handleT_long_long_t)
buffer_handle_tt_declare(unsigned char, SWIGTYPE_p_hamr__buffer_handleT_unsigned_char_t)
buffer_handle_tt_declare(unsigned short, SWIGTYPE_p_hamr__buffer_handleT_unsigned_short_t)
buffer_handle_tt_declare(unsigned int, SWIGTYPE_p_hamr__buffer_handleT_unsigned_int_t)
buffer_handle_tt_declare(unsigned long, SWIGTYPE_p_hamr__buffer_handleT_unsigned_long_t)
buffer_handle_tt_declare(unsigned long long, SWIGTYPE_p_hamr__buffer_handleT_unsigned_long_long_t)


/** A resource management class that is used to keep data shared with Python
 * codes from a ::hamr_buffer alive while it is being accessed by the Python
 * codes. The class also implements the Numpy array interface protocol and
 * the Numba CUDA array interface protocol enabling seamless access by those
 * libraries.
 */
template <typename T>
class HAMR_EXPORT buffer_handle
{
public:
    /// construct an empty, and unusable object
    buffer_handle() : m_data(nullptr), m_size(0),
        m_read_only(0), m_host_accessible(0), m_cuda_accessible(0),
        m_stream(0)
        {}

    /// construct from existing data
    buffer_handle(const std::shared_ptr<T> &src, size_t size,
        int read_only, int host_accessible, int cuda_accessible,
        size_t stream);

    /// construct from existing read only data
    buffer_handle(const std::shared_ptr<const T> &src, size_t size,
        int host_accessible, int cuda_accessible, size_t stream);

    /// destruct
    ~buffer_handle();

    /// copy construct
    buffer_handle(const buffer_handle<T> &other);

    /// move construct
    buffer_handle(buffer_handle<T> &&other);

    /// copy assign
    buffer_handle<T> &operator=(const buffer_handle<T> &) = default;

    /// move assign
    buffer_handle<T> &operator=(buffer_handle<T> &&) = default;


    /** returns a dictionary as described in the Numba CUDA array interface
     * protocol if the data is accessible from CUDA. Otherwise, an
     * AttributeError is rasied. */
    PyObject *get_cuda_array_interface();

    /** returns a dictionary as described in the Numpy __array_interface__
     * protocol if the data is accessible from CUDA. Otherwise, an
     * AttributeError is rasied. */
    PyObject *get_numpy_array_interface();

//private:
    /** returns a dictionary as decribed in the Numpy array interface and Numba
     * CUDA array interface protocols. The caller must ensure the data is
     * accessible in the desired technology before use. See
     * ::get_numpy_array_interface and ::get_cuda_array_interface which
     * implement check ensuring that this is true. */
    PyObject *get_array_interface();

    void to_stream(std::ostream &os) const;

    std::shared_ptr<T> m_data;
    size_t m_size;
    int m_read_only;
    int m_host_accessible;
    int m_cuda_accessible;
    size_t m_stream;
};

// **************************************************************************
template <typename T>
std::ostream &operator<<(std::ostream &os, const buffer_handle<T> &buf)
{
    buf.to_stream(os);
    return os;
}

// --------------------------------------------------------------------------
template <typename T>
void buffer_handle<T>::to_stream(std::ostream &os) const
{
    os << "buffer_handle<" << typeid(T).name() << sizeof(T)
        << "> m_data = " << size_t(this->m_data.get())
        << " m_size = " << this->m_size << " m_read_only = "
        << this->m_read_only << " m_host_accessible = "
        << this->m_host_accessible << " m_cuda_accessible = "
        << this->m_cuda_accessible;
}

// --------------------------------------------------------------------------
template <typename T>
buffer_handle<T>::buffer_handle(const std::shared_ptr<T> &src,
    size_t size, int read_only, int host_accessible, int cuda_accessible,
    size_t stream) : m_data(src), m_size(size), m_read_only(read_only),
    m_host_accessible(host_accessible), m_cuda_accessible(cuda_accessible),
    m_stream(stream)
{
    if (hamr::get_verbose())
    {
        std::cerr << "buffer_handle::construct " << *this << std::endl;
    }
}

// --------------------------------------------------------------------------
template <typename T>
buffer_handle<T>::buffer_handle(const std::shared_ptr<const T> &src,
    size_t size, int host_accessible, int cuda_accessible,
    size_t stream) : buffer_handle(std::const_pointer_cast<T>(src), size,
    1, host_accessible, cuda_accessible, stream)
{
}

// --------------------------------------------------------------------------
template <typename T>
buffer_handle<T>::~buffer_handle()
{
    if (hamr::get_verbose())
    {
        std::cerr << "buffer_handle::destruct " << *this << std::endl;
    }
}

// --------------------------------------------------------------------------
template <typename T>
buffer_handle<T>::buffer_handle(const buffer_handle<T> &other) :
    m_data(other.m_data), m_size(other.m_size),
    m_read_only(other.m_read_only), m_host_accessible(other.m_host_accessible),
    m_cuda_accessible(other.m_cuda_accessible), m_stream(other.m_stream)
{
    if (hamr::get_verbose())
    {
        std::cerr << "buffer_handle::copy construct " << *this << std::endl;
    }
}

// --------------------------------------------------------------------------
template <typename T>
buffer_handle<T>::buffer_handle(buffer_handle<T> &&other) :
    m_data(std::move(other.m_data)), m_size(other.m_size),
    m_read_only(other.m_read_only), m_host_accessible(other.m_host_accessible),
    m_cuda_accessible(other.m_cuda_accessible), m_stream(other.m_stream)
{
    if (hamr::get_verbose())
    {
        std::cerr << "buffer_handle::move construct " << *this << std::endl;
    }
}

// --------------------------------------------------------------------------
template <typename T>
PyObject *buffer_handle<T>::get_cuda_array_interface()
{
    hamr::gil_state gil;

    if (!m_cuda_accessible)
    {
        PyErr_SetString(PyExc_AttributeError,
            "The data is not accessible from CUDA");
        Py_RETURN_NONE;
    }

    if (hamr::get_verbose())
    {
        std::cerr << "buffer_handle<" << typeid(T).name() << sizeof(T)
            << ">::get_cuda_array_interface()" << std::endl;
    }

    return this->get_array_interface();
}

// --------------------------------------------------------------------------
template <typename T>
PyObject *buffer_handle<T>::get_numpy_array_interface()
{
    hamr::gil_state gil;

    if (!m_host_accessible)
    {
        PyErr_SetString(PyExc_AttributeError,
            "The data is not accessible from the host");
        Py_RETURN_NONE;
    }

    if (hamr::get_verbose())
    {
        std::cerr << "buffer_handle<" << typeid(T).name() << sizeof(T)
            << ">::get_numpy_array_interface()" << std::endl;
    }

    return this->get_array_interface();
}

// --------------------------------------------------------------------------
template <typename T>
PyObject *buffer_handle<T>::get_array_interface()
{
    // shape
    PyObject *shape = PyTuple_New(1);
    PyTuple_SetItem(shape, 0, PyLong_FromLong(m_size));

    // typestr
    PyObject *typestr = PyUnicode_FromString(array_interface_tt<T>::descr());
    Py_INCREF(typestr); // for use in descr

    // descr
    PyObject *descr_tup = PyTuple_New(2);
    PyTuple_SetItem(descr_tup, 0, PyUnicode_FromString(""));
    PyTuple_SetItem(descr_tup, 1, typestr);
    PyObject *descr = PyList_New(1);
    PyList_SetItem(descr, 0, descr_tup);

    // data
    PyObject *data = PyTuple_New(2);
    PyTuple_SetItem(data, 0, PyLong_FromSize_t(size_t(m_data.get())));
    PyTuple_SetItem(data, 1, PyBool_FromLong(m_read_only));

    // strides
    PyObject *strides = Py_None;

    // mask
    PyObject *mask = Py_None;

    // stream
    PyObject *strm = PyLong_FromSize_t(m_stream);

    // version
    PyObject *version = PyLong_FromLong(3);

    // build the __array_interface__ dictionary
    PyObject *aint = PyDict_New();
    PyDict_SetItemString(aint, "shape", shape);
    PyDict_SetItemString(aint, "typestr", typestr);
    PyDict_SetItemString(aint, "descr", descr);
    PyDict_SetItemString(aint, "data", data);
    PyDict_SetItemString(aint, "strides", strides);
    PyDict_SetItemString(aint, "mask", mask);
    PyDict_SetItemString(aint, "stream", strm);
    PyDict_SetItemString(aint, "version", version);

    Py_DECREF(shape);
    Py_DECREF(typestr);
    Py_DECREF(descr);
    Py_DECREF(data);
    Py_DECREF(strm);
    Py_DECREF(version);

    return aint;
}

}
#endif
