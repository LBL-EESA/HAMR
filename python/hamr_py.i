%module hamr
%{
#define SWIG_FILE_WITH_INIT

#include "Python.h"
#include "hamr_config.h"
#include "hamr_buffer.h"
#include "hamr_buffer_allocator.h"
#include "hamr_buffer_handle.h"
#include "hamr_python_deleter.h"

#include <iostream>
#include <sstream>

/* disable some warnings that are present in SWIG generated code. */
#if __GNUC__ > 8
#pragma GCC diagnostic ignored "-Wcast-function-type"
#endif
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#if defined(__CUDACC__)
#pragma diag_suppress = set_but_not_used
#endif
%}

/* SWIG doens't understand compiler attriibbutes */
#define __attribute__(x)

/* enable STL classes */
%include "shared_ptr.i"

/***************************************************************************
 * expose the build configuration
 **************************************************************************/
%include "hamr_config.h"

/***************************************************************************
 * buffer allocator enumerations
 **************************************************************************/
%include "hamr_buffer_allocator.h"

/***************************************************************************
 * buffer handle
 **************************************************************************/
%ignore hamr::buffer_handle::buffer_handle(buffer_handle &&);
%ignore hamr::buffer_handle::operator=;

%include "hamr_buffer_handle.h"

%extend hamr::buffer_handle
{
    PyObject *__str__()
    {
        std::ostringstream oss;
        self->to_stream(oss);
        return PyUnicode_FromString(oss.str().c_str());
    }

%pythoncode
{
    @property
    def __array_interface__(self):
        return self.get_numpy_array_interface()

    @property
    def __cuda_array_interface__(self):
        return self.get_cuda_array_interface()
}
}

/* named buffer_handles */
%template(buffer_handle_float) hamr::buffer_handle<float>;
%template(buffer_handle_double) hamr::buffer_handle<double>;
%template(buffer_handle_char) hamr::buffer_handle<char>;
%template(buffer_handle_short) hamr::buffer_handle<short>;
%template(buffer_handle_int) hamr::buffer_handle<int>;
%template(buffer_handle_long) hamr::buffer_handle<long>;
%template(buffer_handle_long_long) hamr::buffer_handle<long long>;
%template(buffer_handle_unsigned_char) hamr::buffer_handle<unsigned char>;
%template(buffer_handle_unsigned_short) hamr::buffer_handle<unsigned short>;
%template(buffer_handle_unsigned_int) hamr::buffer_handle<unsigned int>;
%template(buffer_handle_unsigned_long) hamr::buffer_handle<unsigned long>;
%template(buffer_handle_unsigned_long_long) hamr::buffer_handle<unsigned long long>;


/***************************************************************************
 * buffer
 **************************************************************************/
%rename(_print) hamr::buffer::print;
%ignore hamr::buffer::operator=;
%ignore hamr::buffer::buffer(buffer &&);

%include "hamr_buffer.h"

%extend hamr::buffer
{
    buffer(hamr::buffer_allocator alloc, size_t n_elem,
        int owner, size_t intptr, PyObject *src)
    {
        T *ptr = (T*)intptr;

        return new hamr::buffer<T>(alloc, n_elem, owner,
            ptr, hamr::python_deleter(ptr, n_elem, src));
    }

    PyObject *__str__()
    {
        std::ostringstream oss;
        oss << "{";
        size_t n_elem = self->size();
        if (n_elem)
        {
            auto spb = self->get_cpu_accessible();
            T *pb = spb.get();

            oss << pb[0];

            for (size_t i = 1; i < n_elem; ++i)
            {
                oss << ", ";
                if ((i % 8) == 0)
                    oss << std::endl;
                oss << pb[i];
            }
        }
        oss << "}";
        return PyUnicode_FromString(oss.str().c_str());
    }

    /** return an object that can be used on the CPU */
    hamr::buffer_handle<T> get_cpu_accessible()
    {
        hamr::buffer_handle<T> h(self->get_cpu_accessible(), self->size(),
            0, 1, self->cpu_accessible() && self->cuda_accessible());

        return h;
    }

    /** return an object that can be used from CUDA */
    hamr::buffer_handle<T> get_cuda_accessible()
    {
        hamr::buffer_handle<T> h(self->get_cuda_accessible(), self->size(),
            0, self->cpu_accessible() && self->cuda_accessible(), 1);

        return h;
    }
}


/* named buffers */
%template(buffer_float) hamr::buffer<float>;
%template(buffer_double) hamr::buffer<double>;
%template(buffer_char) hamr::buffer<char>;
%template(buffer_short) hamr::buffer<short>;
%template(buffer_int) hamr::buffer<int>;
%template(buffer_long) hamr::buffer<long>;
%template(buffer_long_long) hamr::buffer<long long>;
%template(buffer_unsigned_char) hamr::buffer<unsigned char>;
%template(buffer_unsigned_short) hamr::buffer<unsigned short>;
%template(buffer_unsigned_int) hamr::buffer<unsigned int>;
%template(buffer_unsigned_long) hamr::buffer<unsigned long>;
%template(buffer_unsigned_long_long) hamr::buffer<unsigned long long>;


%pythoncode
{
def buffer(obj):
    """
    Zero-copy construct a C++ hamr::buffer<T> instance from Python objects that
    support the Numpy array interface protocol or the Numba CUDA array
    interface. The data type of the hamr::buffer is automatically determined. A
    reference to the object sharing the data is held while the hamr::buffer is
    using it.
    """

    aint = None
    alloc = None
    if hasattr(obj, '__cuda_array_interface__'):
        alloc = buffer_allocator_cuda
        aint = obj.__cuda_array_interface__
    elif hasattr(obj, '__array_interface__'):
        alloc = buffer_allocator_malloc
        aint = obj.__array_interface__
    else:
        raise AttributeError('Failed to zero-copy construct the hamr::buffer' \
                             ' because the object providing the data does not' \
                             ' implement the array interface protocol')

    data = aint['data']
    intptr = data[0]

    shape = aint['shape']
    n_elem = 1
    for ndim in shape:
        n_elem *= ndim

    typestr = aint['typestr']

    if not typestr[0] == '<':
        raise TypeError('The shared data must have little endian byte'
                        ' order but it has endian code %s' % (typestr[0]))

    if typestr[1] == 'f':
        if typestr[2] == '8':
            buf = buffer_double(alloc, n_elem, -1, intptr, obj)
        elif typestr[2] == '4':
            buf = buffer_float(alloc, n_elem, -1, intptr, obj)
        else:
            raise TypeError('Unsupported floating point size %s' % (typestr[2]))

    elif typestr[1] == 'i':
        if typestr[2] == '8':
            buf = buffer_long(alloc, n_elem, -1, intptr, obj)
        elif typestr[2] == '4':
            buf = buffer_int(alloc, n_elem, -1, intptr, obj)
        elif typestr[2] == '2':
            buf = buffer_short(alloc, n_elem, -1, intptr, obj)
        elif typestr[2] == '1':
            buf = buffer_char(alloc, n_elem, -1, intptr, obj)
        else:
            raise TypeError('Unsupported integer size %s' % (typestr[2]))

    elif typestr[1] == 'u':
        if typestr[2] == '8':
            buf = buffer_unsigned_long(alloc, n_elem, -1, intptr, obj)
        elif typestr[2] == '4':
            buf = buffer_unsigned_int(alloc, n_elem, -1, intptr, obj)
        elif typestr[2] == '2':
            buf = buffer_unsigned_short(alloc, n_elem, -1, intptr, obj)
        elif typestr[2] == '1':
            buf = buffer_unsigned_char(alloc, n_elem, -1, intptr, obj)
        else:
            raise TypeError('Unsupported integer size %s' % (typestr[2]))

    else:
        raise TypeError('Unsupported data type code %s' % (typestr[1]))

    return buf
}
