%{
#include "hamr_config.h"
#include "hamr_buffer.h"
#include "hamr_buffer_util.h"
#include "hamr_buffer_handle.h"
#include "hamr_gil_state.h"
#include "hamr_stream.h"
%}

/***************************************************************************
 * buffer
 **************************************************************************/
%namewarn("") "print";
%ignore hamr::buffer::operator=;

%include "hamr_buffer.h"

%extend hamr::buffer
{
    /** zero-copy construct from a Python object supporting the Numpy array
     * interface or the NUMBA CUDA array interface
     */
    buffer(hamr::buffer_allocator alloc, size_t intstrm,
        hamr::buffer_transfer sync, size_t n_elem,
        int owner, size_t intptr, PyObject *src)
    {
        hamr::gil_state gil;

        T *ptr = (T*)intptr;

#if defined(HAMR_ENABLE_CUDA)
        hamr::stream strm((cudaStream_t)intstrm);
#elif defined(HAMR_ENABLE_HIP)
        hamr::stream strm((hipStream_t)intstrm);
#else
        hamr::stream strm;
#endif

        return new hamr::buffer<T>(alloc, strm, sync, n_elem, owner,
            ptr, hamr::python_deleter<T>(ptr, n_elem, src));
    }

    PyObject *__str__()
    {
        hamr::gil_state gil;

        std::ostringstream oss;
        oss << "{";
        size_t n_elem = self->size();
        if (n_elem)
        {
            auto [spb, pb] = get_host_accessible(*self);

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

    /** return an object that can be used on the host */
    hamr::buffer_handle<T> get_host_accessible()
    {
        hamr::gil_state gil;

        std::shared_ptr<T> ptr
            (std::const_pointer_cast<T>(self->get_host_accessible()));

        hamr::buffer_handle<T> h(ptr, self->size(), 0, 1,
            self->host_accessible() && self->cuda_accessible(),
            self->get_stream().get_stream());

        return h;
    }

    /** return an object that can be used from CUDA */
    hamr::buffer_handle<T> get_cuda_accessible()
    {
        hamr::gil_state gil;

        std::shared_ptr<T> ptr
            (std::const_pointer_cast<T>(self->get_cuda_accessible()));

        hamr::buffer_handle<T> h(ptr, self->size(), 0,
            self->host_accessible() && self->cuda_accessible(), 1,
            self->get_stream().get_stream());

        return h;
    }
}

/* named buffers */
%template(buffer_float) hamr::buffer<float>;
%template(buffer_double) hamr::buffer<double>;
%template(buffer_char) hamr::buffer<char>;
%template(buffer_signed_char) hamr::buffer<signed char>;
%template(buffer_short) hamr::buffer<short>;
%template(buffer_int) hamr::buffer<int>;
%template(buffer_long) hamr::buffer<long>;
%template(buffer_long_long) hamr::buffer<long long>;
%template(buffer_unsigned_char) hamr::buffer<unsigned char>;
%template(buffer_unsigned_short) hamr::buffer<unsigned short>;
%template(buffer_unsigned_int) hamr::buffer<unsigned int>;
%template(buffer_unsigned_long) hamr::buffer<unsigned long>;
%template(buffer_unsigned_long_long) hamr::buffer<unsigned long long>;

/* zero-copy constructor */
%pythoncode
{
def buffer(obj, **kwargs):
    """
    Zero-copy construct a C++ hamr::buffer<T> instance from Python objects that
    support the Numpy array interface protocol or the Numba CUDA array
    interface. The data type of the hamr::buffer is automatically determined. A
    reference to the object sharing the data is held while the hamr::buffer is
    using it. By default the buffer will use the same stream as the cupy
    object. This can be overridden by passing a 'stream' kwargs. Optionally a
    transfer mode may be specified as well via the 'sync' kwargs. The default
    transfer mode is sync_host.
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

    strm = None
    if 'stream' in kwargs:
        strm = kwargs['stream']
    elif strm in aint:
        strm = aint['stream']
    if strm is None:
        strm = 2

    if 'sync' in kwargs:
        sync = kwargs['sync']
    else:
        sync = buffer_transfer_sync_host

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
            buf = buffer_double(alloc, strm, sync, n_elem, -1, intptr, obj)
        elif typestr[2] == '4':
            buf = buffer_float(alloc, strm, sync, n_elem, -1, intptr, obj)
        else:
            raise TypeError('Unsupported floating point size %s' % (typestr[2]))

    elif typestr[1] == 'i':
        if typestr[2] == '8':
            buf = buffer_long(alloc, strm, sync, n_elem, -1, intptr, obj)
        elif typestr[2] == '4':
            buf = buffer_int(alloc, strm, sync, n_elem, -1, intptr, obj)
        elif typestr[2] == '2':
            buf = buffer_short(alloc, strm, sync, n_elem, -1, intptr, obj)
        elif typestr[2] == '1':
            buf = buffer_char(alloc, strm, sync, n_elem, -1, intptr, obj)
        else:
            raise TypeError('Unsupported integer size %s' % (typestr[2]))

    elif typestr[1] == 'u':
        if typestr[2] == '8':
            buf = buffer_unsigned_long(alloc, strm, sync, n_elem, -1, intptr, obj)
        elif typestr[2] == '4':
            buf = buffer_unsigned_int(alloc, strm, sync, n_elem, -1, intptr, obj)
        elif typestr[2] == '2':
            buf = buffer_unsigned_short(alloc, strm, sync, n_elem, -1, intptr, obj)
        elif typestr[2] == '1':
            buf = buffer_unsigned_char(alloc, strm, sync, n_elem, -1, intptr, obj)
        else:
            raise TypeError('Unsupported integer size %s' % (typestr[2]))

    else:
        raise TypeError('Unsupported data type code %s' % (typestr[1]))

    return buf
}
