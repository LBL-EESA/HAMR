%module hamr
%{
#define SWIG_FILE_WITH_INIT

#include "Python.h"
#include "hamr_config.h"
#include "hamr_buffer.h"
#include "hamr_buffer_allocator.h"
#include "hamr_buffer_handle.h"

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
