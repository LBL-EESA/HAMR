%{
#include "hamr_config.h"
#include "hamr_buffer_handle.h"
#include "hamr_gil_state.h"
%}

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
        hamr::gil_state gil;
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
