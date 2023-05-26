#include "hamr_config.h"

#include "hamr_python_deleter.h"
#include "hamr_python_deleter_impl.h"

template class hamr::python_deleter<float>;
template class hamr::python_deleter<double>;
template class hamr::python_deleter<char>;
template class hamr::python_deleter<signed char>;
template class hamr::python_deleter<short>;
template class hamr::python_deleter<int>;
template class hamr::python_deleter<long>;
template class hamr::python_deleter<long long>;
template class hamr::python_deleter<unsigned char>;
template class hamr::python_deleter<unsigned short>;
template class hamr::python_deleter<unsigned int>;
template class hamr::python_deleter<unsigned long>;
template class hamr::python_deleter<unsigned long long>;

#include "hamr_buffer.h"
#include "hamr_buffer_impl.h"

#define hamr_buffer_instantiate_python(T) \
template hamr::buffer<T>::buffer(allocator alloc, const hamr::stream &strm, transfer sync, size_t size, int owner, T *ptr, hamr::python_deleter<T> df);

hamr_buffer_instantiate_python(float)
hamr_buffer_instantiate_python(double)
hamr_buffer_instantiate_python(char)
hamr_buffer_instantiate_python(signed char)
hamr_buffer_instantiate_python(short)
hamr_buffer_instantiate_python(int)
hamr_buffer_instantiate_python(long)
hamr_buffer_instantiate_python(long long)
hamr_buffer_instantiate_python(unsigned char)
hamr_buffer_instantiate_python(unsigned short)
hamr_buffer_instantiate_python(unsigned int)
hamr_buffer_instantiate_python(unsigned long)
hamr_buffer_instantiate_python(unsigned long long)
