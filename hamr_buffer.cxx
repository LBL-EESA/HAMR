#include "hamr_config.h"

#include "hamr_buffer.h"
#include "hamr_buffer_impl.h"

#define hamr_buffer_instantiate_members(T, U) \
\
template hamr::buffer<T>::buffer(const buffer<U> &other); \
template hamr::buffer<T>::buffer(allocator alloc, const hamr::stream &strm, transfer sync, const buffer<U> &other); \
template hamr::buffer<T>::buffer(allocator alloc, const hamr::stream &strm, const buffer<U> &other); \
\
template void hamr::buffer<T>::operator=(const buffer<U> &other);\
\
template int hamr::buffer<T>::assign(const U *src, size_t src_start, size_t n_vals); \
template int hamr::buffer<T>::assign(const buffer<U> &src, size_t src_start, size_t n_vals); \
template int hamr::buffer<T>::assign(const buffer<U> &src); \
\
template int hamr::buffer<T>::append(const U *src, size_t src_start, size_t n_vals); \
template int hamr::buffer<T>::append(const buffer<U> &src, size_t src_start, size_t n_vals); \
template int hamr::buffer<T>::append(const buffer<U> &src); \
\
template int hamr::buffer<T>::set(size_t dest_start, const U *src, size_t src_start, size_t n_vals); \
template int hamr::buffer<T>::set(size_t dest_start, const buffer<U> &src, size_t src_start, size_t n_vals); \
\
template int hamr::buffer<T>::get(size_t src_start, U *dest, size_t dest_start, size_t n_vals) const; \
template int hamr::buffer<T>::get(size_t src_start, buffer<U> &dest, size_t dest_start, size_t n_vals) const;

#define hamr_buffer_instantiate(T) \
template class hamr::buffer<T>; \
template hamr::buffer<T>::buffer(allocator alloc, const hamr::stream &strm, transfer sync, const buffer<T> &other); \
template hamr::buffer<T>::buffer(allocator alloc, const hamr::stream &strm, const buffer<T> &other); \
template int hamr::buffer<T>::set(size_t dest_start, const T *src, size_t src_start, size_t n_vals); \
template int hamr::buffer<T>::set(size_t dest_start, const buffer<T> &src, size_t src_start, size_t n_vals); \
template int hamr::buffer<T>::set(const buffer<T> &src); \
template int hamr::buffer<T>::get(size_t src_start, T *dest, size_t dest_start, size_t n_vals) const; \
template int hamr::buffer<T>::get(size_t src_start, buffer<T> &dest, size_t dest_start, size_t n_vals) const; \
template int hamr::buffer<T>::get(buffer<T> &dest) const; \
template int hamr::buffer<T>::append(const T *src, size_t src_start, size_t n_vals); \
template int hamr::buffer<T>::append(const buffer<T> &src, size_t src_start, size_t n_vals); \
template int hamr::buffer<T>::append(const buffer<T> &src); \
template int hamr::buffer<T>::assign(const T *src, size_t src_start, size_t n_vals); \
template int hamr::buffer<T>::assign(const buffer<T> &src, size_t src_start, size_t n_vals); \
template int hamr::buffer<T>::assign(const buffer<T> &src);

hamr_buffer_instantiate(float)
//hamr_buffer_instantiate_members(float, float)
hamr_buffer_instantiate_members(float, double)
hamr_buffer_instantiate_members(float, char)
hamr_buffer_instantiate_members(float, signed char)
hamr_buffer_instantiate_members(float, short)
hamr_buffer_instantiate_members(float, int)
hamr_buffer_instantiate_members(float, long)
hamr_buffer_instantiate_members(float, long long)
hamr_buffer_instantiate_members(float, unsigned char)
hamr_buffer_instantiate_members(float, unsigned short)
hamr_buffer_instantiate_members(float, unsigned int)
hamr_buffer_instantiate_members(float, unsigned long)
hamr_buffer_instantiate_members(float, unsigned long long)

hamr_buffer_instantiate(double)
hamr_buffer_instantiate_members(double, float)
//hamr_buffer_instantiate_members(double, double)
hamr_buffer_instantiate_members(double, char)
hamr_buffer_instantiate_members(double, signed char)
hamr_buffer_instantiate_members(double, short)
hamr_buffer_instantiate_members(double, int)
hamr_buffer_instantiate_members(double, long)
hamr_buffer_instantiate_members(double, long long)
hamr_buffer_instantiate_members(double, unsigned char)
hamr_buffer_instantiate_members(double, unsigned short)
hamr_buffer_instantiate_members(double, unsigned int)
hamr_buffer_instantiate_members(double, unsigned long)
hamr_buffer_instantiate_members(double, unsigned long long)

hamr_buffer_instantiate(char)
hamr_buffer_instantiate_members(char, float)
hamr_buffer_instantiate_members(char, double)
//hamr_buffer_instantiate_members(char, char)
hamr_buffer_instantiate_members(char, signed char)
hamr_buffer_instantiate_members(char, short)
hamr_buffer_instantiate_members(char, int)
hamr_buffer_instantiate_members(char, long)
hamr_buffer_instantiate_members(char, long long)
hamr_buffer_instantiate_members(char, unsigned char)
hamr_buffer_instantiate_members(char, unsigned short)
hamr_buffer_instantiate_members(char, unsigned int)
hamr_buffer_instantiate_members(char, unsigned long)
hamr_buffer_instantiate_members(char, unsigned long long)

hamr_buffer_instantiate(signed char)
hamr_buffer_instantiate_members(signed char, float)
hamr_buffer_instantiate_members(signed char, double)
hamr_buffer_instantiate_members(signed char, char)
//hamr_buffer_instantiate_members(signed char, signed char)
hamr_buffer_instantiate_members(signed char, short)
hamr_buffer_instantiate_members(signed char, int)
hamr_buffer_instantiate_members(signed char, long)
hamr_buffer_instantiate_members(signed char, long long)
hamr_buffer_instantiate_members(signed char, unsigned char)
hamr_buffer_instantiate_members(signed char, unsigned short)
hamr_buffer_instantiate_members(signed char, unsigned int)
hamr_buffer_instantiate_members(signed char, unsigned long)
hamr_buffer_instantiate_members(signed char, unsigned long long)

hamr_buffer_instantiate(short)
hamr_buffer_instantiate_members(short, float)
hamr_buffer_instantiate_members(short, double)
hamr_buffer_instantiate_members(short, char)
hamr_buffer_instantiate_members(short, signed char)
//hamr_buffer_instantiate_members(short, short)
hamr_buffer_instantiate_members(short, int)
hamr_buffer_instantiate_members(short, long)
hamr_buffer_instantiate_members(short, long long)
hamr_buffer_instantiate_members(short, unsigned char)
hamr_buffer_instantiate_members(short, unsigned short)
hamr_buffer_instantiate_members(short, unsigned int)
hamr_buffer_instantiate_members(short, unsigned long)
hamr_buffer_instantiate_members(short, unsigned long long)

hamr_buffer_instantiate(int)
hamr_buffer_instantiate_members(int, float)
hamr_buffer_instantiate_members(int, double)
hamr_buffer_instantiate_members(int, char)
hamr_buffer_instantiate_members(int, signed char)
hamr_buffer_instantiate_members(int, short)
//hamr_buffer_instantiate_members(int, int)
hamr_buffer_instantiate_members(int, long)
hamr_buffer_instantiate_members(int, long long)
hamr_buffer_instantiate_members(int, unsigned char)
hamr_buffer_instantiate_members(int, unsigned short)
hamr_buffer_instantiate_members(int, unsigned int)
hamr_buffer_instantiate_members(int, unsigned long)
hamr_buffer_instantiate_members(int, unsigned long long)

hamr_buffer_instantiate(long)
hamr_buffer_instantiate_members(long, float)
hamr_buffer_instantiate_members(long, double)
hamr_buffer_instantiate_members(long, char)
hamr_buffer_instantiate_members(long, signed char)
hamr_buffer_instantiate_members(long, short)
hamr_buffer_instantiate_members(long, int)
//hamr_buffer_instantiate_members(long, long)
hamr_buffer_instantiate_members(long, long long)
hamr_buffer_instantiate_members(long, unsigned char)
hamr_buffer_instantiate_members(long, unsigned short)
hamr_buffer_instantiate_members(long, unsigned int)
hamr_buffer_instantiate_members(long, unsigned long)
hamr_buffer_instantiate_members(long, unsigned long long)

hamr_buffer_instantiate(long long)
hamr_buffer_instantiate_members(long long, float)
hamr_buffer_instantiate_members(long long, double)
hamr_buffer_instantiate_members(long long, char)
hamr_buffer_instantiate_members(long long, signed char)
hamr_buffer_instantiate_members(long long, short)
hamr_buffer_instantiate_members(long long, int)
hamr_buffer_instantiate_members(long long, long)
//hamr_buffer_instantiate_members(long long, long long)
hamr_buffer_instantiate_members(long long, unsigned char)
hamr_buffer_instantiate_members(long long, unsigned short)
hamr_buffer_instantiate_members(long long, unsigned int)
hamr_buffer_instantiate_members(long long, unsigned long)
hamr_buffer_instantiate_members(long long, unsigned long long)

hamr_buffer_instantiate(unsigned char)
hamr_buffer_instantiate_members(unsigned char, float)
hamr_buffer_instantiate_members(unsigned char, double)
hamr_buffer_instantiate_members(unsigned char, char)
hamr_buffer_instantiate_members(unsigned char, signed char)
hamr_buffer_instantiate_members(unsigned char, short)
hamr_buffer_instantiate_members(unsigned char, int)
hamr_buffer_instantiate_members(unsigned char, long)
hamr_buffer_instantiate_members(unsigned char, long long)
//hamr_buffer_instantiate_members(unsigned char, unsigned char)
hamr_buffer_instantiate_members(unsigned char, unsigned short)
hamr_buffer_instantiate_members(unsigned char, unsigned int)
hamr_buffer_instantiate_members(unsigned char, unsigned long)
hamr_buffer_instantiate_members(unsigned char, unsigned long long)


hamr_buffer_instantiate(unsigned short)
hamr_buffer_instantiate_members(unsigned short, float)
hamr_buffer_instantiate_members(unsigned short, double)
hamr_buffer_instantiate_members(unsigned short, char)
hamr_buffer_instantiate_members(unsigned short, signed char)
hamr_buffer_instantiate_members(unsigned short, short)
hamr_buffer_instantiate_members(unsigned short, int)
hamr_buffer_instantiate_members(unsigned short, long)
hamr_buffer_instantiate_members(unsigned short, long long)
hamr_buffer_instantiate_members(unsigned short, unsigned char)
//hamr_buffer_instantiate_members(unsigned short, unsigned short)
hamr_buffer_instantiate_members(unsigned short, unsigned int)
hamr_buffer_instantiate_members(unsigned short, unsigned long)
hamr_buffer_instantiate_members(unsigned short, unsigned long long)


hamr_buffer_instantiate(unsigned int)
hamr_buffer_instantiate_members(unsigned int, float)
hamr_buffer_instantiate_members(unsigned int, double)
hamr_buffer_instantiate_members(unsigned int, char)
hamr_buffer_instantiate_members(unsigned int, signed char)
hamr_buffer_instantiate_members(unsigned int, short)
hamr_buffer_instantiate_members(unsigned int, int)
hamr_buffer_instantiate_members(unsigned int, long)
hamr_buffer_instantiate_members(unsigned int, long long)
hamr_buffer_instantiate_members(unsigned int, unsigned char)
hamr_buffer_instantiate_members(unsigned int, unsigned short)
//hamr_buffer_instantiate_members(unsigned int, unsigned int)
hamr_buffer_instantiate_members(unsigned int, unsigned long)
hamr_buffer_instantiate_members(unsigned int, unsigned long long)

hamr_buffer_instantiate(unsigned long)
hamr_buffer_instantiate_members(unsigned long, float)
hamr_buffer_instantiate_members(unsigned long, double)
hamr_buffer_instantiate_members(unsigned long, char)
hamr_buffer_instantiate_members(unsigned long, signed char)
hamr_buffer_instantiate_members(unsigned long, short)
hamr_buffer_instantiate_members(unsigned long, int)
hamr_buffer_instantiate_members(unsigned long, long)
hamr_buffer_instantiate_members(unsigned long, long long)
hamr_buffer_instantiate_members(unsigned long, unsigned char)
hamr_buffer_instantiate_members(unsigned long, unsigned short)
hamr_buffer_instantiate_members(unsigned long, unsigned int)
//hamr_buffer_instantiate_members(unsigned long, unsigned long)
hamr_buffer_instantiate_members(unsigned long, unsigned long long)

hamr_buffer_instantiate(unsigned long long)
hamr_buffer_instantiate_members(unsigned long long, float)
hamr_buffer_instantiate_members(unsigned long long, double)
hamr_buffer_instantiate_members(unsigned long long, char)
hamr_buffer_instantiate_members(unsigned long long, signed char)
hamr_buffer_instantiate_members(unsigned long long, short)
hamr_buffer_instantiate_members(unsigned long long, int)
hamr_buffer_instantiate_members(unsigned long long, long)
hamr_buffer_instantiate_members(unsigned long long, long long)
hamr_buffer_instantiate_members(unsigned long long, unsigned char)
hamr_buffer_instantiate_members(unsigned long long, unsigned short)
hamr_buffer_instantiate_members(unsigned long long, unsigned int)
hamr_buffer_instantiate_members(unsigned long long, unsigned long)
//hamr_buffer_instantiate_members(unsigned long long, unsigned long long)
