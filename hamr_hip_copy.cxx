#include "hamr_config.h"

#include "hamr_hip_copy.h"
#include "hamr_hip_copy_impl.h"

#if !defined(HAMR_ENABLE_OBJECTS)

#define hamr_hip_copy_instantiate_(T, U) \
template int hamr::copy_to_hip_from_host<T,U>(T *dest, const U *src, size_t n_elem, void *); \
template int hamr::copy_to_hip_from_hip<T,U>(T *dest, const U *src, size_t n_elem, void *); \
template int hamr::copy_to_hip_from_hip<T,U>(T *dest, const U *src, int src_device, size_t n_elem, void *); \
template int hamr::copy_to_host_from_hip<T,U>(T *dest, const U *src, size_t n_elem, void *);

#else

#define hamr_hip_copy_instantiate_(T, U) \
template int hamr::copy_to_hip_from_host<T,U>(T *dest, const U *src, size_t n_elem); \
template int hamr::copy_to_hip_from_hip<T,U>(T *dest, const U *src, size_t n_elem); \
template int hamr::copy_to_hip_from_hip<T,U>(T *dest, const U *src, int src_device, size_t n_elem); \
template int hamr::copy_to_host_from_hip<T,U>(T *dest, const U *src, size_t n_elem);

#endif

#define hamr_hip_copy_instantiate__(T) \
template int hamr::copy_to_hip_from_host<T>(T *dest, const T *src, size_t n_elem, void *); \
template int hamr::copy_to_hip_from_hip<T>(T *dest, const T *src, size_t n_elem, void *); \
template int hamr::copy_to_hip_from_hip<T>(T *dest, const T *src, int src_device, size_t n_elem, void *); \
template int hamr::copy_to_host_from_hip<T>(T *dest, const T *src, size_t n_elem, void *);

hamr_hip_copy_instantiate__(float)
hamr_hip_copy_instantiate__(double)
hamr_hip_copy_instantiate__(char)
hamr_hip_copy_instantiate__(signed char)
hamr_hip_copy_instantiate__(short)
hamr_hip_copy_instantiate__(int)
hamr_hip_copy_instantiate__(long)
hamr_hip_copy_instantiate__(long long)
hamr_hip_copy_instantiate__(unsigned char)
hamr_hip_copy_instantiate__(unsigned short)
hamr_hip_copy_instantiate__(unsigned int)
hamr_hip_copy_instantiate__(unsigned long)
hamr_hip_copy_instantiate__(unsigned long long)

//hamr_hip_copy_instantiate_(float, float)
hamr_hip_copy_instantiate_(float, double)
hamr_hip_copy_instantiate_(float, char)
hamr_hip_copy_instantiate_(float, signed char)
hamr_hip_copy_instantiate_(float, short)
hamr_hip_copy_instantiate_(float, int)
hamr_hip_copy_instantiate_(float, long)
hamr_hip_copy_instantiate_(float, long long)
hamr_hip_copy_instantiate_(float, unsigned char)
hamr_hip_copy_instantiate_(float, unsigned short)
hamr_hip_copy_instantiate_(float, unsigned int)
hamr_hip_copy_instantiate_(float, unsigned long)
hamr_hip_copy_instantiate_(float, unsigned long long)

hamr_hip_copy_instantiate_(double, float)
//hamr_hip_copy_instantiate_(double, double)
hamr_hip_copy_instantiate_(double, char)
hamr_hip_copy_instantiate_(double, signed char)
hamr_hip_copy_instantiate_(double, short)
hamr_hip_copy_instantiate_(double, int)
hamr_hip_copy_instantiate_(double, long)
hamr_hip_copy_instantiate_(double, long long)
hamr_hip_copy_instantiate_(double, unsigned char)
hamr_hip_copy_instantiate_(double, unsigned short)
hamr_hip_copy_instantiate_(double, unsigned int)
hamr_hip_copy_instantiate_(double, unsigned long)
hamr_hip_copy_instantiate_(double, unsigned long long)

hamr_hip_copy_instantiate_(char, float)
hamr_hip_copy_instantiate_(char, double)
//hamr_hip_copy_instantiate_(char, char)
hamr_hip_copy_instantiate_(char, signed char)
hamr_hip_copy_instantiate_(char, short)
hamr_hip_copy_instantiate_(char, int)
hamr_hip_copy_instantiate_(char, long)
hamr_hip_copy_instantiate_(char, long long)
hamr_hip_copy_instantiate_(char, unsigned char)
hamr_hip_copy_instantiate_(char, unsigned short)
hamr_hip_copy_instantiate_(char, unsigned int)
hamr_hip_copy_instantiate_(char, unsigned long)
hamr_hip_copy_instantiate_(char, unsigned long long)

hamr_hip_copy_instantiate_(signed char, float)
hamr_hip_copy_instantiate_(signed char, double)
hamr_hip_copy_instantiate_(signed char, char)
//hamr_hip_copy_instantiate_(signed char, signed char)
hamr_hip_copy_instantiate_(signed char, short)
hamr_hip_copy_instantiate_(signed char, int)
hamr_hip_copy_instantiate_(signed char, long)
hamr_hip_copy_instantiate_(signed char, long long)
hamr_hip_copy_instantiate_(signed char, unsigned char)
hamr_hip_copy_instantiate_(signed char, unsigned short)
hamr_hip_copy_instantiate_(signed char, unsigned int)
hamr_hip_copy_instantiate_(signed char, unsigned long)
hamr_hip_copy_instantiate_(signed char, unsigned long long)

hamr_hip_copy_instantiate_(short, float)
hamr_hip_copy_instantiate_(short, double)
hamr_hip_copy_instantiate_(short, char)
hamr_hip_copy_instantiate_(short, signed char)
//hamr_hip_copy_instantiate_(short, short)
hamr_hip_copy_instantiate_(short, int)
hamr_hip_copy_instantiate_(short, long)
hamr_hip_copy_instantiate_(short, long long)
hamr_hip_copy_instantiate_(short, unsigned char)
hamr_hip_copy_instantiate_(short, unsigned short)
hamr_hip_copy_instantiate_(short, unsigned int)
hamr_hip_copy_instantiate_(short, unsigned long)
hamr_hip_copy_instantiate_(short, unsigned long long)

hamr_hip_copy_instantiate_(int, float)
hamr_hip_copy_instantiate_(int, double)
hamr_hip_copy_instantiate_(int, char)
hamr_hip_copy_instantiate_(int, signed char)
hamr_hip_copy_instantiate_(int, short)
//hamr_hip_copy_instantiate_(int, int)
hamr_hip_copy_instantiate_(int, long)
hamr_hip_copy_instantiate_(int, long long)
hamr_hip_copy_instantiate_(int, unsigned char)
hamr_hip_copy_instantiate_(int, unsigned short)
hamr_hip_copy_instantiate_(int, unsigned int)
hamr_hip_copy_instantiate_(int, unsigned long)
hamr_hip_copy_instantiate_(int, unsigned long long)

hamr_hip_copy_instantiate_(long, float)
hamr_hip_copy_instantiate_(long, double)
hamr_hip_copy_instantiate_(long, char)
hamr_hip_copy_instantiate_(long, signed char)
hamr_hip_copy_instantiate_(long, short)
hamr_hip_copy_instantiate_(long, int)
//hamr_hip_copy_instantiate_(long, long)
hamr_hip_copy_instantiate_(long, long long)
hamr_hip_copy_instantiate_(long, unsigned char)
hamr_hip_copy_instantiate_(long, unsigned short)
hamr_hip_copy_instantiate_(long, unsigned int)
hamr_hip_copy_instantiate_(long, unsigned long)
hamr_hip_copy_instantiate_(long, unsigned long long)

hamr_hip_copy_instantiate_(long long, float)
hamr_hip_copy_instantiate_(long long, double)
hamr_hip_copy_instantiate_(long long, char)
hamr_hip_copy_instantiate_(long long, signed char)
hamr_hip_copy_instantiate_(long long, short)
hamr_hip_copy_instantiate_(long long, int)
hamr_hip_copy_instantiate_(long long, long)
//hamr_hip_copy_instantiate_(long long, long long)
hamr_hip_copy_instantiate_(long long, unsigned char)
hamr_hip_copy_instantiate_(long long, unsigned short)
hamr_hip_copy_instantiate_(long long, unsigned int)
hamr_hip_copy_instantiate_(long long, unsigned long)
hamr_hip_copy_instantiate_(long long, unsigned long long)

hamr_hip_copy_instantiate_(unsigned char, float)
hamr_hip_copy_instantiate_(unsigned char, double)
hamr_hip_copy_instantiate_(unsigned char, char)
hamr_hip_copy_instantiate_(unsigned char, signed char)
hamr_hip_copy_instantiate_(unsigned char, short)
hamr_hip_copy_instantiate_(unsigned char, int)
hamr_hip_copy_instantiate_(unsigned char, long)
hamr_hip_copy_instantiate_(unsigned char, long long)
//hamr_hip_copy_instantiate_(unsigned char, unsigned char)
hamr_hip_copy_instantiate_(unsigned char, unsigned short)
hamr_hip_copy_instantiate_(unsigned char, unsigned int)
hamr_hip_copy_instantiate_(unsigned char, unsigned long)
hamr_hip_copy_instantiate_(unsigned char, unsigned long long)

hamr_hip_copy_instantiate_(unsigned short, float)
hamr_hip_copy_instantiate_(unsigned short, double)
hamr_hip_copy_instantiate_(unsigned short, char)
hamr_hip_copy_instantiate_(unsigned short, signed char)
hamr_hip_copy_instantiate_(unsigned short, short)
hamr_hip_copy_instantiate_(unsigned short, int)
hamr_hip_copy_instantiate_(unsigned short, long)
hamr_hip_copy_instantiate_(unsigned short, long long)
hamr_hip_copy_instantiate_(unsigned short, unsigned char)
//hamr_hip_copy_instantiate_(unsigned short, unsigned short)
hamr_hip_copy_instantiate_(unsigned short, unsigned int)
hamr_hip_copy_instantiate_(unsigned short, unsigned long)
hamr_hip_copy_instantiate_(unsigned short, unsigned long long)

hamr_hip_copy_instantiate_(unsigned int, float)
hamr_hip_copy_instantiate_(unsigned int, double)
hamr_hip_copy_instantiate_(unsigned int, char)
hamr_hip_copy_instantiate_(unsigned int, signed char)
hamr_hip_copy_instantiate_(unsigned int, short)
hamr_hip_copy_instantiate_(unsigned int, int)
hamr_hip_copy_instantiate_(unsigned int, long)
hamr_hip_copy_instantiate_(unsigned int, long long)
hamr_hip_copy_instantiate_(unsigned int, unsigned char)
hamr_hip_copy_instantiate_(unsigned int, unsigned short)
//hamr_hip_copy_instantiate_(unsigned int, unsigned int)
hamr_hip_copy_instantiate_(unsigned int, unsigned long)
hamr_hip_copy_instantiate_(unsigned int, unsigned long long)

hamr_hip_copy_instantiate_(unsigned long, float)
hamr_hip_copy_instantiate_(unsigned long, double)
hamr_hip_copy_instantiate_(unsigned long, char)
hamr_hip_copy_instantiate_(unsigned long, signed char)
hamr_hip_copy_instantiate_(unsigned long, short)
hamr_hip_copy_instantiate_(unsigned long, int)
hamr_hip_copy_instantiate_(unsigned long, long)
hamr_hip_copy_instantiate_(unsigned long, long long)
hamr_hip_copy_instantiate_(unsigned long, unsigned char)
hamr_hip_copy_instantiate_(unsigned long, unsigned short)
hamr_hip_copy_instantiate_(unsigned long, unsigned int)
//hamr_hip_copy_instantiate_(unsigned long, unsigned long)
hamr_hip_copy_instantiate_(unsigned long, unsigned long long)

hamr_hip_copy_instantiate_(unsigned long long, float)
hamr_hip_copy_instantiate_(unsigned long long, double)
hamr_hip_copy_instantiate_(unsigned long long, char)
hamr_hip_copy_instantiate_(unsigned long long, signed char)
hamr_hip_copy_instantiate_(unsigned long long, short)
hamr_hip_copy_instantiate_(unsigned long long, int)
hamr_hip_copy_instantiate_(unsigned long long, long)
hamr_hip_copy_instantiate_(unsigned long long, long long)
hamr_hip_copy_instantiate_(unsigned long long, unsigned char)
hamr_hip_copy_instantiate_(unsigned long long, unsigned short)
hamr_hip_copy_instantiate_(unsigned long long, unsigned int)
hamr_hip_copy_instantiate_(unsigned long long, unsigned long)
//hamr_hip_copy_instantiate_(unsigned long long, unsigned long long)