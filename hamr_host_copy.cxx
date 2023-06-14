#include "hamr_config.h"

#include "hamr_host_copy.h"
#include "hamr_host_copy_impl.h"

#define hamr_host_copy_instantiate_(T, U) \
template int hamr::copy_to_host_from_host<T,U>(T *dest, const U *src, size_t n_elem);

#define hamr_host_copy_instantiate(T) \
template int hamr::copy_to_host_from_host<T>(T *dest, const T *src, size_t n_elem, void *); \
hamr_host_copy_instantiate_(T, float) \
hamr_host_copy_instantiate_(T, double) \
hamr_host_copy_instantiate_(T, char) \
hamr_host_copy_instantiate_(T, signed char) \
hamr_host_copy_instantiate_(T, short) \
hamr_host_copy_instantiate_(T, int) \
hamr_host_copy_instantiate_(T, long) \
hamr_host_copy_instantiate_(T, long long) \
hamr_host_copy_instantiate_(T, unsigned char) \
hamr_host_copy_instantiate_(T, unsigned short) \
hamr_host_copy_instantiate_(T, unsigned int) \
hamr_host_copy_instantiate_(T, unsigned long) \
hamr_host_copy_instantiate_(T, unsigned long long)

hamr_host_copy_instantiate(float)
hamr_host_copy_instantiate(double)
hamr_host_copy_instantiate(char)
hamr_host_copy_instantiate(signed char)
hamr_host_copy_instantiate(short)
hamr_host_copy_instantiate(int)
hamr_host_copy_instantiate(long)
hamr_host_copy_instantiate(long long)
hamr_host_copy_instantiate(unsigned char)
hamr_host_copy_instantiate(unsigned short)
hamr_host_copy_instantiate(unsigned int)
hamr_host_copy_instantiate(unsigned long)
hamr_host_copy_instantiate(unsigned long long)
