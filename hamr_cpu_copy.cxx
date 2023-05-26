#include "hamr_config.h"

#include "hamr_cpu_copy.h"
#include "hamr_cpu_copy_impl.h"

#define hamr_cpu_copy_instantiate_(T, U) \
template int hamr::copy_to_cpu_from_cpu<T,U>(T *dest, const U *src, size_t n_elem);

#define hamr_cpu_copy_instantiate(T) \
template int hamr::copy_to_cpu_from_cpu<T>(T *dest, const T *src, size_t n_elem, void *); \
hamr_cpu_copy_instantiate_(T, float) \
hamr_cpu_copy_instantiate_(T, double) \
hamr_cpu_copy_instantiate_(T, char) \
hamr_cpu_copy_instantiate_(T, signed char) \
hamr_cpu_copy_instantiate_(T, short) \
hamr_cpu_copy_instantiate_(T, int) \
hamr_cpu_copy_instantiate_(T, long) \
hamr_cpu_copy_instantiate_(T, long long) \
hamr_cpu_copy_instantiate_(T, unsigned char) \
hamr_cpu_copy_instantiate_(T, unsigned short) \
hamr_cpu_copy_instantiate_(T, unsigned int) \
hamr_cpu_copy_instantiate_(T, unsigned long) \
hamr_cpu_copy_instantiate_(T, unsigned long long)

hamr_cpu_copy_instantiate(float)
hamr_cpu_copy_instantiate(double)
hamr_cpu_copy_instantiate(char)
hamr_cpu_copy_instantiate(signed char)
hamr_cpu_copy_instantiate(short)
hamr_cpu_copy_instantiate(int)
hamr_cpu_copy_instantiate(long)
hamr_cpu_copy_instantiate(long long)
hamr_cpu_copy_instantiate(unsigned char)
hamr_cpu_copy_instantiate(unsigned short)
hamr_cpu_copy_instantiate(unsigned int)
hamr_cpu_copy_instantiate(unsigned long)
hamr_cpu_copy_instantiate(unsigned long long)
