#include "hamr_config.h"

#include "hamr_hip_print.h"
#include "hamr_hip_print_impl.h"

template int hamr::hip_print<float>(float *vals, size_t n_elem);
template int hamr::hip_print<double>(double *vals, size_t n_elem);
template int hamr::hip_print<char>(char *vals, size_t n_elem);
template int hamr::hip_print<signed char>(signed char *vals, size_t n_elem);
template int hamr::hip_print<short>(short *vals, size_t n_elem);
template int hamr::hip_print<int>(int *vals, size_t n_elem);
template int hamr::hip_print<long>(long *vals, size_t n_elem);
template int hamr::hip_print<long long>(long long *vals, size_t n_elem);
template int hamr::hip_print<unsigned char>(unsigned char *vals, size_t n_elem);
template int hamr::hip_print<unsigned short>(unsigned short *vals, size_t n_elem);
template int hamr::hip_print<unsigned int>(unsigned int *vals, size_t n_elem);
template int hamr::hip_print<unsigned long>(unsigned long *vals, size_t n_elem);
template int hamr::hip_print<unsigned long long>(unsigned long long *vals, size_t n_elem);
