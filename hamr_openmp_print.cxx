#include "hamr_config.h"

#include "hamr_openmp_print.h"
#include "hamr_openmp_print_impl.h"

template int hamr::openmp_print<float>(float *vals, size_t n_elem);
template int hamr::openmp_print<double>(double *vals, size_t n_elem);
template int hamr::openmp_print<char>(char *vals, size_t n_elem);
template int hamr::openmp_print<signed char>(signed char *vals, size_t n_elem);
template int hamr::openmp_print<short>(short *vals, size_t n_elem);
template int hamr::openmp_print<int>(int *vals, size_t n_elem);
template int hamr::openmp_print<long>(long *vals, size_t n_elem);
template int hamr::openmp_print<long long>(long long *vals, size_t n_elem);
template int hamr::openmp_print<unsigned char>(unsigned char *vals, size_t n_elem);
template int hamr::openmp_print<unsigned short>(unsigned short *vals, size_t n_elem);
template int hamr::openmp_print<unsigned int>(unsigned int *vals, size_t n_elem);
template int hamr::openmp_print<unsigned long>(unsigned long *vals, size_t n_elem);
template int hamr::openmp_print<unsigned long long>(unsigned long long *vals, size_t n_elem);
