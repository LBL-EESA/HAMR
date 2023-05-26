#include "hamr_config.h"

#include "hamr_cuda_print.h"
#include "hamr_cuda_print_impl.h"

template int hamr::cuda_print<float>(const hamr::stream &strm, float *vals, size_t n_elem);
template int hamr::cuda_print<double>(const hamr::stream &strm, double *vals, size_t n_elem);
template int hamr::cuda_print<char>(const hamr::stream &strm, char *vals, size_t n_elem);
template int hamr::cuda_print<signed char>(const hamr::stream &strm, signed char *vals, size_t n_elem);
template int hamr::cuda_print<short>(const hamr::stream &strm, short *vals, size_t n_elem);
template int hamr::cuda_print<int>(const hamr::stream &strm, int *vals, size_t n_elem);
template int hamr::cuda_print<long>(const hamr::stream &strm, long *vals, size_t n_elem);
template int hamr::cuda_print<long long>(const hamr::stream &strm, long long *vals, size_t n_elem);
template int hamr::cuda_print<unsigned char>(const hamr::stream &strm, unsigned char *vals, size_t n_elem);
template int hamr::cuda_print<unsigned short>(const hamr::stream &strm, unsigned short *vals, size_t n_elem);
template int hamr::cuda_print<unsigned int>(const hamr::stream &strm, unsigned int *vals, size_t n_elem);
template int hamr::cuda_print<unsigned long>(const hamr::stream &strm, unsigned long *vals, size_t n_elem);
template int hamr::cuda_print<unsigned long long>(const hamr::stream &strm, unsigned long long *vals, size_t n_elem);
