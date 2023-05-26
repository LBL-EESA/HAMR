#include "hamr_config.h"

#include "hamr_cuda_malloc_async_allocator.h"
#include "hamr_cuda_malloc_async_allocator_impl.h"

template class hamr::cuda_malloc_async_deleter<float>;
template class hamr::cuda_malloc_async_deleter<double>;
template class hamr::cuda_malloc_async_deleter<char>;
template class hamr::cuda_malloc_async_deleter<signed char>;
template class hamr::cuda_malloc_async_deleter<short>;
template class hamr::cuda_malloc_async_deleter<int>;
template class hamr::cuda_malloc_async_deleter<long>;
template class hamr::cuda_malloc_async_deleter<long long>;
template class hamr::cuda_malloc_async_deleter<unsigned char>;
template class hamr::cuda_malloc_async_deleter<unsigned short>;
template class hamr::cuda_malloc_async_deleter<unsigned int>;
template class hamr::cuda_malloc_async_deleter<unsigned long>;
template class hamr::cuda_malloc_async_deleter<unsigned long long>;

#define hamr_cuda_malloc_async_allocator_instantiate_members(_T) \
template std::shared_ptr<_T> hamr::cuda_malloc_async_allocator<_T>::allocate(cudaStream_t strm, size_t n, const float *vals, bool cudaVals); \
template std::shared_ptr<_T> hamr::cuda_malloc_async_allocator<_T>::allocate(cudaStream_t strm, size_t n, const double *vals, bool cudaVals); \
template std::shared_ptr<_T> hamr::cuda_malloc_async_allocator<_T>::allocate(cudaStream_t strm, size_t n, const char *vals, bool cudaVals); \
template std::shared_ptr<_T> hamr::cuda_malloc_async_allocator<_T>::allocate(cudaStream_t strm, size_t n, const signed char *vals, bool cudaVals); \
template std::shared_ptr<_T> hamr::cuda_malloc_async_allocator<_T>::allocate(cudaStream_t strm, size_t n, const short *vals, bool cudaVals); \
template std::shared_ptr<_T> hamr::cuda_malloc_async_allocator<_T>::allocate(cudaStream_t strm, size_t n, const int *vals, bool cudaVals); \
template std::shared_ptr<_T> hamr::cuda_malloc_async_allocator<_T>::allocate(cudaStream_t strm, size_t n, const long *vals, bool cudaVals); \
template std::shared_ptr<_T> hamr::cuda_malloc_async_allocator<_T>::allocate(cudaStream_t strm, size_t n, const long long *vals, bool cudaVals); \
template std::shared_ptr<_T> hamr::cuda_malloc_async_allocator<_T>::allocate(cudaStream_t strm, size_t n, const unsigned char *vals, bool cudaVals); \
template std::shared_ptr<_T> hamr::cuda_malloc_async_allocator<_T>::allocate(cudaStream_t strm, size_t n, const unsigned short *vals, bool cudaVals); \
template std::shared_ptr<_T> hamr::cuda_malloc_async_allocator<_T>::allocate(cudaStream_t strm, size_t n, const unsigned int *vals, bool cudaVals); \
template std::shared_ptr<_T> hamr::cuda_malloc_async_allocator<_T>::allocate(cudaStream_t strm, size_t n, const unsigned long *vals, bool cudaVals); \
template std::shared_ptr<_T> hamr::cuda_malloc_async_allocator<_T>::allocate(cudaStream_t strm, size_t n, const unsigned long long *vals, bool cudaVals);

#define hamr_cuda_malloc_async_allocator_instantiate(_T) \
template struct hamr::cuda_malloc_async_allocator<_T>; \
hamr_cuda_malloc_async_allocator_instantiate_members(_T)

hamr_cuda_malloc_async_allocator_instantiate(float)
hamr_cuda_malloc_async_allocator_instantiate(double)
hamr_cuda_malloc_async_allocator_instantiate(char)
hamr_cuda_malloc_async_allocator_instantiate(signed char)
hamr_cuda_malloc_async_allocator_instantiate(short)
hamr_cuda_malloc_async_allocator_instantiate(int)
hamr_cuda_malloc_async_allocator_instantiate(long)
hamr_cuda_malloc_async_allocator_instantiate(long long)
hamr_cuda_malloc_async_allocator_instantiate(unsigned char)
hamr_cuda_malloc_async_allocator_instantiate(unsigned short)
hamr_cuda_malloc_async_allocator_instantiate(unsigned int)
hamr_cuda_malloc_async_allocator_instantiate(unsigned long)
hamr_cuda_malloc_async_allocator_instantiate(unsigned long long)
