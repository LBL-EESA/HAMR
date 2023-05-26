#include "hamr_config.h"

#include "hamr_hip_malloc_allocator.h"
#include "hamr_hip_malloc_allocator_impl.h"

template class hamr::hip_malloc_deleter<float>;
template class hamr::hip_malloc_deleter<double>;
template class hamr::hip_malloc_deleter<char>;
template class hamr::hip_malloc_deleter<signed char>;
template class hamr::hip_malloc_deleter<short>;
template class hamr::hip_malloc_deleter<int>;
template class hamr::hip_malloc_deleter<long>;
template class hamr::hip_malloc_deleter<long long>;
template class hamr::hip_malloc_deleter<unsigned char>;
template class hamr::hip_malloc_deleter<unsigned short>;
template class hamr::hip_malloc_deleter<unsigned int>;
template class hamr::hip_malloc_deleter<unsigned long>;
template class hamr::hip_malloc_deleter<unsigned long long>;

#define hamr_hip_malloc_allocator_instantiate_members(_T) \
template std::shared_ptr<_T> hamr::hip_malloc_allocator<_T>::allocate(size_t n, const float *vals, bool hipVals); \
template std::shared_ptr<_T> hamr::hip_malloc_allocator<_T>::allocate(size_t n, const double *vals, bool hipVals); \
template std::shared_ptr<_T> hamr::hip_malloc_allocator<_T>::allocate(size_t n, const char *vals, bool hipVals); \
template std::shared_ptr<_T> hamr::hip_malloc_allocator<_T>::allocate(size_t n, const signed char *vals, bool hipVals); \
template std::shared_ptr<_T> hamr::hip_malloc_allocator<_T>::allocate(size_t n, const short *vals, bool hipVals); \
template std::shared_ptr<_T> hamr::hip_malloc_allocator<_T>::allocate(size_t n, const int *vals, bool hipVals); \
template std::shared_ptr<_T> hamr::hip_malloc_allocator<_T>::allocate(size_t n, const long *vals, bool hipVals); \
template std::shared_ptr<_T> hamr::hip_malloc_allocator<_T>::allocate(size_t n, const long long *vals, bool hipVals); \
template std::shared_ptr<_T> hamr::hip_malloc_allocator<_T>::allocate(size_t n, const unsigned char *vals, bool hipVals); \
template std::shared_ptr<_T> hamr::hip_malloc_allocator<_T>::allocate(size_t n, const unsigned short *vals, bool hipVals); \
template std::shared_ptr<_T> hamr::hip_malloc_allocator<_T>::allocate(size_t n, const unsigned int *vals, bool hipVals); \
template std::shared_ptr<_T> hamr::hip_malloc_allocator<_T>::allocate(size_t n, const unsigned long *vals, bool hipVals); \
template std::shared_ptr<_T> hamr::hip_malloc_allocator<_T>::allocate(size_t n, const unsigned long long *vals, bool hipVals); \

#define hamr_hip_malloc_allocator_instantiate(_T) \
template struct hamr::hip_malloc_allocator<_T>; \
hamr_hip_malloc_allocator_instantiate_members(_T)

hamr_hip_malloc_allocator_instantiate(float)
hamr_hip_malloc_allocator_instantiate(double)
hamr_hip_malloc_allocator_instantiate(char)
hamr_hip_malloc_allocator_instantiate(signed char)
hamr_hip_malloc_allocator_instantiate(short)
hamr_hip_malloc_allocator_instantiate(int)
hamr_hip_malloc_allocator_instantiate(long)
hamr_hip_malloc_allocator_instantiate(long long)
hamr_hip_malloc_allocator_instantiate(unsigned char)
hamr_hip_malloc_allocator_instantiate(unsigned short)
hamr_hip_malloc_allocator_instantiate(unsigned int)
hamr_hip_malloc_allocator_instantiate(unsigned long)
hamr_hip_malloc_allocator_instantiate(unsigned long long)
