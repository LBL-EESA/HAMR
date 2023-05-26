#include "hamr_config.h"

#include "hamr_new_allocator.h"
#include "hamr_new_allocator_impl.h"

template class hamr::new_deleter<float>;
template class hamr::new_deleter<double>;
template class hamr::new_deleter<char>;
template class hamr::new_deleter<signed char>;
template class hamr::new_deleter<short>;
template class hamr::new_deleter<int>;
template class hamr::new_deleter<long>;
template class hamr::new_deleter<long long>;
template class hamr::new_deleter<unsigned char>;
template class hamr::new_deleter<unsigned short>;
template class hamr::new_deleter<unsigned int>;
template class hamr::new_deleter<unsigned long>;
template class hamr::new_deleter<unsigned long long>;

#define hamr_new_allocator_instantiate_members(_T) \
template std::shared_ptr<_T> hamr::new_allocator<_T>::allocate(size_t n, const float *vals); \
template std::shared_ptr<_T> hamr::new_allocator<_T>::allocate(size_t n, const double *vals); \
template std::shared_ptr<_T> hamr::new_allocator<_T>::allocate(size_t n, const char *vals); \
template std::shared_ptr<_T> hamr::new_allocator<_T>::allocate(size_t n, const signed char *vals); \
template std::shared_ptr<_T> hamr::new_allocator<_T>::allocate(size_t n, const short *vals); \
template std::shared_ptr<_T> hamr::new_allocator<_T>::allocate(size_t n, const int *vals); \
template std::shared_ptr<_T> hamr::new_allocator<_T>::allocate(size_t n, const long *vals); \
template std::shared_ptr<_T> hamr::new_allocator<_T>::allocate(size_t n, const long long *vals); \
template std::shared_ptr<_T> hamr::new_allocator<_T>::allocate(size_t n, const unsigned char *vals); \
template std::shared_ptr<_T> hamr::new_allocator<_T>::allocate(size_t n, const unsigned short *vals); \
template std::shared_ptr<_T> hamr::new_allocator<_T>::allocate(size_t n, const unsigned int *vals); \
template std::shared_ptr<_T> hamr::new_allocator<_T>::allocate(size_t n, const unsigned long *vals); \
template std::shared_ptr<_T> hamr::new_allocator<_T>::allocate(size_t n, const unsigned long long *vals);


#define hamr_new_allocator_instantiate(_T) \
template struct hamr::new_allocator<_T>; \
hamr_new_allocator_instantiate_members(_T)

hamr_new_allocator_instantiate(float)
hamr_new_allocator_instantiate(double)
hamr_new_allocator_instantiate(char)
hamr_new_allocator_instantiate(signed char)
hamr_new_allocator_instantiate(short)
hamr_new_allocator_instantiate(int)
hamr_new_allocator_instantiate(long)
hamr_new_allocator_instantiate(long long)
hamr_new_allocator_instantiate(unsigned char)
hamr_new_allocator_instantiate(unsigned short)
hamr_new_allocator_instantiate(unsigned int)
hamr_new_allocator_instantiate(unsigned long)
hamr_new_allocator_instantiate(unsigned long long)
