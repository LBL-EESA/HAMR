template <typename T, typename U>
hamr::buffer<T> add(const hamr::buffer<T> &a1, const hamr::buffer<U> &a2)
{
    // get pointers to the input arrays that are safe to use on the GPU
    auto spa1 = a1.get_openmp_accessible();
    const T *pa1 = spa1.get();

    auto spa2 = a2.get_openmp_accessible();
    const U *pa2 = spa2.get();

    // allocate the memory for the result on the GPU, and get a pointer to it
    size_t n_vals = a1.size();
    hamr::buffer<T> ao(hamr::buffer_allocator::openmp, n_vals, T(0));
    T *pao = ao.data();

    // launch the kernel to add the arrays
    #pragma omp target teams distribute parallel for is_device_ptr(pao, pa1, pa2)
    for (size_t i = 0; i < n_vals; ++i)
    {
        pao[i] = pa1[i] + pa2[i];
    }

    return ao;
}
