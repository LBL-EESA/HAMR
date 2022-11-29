template <typename T, typename U>
hamr::buffer<T> add(const hamr::buffer<T> &a1, const hamr::buffer<U> &a2)
{
    size_t n_vals = a1.size();

    // get pointers to the input arrays that are safe to use on the GPU
    auto [spa1, pa1] = hamr::get_openmp_accessible(a1);
    auto [spa2, pa2] = hamr::get_openmp_accessible(a2);

    // allocate the memory for the result on the GPU, and get a pointer to it
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
