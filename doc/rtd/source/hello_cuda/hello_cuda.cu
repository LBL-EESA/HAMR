#include "add_cuda_dispatch.h"

#include <hamr_buffer.h>
#include <iostream>
#include <memory>

using hamr::buffer;
using allocator = hamr::buffer_allocator;

int main(int, char **)
{
    size_t n_vals = 400;

    // allocate an array initialized to 1 on the GPU
    buffer<float> a0(allocator::cuda, n_vals, 1.0f);

    // allocate an array initialized to 1 on the CPU
    buffer<float> a1(allocator::malloc, n_vals, 1.0f);

    // add the two arrays on the GPU
    buffer<float> a2 = add_cuda(a0, a1);

    // print the result on the CPU
    auto spa2 = a2.get_cpu_accessible();
    float *pa2 = spa2.get();

    std::cerr << "a2 = ";
    for (int i = 0; i < a2.size(); ++i)
        std::cerr << pa2[i] << " ";
    std::cerr << std::endl;

    return 0;
}
