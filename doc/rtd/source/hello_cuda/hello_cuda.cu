#include "add_cuda_dispatch.h"

#include <hamr_buffer.h>
#include <iostream>
#include <memory>

int main(int, char **)
{
    size_t n_vals = 400;

    // allocate an array initialized to 1 on the GPU
    auto a0 = std::make_shared<buffer<float>>(allocator::cuda, n_vals, 1.0f);

    // allocate an array initialized to 1 on the CPU
    auto a1 = std::make_shared<buffer<float>>(allocator::malloc, n_vals, 1.0f);

    // add the two arrays on the GPU
    auto a2 = add_cuda(a0, a1);

    // access the result on the CPU
    auto spa2 = a2->get_cpu_accessible();
    float *pa2 = spa2.get();

    // print the result on the CPU
    std::cerr << "a2 = ";
    for (int i = 0; i < a2->size(); ++i)
        std::cerr << pa2[i] << " ";
    std::cerr << std::endl;

    return 0;
}
