#include "hamr_config.h"
#include "hamr_buffer.h"
#include <iostream>

int main(int, char **argv)
{
    using T = double;

    hamr::buffer_allocator alloc;
    std::string alloc_name = argv[1];

    if (alloc_name == "malloc")
        alloc = hamr::buffer_allocator::malloc;
    else if (alloc_name == "cuda")
        alloc = hamr::buffer_allocator::cuda;
    else if (alloc_name == "cuda_async")
        alloc = hamr::buffer_allocator::cuda_async;
    else if (alloc_name == "cuda_uva")
        alloc = hamr::buffer_allocator::cuda_uva;
    else if (alloc_name == "cuda_host")
        alloc = hamr::buffer_allocator::cuda_host;
    else if (alloc_name == "hip")
        alloc = hamr::buffer_allocator::hip;
    /*else if (alloc_name == "hip_async")
        alloc = hamr::buffer_allocator::hip_async;
    else if (alloc_name == "hip_host")
        alloc = hamr::buffer_allocator::hip_host;*/
    else if (alloc_name == "open_mp")
        alloc = hamr::buffer_allocator::openmp;
    else
    {
        std::cerr << "ERROR: invlalid allocator name " << alloc_name << std::endl;
        return -1;
    }


    hamr::buffer<T> buf(alloc);

    std::cerr << "resize to 10 elements initialized to 0" << std::endl;
    buf.resize(10, 0.0);
    buf.print();

    std::cerr << "resize to 20 elements initialized to -3.1415" << std::endl;
    buf.resize(20, -3.1415);
    buf.print();

    std::cerr << "resize to 15 elements" << std::endl;
    buf.resize(15);
    buf.print();

    std::cerr << "resize to 25 elements initialized to 2.718" << std::endl;
    buf.resize(25, 2.718);
    buf.print();

    buf.synchronize();

    return 0;
}
