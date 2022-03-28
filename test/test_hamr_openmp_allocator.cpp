#include "hamr_config.h"
#include "hamr_openmp_allocator.h"
#include "hamr_openmp_print.h"

int main(int argc, char **argv)
{
    (void) argc;
    (void) argv;

    {
    auto data = hamr::openmp_allocator<double>::allocate(400, 3.1415);

    hamr::openmp_print(data.get(), 400);
    }

    return 0;
}
