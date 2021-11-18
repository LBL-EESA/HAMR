#include "hamr_env.h"

#if defined(HAMR_VERBOSE)

#include <cstdlib>
#include <iostream>

namespace hamr
{

// **************************************************************************
int get_verbose()
{
    static int ival = -1;

    if (ival < 0)
    {
        char *cval = getenv("HAMR_VERBOSE");
        if (cval)
        {
            ival = atoi(cval);
            std::cerr << "HAMR_VERBOSE=" << ival << std::endl;
        }
        else
        {
            ival = 0;
        }
    }

    return ival;
}

}

#endif
