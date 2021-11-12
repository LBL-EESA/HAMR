#ifndef hamr_env_h
#define hamr_env_h

#include "hamr_config.h"

/// heterogeneous accelerator memory resource
namespace hamr
{

/// returns the value of the HAMR_VERBOSE environment variable
HAMR_EXPORT int get_verbose();

}

#endif
