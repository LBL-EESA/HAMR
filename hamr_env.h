#ifndef hamr_env_h
#define hamr_env_h

#include "hamr_config.h"

/// heterogeneous accelerator memory resource
namespace hamr
{

/// returns the value of the HAMR_VERBOSE environment variable
#if defined(HAMR_VERBOSE)
HAMR_EXPORT int get_verbose();
#else
constexpr HAMR_EXPORT int get_verbose() { return 0; }
#endif

}

#endif
