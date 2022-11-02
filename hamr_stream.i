%{
#include "hamr_config.h"
#include "hamr_stream.h"
%}

/***************************************************************************
 * stream
 **************************************************************************/
%namewarn("") "print";
%ignore hamr::stream::operator=;
#if defined(HAMR_ENABLE_CUDA)
%ignore hamr::stream::operator cudaStream_t;
#endif
#if defined(HAMR_ENABLE_HIP)
%ignore hamr::stream::operator hipStream_t;
#endif
%ignore hamr::stream::stream(stream &&);
%include "hamr_stream.h"
