%{
#include "hamr_config.h"
#include "hamr_stream.h"
%}

/***************************************************************************
 * stream
 **************************************************************************/
%namewarn("") "print";
%ignore hamr::stream::operator=;
%ignore hamr::stream::stream(stream &&);
%include "hamr_stream.h"
