%define DOCSTR
"HAMR - Heterogeneous Accelerator Memory Resource. A library for autmated
memory management on systems with heterogeneous accellerators."
%enddef
%module(docstring=DOCSTR) hamr
%feature("autodoc", "3");
%{
#define SWIG_FILE_WITH_INIT

#include <Python.h>

#include "hamr_config.h"
#include "hamr_buffer.h"
#include "hamr_buffer_allocator.h"
#include "hamr_buffer_handle.h"
#include "hamr_python_deleter.h"
#include "hamr_stream.h"

#include <iostream>
#include <sstream>

/* disable some warnings that are present in SWIG generated code. */
#if __GNUC__ > 8
#pragma GCC diagnostic ignored "-Wcast-function-type"
#endif
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#if defined(__CUDACC__)
#pragma nv_diag_suppress = set_but_not_used
#endif
%}

/* SWIG doens't understand compiler attriibbutes */
#define __attribute__(x)

/* enable STL classes */
%include "shared_ptr.i"

/***************************************************************************
 * expose the build configuration
 **************************************************************************/
%include "hamr_config.h"

/***************************************************************************
 * buffer allocator enumerations
 **************************************************************************/
%include "hamr_buffer_allocator.h"

/***************************************************************************
 * buffer transfer mode enumerations
 **************************************************************************/
%include "hamr_buffer_transfer.i"

/***************************************************************************
 * stream
 **************************************************************************/
%include "hamr_stream.i"

/***************************************************************************
 * buffer_handle
 **************************************************************************/
%include "hamr_buffer_handle.i"

/***************************************************************************
 * buffer
 **************************************************************************/
%include "hamr_buffer.i"
