The HAMR User’s Guide
=====================
HAMR is a library defining an accelerator technology agnostic memory model that
bridges between accelerator technologies (CUDA, HIP, ROCm, OpenMP, Sycl,
OpenCL, Kokos, etc) and traditional hosts in heterogeneous computing
environments.   HAMR is light weight and implemented in modern C++. HAMR can be
used to manage memory with in a single code or as a data model for coupling
codes in a technologically agnostic way. HAMR provides a Python module for
coupling C++ and Python codes which implements zero-copy data transfers to and
from Python using the Numpy array interface and Numba CUDA array interface
protocols.

Design
------
**Modern:** Implemented in C++20 HAMR is efficient and easy to use.
**Declarative:** Producers and consumers *delcare* where data will be accessed.
If there is a missmatch HAMR automatically moves the data.  **Lazy:** Data is
left in place until it is accessed. Zero-copy constructors enable efficient
code coupling.

Source Code
-----------
Source code can be obtained at the `HAMR github repository <https://github.com/LBL-EESA/HAMR>`_.

Documentation
-------------
HAMR's C++ sources are documented via Doxygen at the `HAMR Doxygen site <doxygen/index.html>`_.
The `hamr::buffer <doxygen/classhamr_1_1buffer.html>`_ is a container
that has capabilities similar to `std::vector` and can provide access to data
in different accelerator execution environments.

Build and Install
-----------------
HAMR is configured with CMake. The following CMake variables influence the build.

+-------------------------+----------------------------------------------------+
| CMake Variable          | Description                                        |
+-------------------------+----------------------------------------------------+
| CMAKE_BUILD_TYPE        | Release or Debug. The default is Release.          |
+-------------------------+----------------------------------------------------+
| CMAKE_CXX_FLAGS         | HAMR will set the C++ compiler flags if not set.   |
+-------------------------+----------------------------------------------------+
| CMAKE_CUDA_FLAGS        | HAMR will set the CUDA compiler flags if not set.  |
+-------------------------+----------------------------------------------------+
| HAMR_ENABLE_CUDA        | If set to ON enables CUDA features. Default OFF    |
+-------------------------+----------------------------------------------------+
| HAMR_ENABLE_HIP         | If set to ON enables HIP features. Default OFF     |
+-------------------------+----------------------------------------------------+
| HAMR_ENABLE_OPENMP      | If set to ON enables OpenMP features. Default OFF  |
+-------------------------+----------------------------------------------------+
| HAMR_ENABLE_PYTHON      | If set to ON enables Python features. Default OFF  |
+-------------------------+----------------------------------------------------+
| BUILD_TESTING           | If set to ON enables regression tests. Default OFF |
+-------------------------+----------------------------------------------------+

Note, use with HIP one must tell CMake to use `clang` and `clang++` from the ROCm
install via the `CMAKE_C_COMPILER` and `CMAKE_CXX_COMPILER` CMake options.

Introduction
------------
HAMR deals only with memory models and serves as a bridge for moving data
between various low and high level accelerator technologies and the host at run
time. HAMR is designed as a data model for coupling codes such that developers
need not code to a specific technology in order to share data.  For this reason
HAMR does not implement an execution environment. Developers write their codes
for the technology of their choice. The technology's native execution
environment is used for computation.  HAMR provides data structures that manage
memory and can be used to share data and couple codes written in different
accelerator technologies (including host based codes), by different developers,
such that the receiving code need not have knowledge of technology used to
generate that data, and the sending code need not have knowledge of the
technology that will be used to consume the data.  HAMR manages the necessary
memory movements such that the codes have access to the data in the technology
where they will use it.

When allocating or accessing data, codes declare the environment (CUDA, HIP,
ROCm, OpenMP, Sycl, OpenCL, Kokos etc) in which data will be accessed. Direct
access to device pointers in that environment is cheap. When the data is shared
with other codes which may not necessarily be written in the same technology
and the shared data is accessed, the consumer  declares the environment in
which the data will be processed. If the data is not already accessible in that
environment, it is moved upon access.

Modern C++ design patterns alleviate the burden for explicit management of
temporary buffers. Lazy movement of data means data can be left in place. This
is an important feature for coupling codes, where it is not known in advance
which computational technology the consumer of the data will make use of.

Technology Agnostic Memory Management
-------------------------------------

.. _buffer:

hamr::buffer
~~~~~~~~~~~~
The `hamr::buffer <doxygen/classhamr_1_1buffer.html>`_ class is a container
that has capabilities similar to `std::vector` and can provide access to data
in different accelerator execution environments. During construction Producers
of data declare in which environment (CUDA, ROCm, HIP, OpernMP, etc) the data
will initially be accessible in. Access to the data in the declared environment
is essentially free.
When consumers of the data need to access the data, they
declare in which environment access is needed. If the consumers are accessing
in an environment in which the data is inaccessible, a temporary allocation is
created and the data is moved. Reference counting is used to manage temporary
allocations.

For instance a code that runs in CUDA would allocate a buffer for the results
of a calculation as follows:

.. code-block:: c++

    size_t n_vals = 10000;
    hamr::buffer<float> data = buffer<float>(allocator::cuda, n_vals);

This memory is allocated on the active CUDA device. A device pointer may be
obtained as follows:

.. code-block:: c++

    std::shared_ptr<float> spdata = data->get_cuda_accessible();
    float *pdata = spdata.get();

Because the buffer `data` was allocated for use in CUDA, `spdata` points to the
buffer's contents, no data was moved. `pdata` is a device pointer that can be
passed to a CUDA kernel.

The contents of this buffer can then be passed to codes written for other
technologies. For instance a code written for the host could access the data as
follows:

.. code-block:: c++

    std::shared_ptr<float> spdata = data->get_host_accessible();
    float *pdata = spdata.get();

Because the buffer `data` was allocated for use in CUDA, here `spdata` points
to a temporary buffer that has been moved to the host. `pdata` is a pointer that
can be used to access the buffers contents on the host. Modern C++
`std::shared_ptr` is used to manage the temporary. `pdata` is valid as long as
`spdata` is in scope. In this way the consumer of the data need not know if the
data was moved or accessed in place.


.. _python_int:

Python Integration
~~~~~~~~~~~~~~~~~~
HAMR provides Python bindings that enable interoperability and data sharing
between C++ and Python codes. This is accomplished via the Numpy array
interface protocol, and the Numba CUDA array interface protocol. HAMR manages
the C++ and Python data structures such that they will persist while in use in
the other language.


C++ to Python
^^^^^^^^^^^^^
In Python one constructs `hamr::buffer` objects using the wrapped C++ API. The
passed allocator flag indicates where the memory will be allocated. For
instance an buffer of 16 values initialized to 3.1415 allocated on the GPU
would be created with

.. code-block:: python

    buf = buffer_float(buffer_allocator_cuda, 16, 3.1415)

When sharing data with Numpy or Cupy a `hamr::buffer_handle` must be used.  The
`hamr::buffer_handle` implements both the Numpy array interface and the Numba
CUDA array interface protocols enabling Numpy and Cupy to directly make use of
the shared data.

When sharing with Numpy the `hamr::buffer_handle` is obtained by
calling `hamr::buffer::get_host_accessible`.

.. code-block:: python

   arr = numpy.array(buf.get_host_accessible())

As explained above, HAMR ensures that the returned data is accessible on the
host by either moving it to a temporary if necessary, or returning a pointer to
the buffer contents.

Similarly, when sharing with Cupy the `hamr::buffer_handle` instance is
obtained by calling `hamr::buffer::get_cuda_accessible` ensuring that the
returned data is accessible in CUDA.

.. code-block:: python

   arr = cupy.array(buf.get_cuda_accessible())

The `hamr::bufer_handle` returned holds a reference to the shared data, or the
temporary, if data was automatically moved. In turn the Numpy or Cupy array
holds a reference to the `hamr::buffer_handle`, which keeps the data alive
while Numpy or Cupy uses it.

While the data is transferred to Numpy or Cupy via the array interface
protocol which is a zero-copy transfer mechanism, modifications made to the
data will only be visible in the other language if a temporary was not needed,
that is, if the data was already accessible in the location it was requested.

Python to C++
^^^^^^^^^^^^^
`hamr::buffer` instances may be zero-copy constructed directly from both Numpy
and Cupy arrays. For example

.. code-block:: python

   arr = cupy.full(16, 3.1415, dtype='float32')
   buf = buffer(arr)

The `hamr::buffer` zero-copy constructor obtains pointers to the Numpy or Cupy
array's contents via the appropriate array interface protocol. The new
`hamr::buffer` instance holds a reference to object sharing the data ensuring
that it is kept alive while in use.

This is always a zero-copy transfer and modifications in one language will be
visible in the other.


Examples
--------
The source code for the following examples is located in the `doc/rtd/source`
folder. The C++ examples include a simple Makefile that can be edited to point
to a build.

.. _hello_openmp:

Hello World! w/ C++ and OpenMP
------------------------------
This example illustrates coupling two codes, in this case functions, using HAMR
so that they can process data produced either on the host or GPU without knowing
specifically where the data passed to them resides. C++ smart pointers are used
to manage temporary buffers if the passed data needed to be moved to the device
where it was accessed.  See :ref:`buffer` for more information.  See
:ref:`hello_cuda` for a CUDA implementation of this example. See :ref:`hello_hip`
for a HIP implementation of this example.See :ref:`hello_cupy` for a Python
implementation of this example.

.. _openmp_add_array:

.. literalinclude:: source/hello_openmp/add.h
    :language: c++
    :linenos:
    :caption: Code that uses HAMR to access array based data in OpenMP. Calling `get_openmp_accessible` makes the array's available in OpenMP if they are not. Then OpenMP device oofloading may be applied as usual.


.. literalinclude:: source/hello_openmp/write.h
    :language: c++
    :linenos:
    :caption: Code that uses HAMR to access array based data on the host. Calling `get_host_accessible` makes the array available on the host if they are not.


.. literalinclude:: source/hello_openmp/hello_openmp.cpp
    :language: c++
    :linenos:
    :caption: This simple Hello world! program allocates an array on the GPU and an array on the host, both are initialized to 1. Then dispatch code use HAMR API's to make sure that the data is accessible in OpenMP before launching a simple kernel that adds the two arrays. HMAR is used to make the data accessible on the host and print the result.


.. _hello_cuda:

Hello World! w/ C++ and CUDA
------------------------------
This example illustrates coupling two codes, in this case functions, using HAMR
so that they can process data produced either on the host or GPU without knowing
specifically where the data passed to them resides. C++ smart pointers are used
to manage temporary buffers if the passed data needed to be moved to the device
where it was accessed.  See :ref:`buffer` for more information.  See
:ref:`hello_hip` for a HIP implementation of this example.  See
:ref:`hello_cupy` for a Python implementation of this example.

.. _cuda_add_array:

.. literalinclude:: source/hello_cuda/add.cuh
    :language: c++
    :linenos:
    :caption: A simple CUDA kernel that adds two arrays.


.. literalinclude:: source/hello_cuda/add.h
    :language: c++
    :linenos:
    :caption: Code that uses HAMR to access array based data in CUDA. Calling `get_cuda_accessible` makes the array's available in CUDA if they are not.  Then CUDA kernels may be applied as usual.


.. literalinclude:: source/hello_cuda/write.h
    :language: c++
    :linenos:
    :caption: Code that uses HAMR to access array based data on the host. Calling `get_host_accessible` makes the array available on the host if they are not.


.. literalinclude:: source/hello_cuda/hello_cuda.cu
    :language: c++
    :linenos:
    :caption: This simple Hello world! program allocates an array on the GPU and an array on the host, both are initialized to 1. Then dispatch code use HAMR API's to make sure that the data is accessible in CUDA before launching a simple kernel that adds the two arrays. HMAR is used to make the data accessible on the host and print the result.


.. _hello_hip:

Hello World! w/ C++ and HIP
------------------------------
This example illustrates coupling two codes, in this case functions, using HAMR so that
they can process data produced either on the host or GPU without knowing
specifically where the data passed to them resides. C++ smart pointers are used
to manage temporary buffers if the passed data needed to be moved to the device
where it was accessed.  See :ref:`buffer` for more information. See
:ref:`hello_cuda` for a CUDA implementation of this example.

.. _hip_add_array:

.. literalinclude:: source/hello_hip/add_kernel.h
    :language: c++
    :linenos:
    :caption: A simple HIP kernel that adds two arrays.


.. literalinclude:: source/hello_hip/add.h
    :language: c++
    :linenos:
    :caption: Code that uses HAMR to access array based data in HIP. Calling `get_hip_accessible` makes the array's available in HIP if they are not.  Then HIP kernels may be applied as usual.


.. literalinclude:: source/hello_hip/write.h
    :language: c++
    :linenos:
    :caption: Code that uses HAMR to access array based data on the host. Calling `get_host_accessible` makes the array available on the host if they are not.


.. literalinclude:: source/hello_hip/hello_hip.cpp
    :language: c++
    :linenos:
    :caption: This simple Hello world! program allocates an array on the GPU and an array on the host, both are initialized to 1. Then dispatch code use HAMR API's to make sure that the data is accessible in HIP before launching a simple kernel that adds the two arrays. HMAR is used to make the data accessible on the host and print the result.


.. _hello_cupy:

Hello World! w/ Python and cupy
-------------------------------
This example illustrates the coupling of two codes using hamr so that they can
process data produced either on the host or GPU without knowing specifically
where the data passed to them resides. HAMR's Python integration handle data
sharing between C++ and Python objects. See :ref:`python_int` for more
information. See :ref:`hello_cuda` for a C++ implementation of this example.


.. literalinclude:: source/hello_cupy/hello_cupy.py
    :language: python
    :linenos:
    :caption: This simple Hello world! program allocates an array on the GPU and an array on the host, both are initialized to 1. Then dispatch code use HAMR API's to make sure that the data is accessible in CUDA before launching a simple kernel that adds the two arrays. HMAR is used to make the data accessible on the host and print the result.
