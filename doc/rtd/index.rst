The HAMR User’s Guide
=====================
HAMR is a library defining an accelerator technology agnostic memory model that
bridges between accelerator technologies (CUDA, HIP, ROCm, OpenMP, Kokos, etc)
and traditional CPUs in heterogeneous computing environments.  HAMR is light
weight and implemented in modern C++.

Unlike other platform portability libraries HAMR deals only with the memory
model and serves as a bridge for moving data between technologies at run time.
HAMR is designed to make data easily accessible when coupling codes written for
use in different technologies.  For this reason HAMR does not implemnent an
execution environment. Instead the technology's native execution environment is
used.

When allocating or accessing data, codes declare the envirnment in which data
will be accessed. Access to the data in that environment is essentially free.
The data can then be passed to other codes which may not neccessarily be
written in the same technology. Those codes declare the environment in which
the data will be accessed, if the data is not accessibile in that environment,
it is moved.

Technology Agnostic Memory Management
-------------------------------------

hamr::buffer
~~~~~~~~~~~~
The `hamr::buffer <doxygen/classhamr_1_1buffer.html>`_ class is a container
that has capabilities similar to `std::vector` and can provide access to data
in different accelerator execution environments. Durinng construction Producers
of data declare in which environment (CUDA,ROCm, HIP, OpernMP, etc) the data
will initially be accessible in. Access to the data in the declared environment
is essentially free. When consumers of the data need to access the data, they
declare in which environment access is needed. If the consumers are accessing
in an environment in which the data is in accessibl, a temporary allocation is
created and the data is moved. Reference counting is used to manage temporary
allocations.


Online Source Code Documentation
--------------------------------
HAMR's C++ sources are documented via Doxygen at the `HAMR Doxygen site <doxygen/index.html>`_.

Examples
--------

CUDA
~~~~
This example illustrates the use of hamr moving data to and from the GPU and
CPU for use with CUDA.


.. _cuda_add_array:

.. literalinclude:: source/hello_cuda/add_cuda.h
    :language: c++
    :linenos:
    :caption: A simple CUDA kernel that adds two arrays.



.. literalinclude:: source/hello_cuda/add_cuda_dispatch.h
    :language: c++
    :linenos:
    :caption: Code that uses HAMR to access array based data in CUDA. Calling `get_cuda_accessible` makes the array's available in CUDA if they are not.  Then CUDA kernels may be applied as usual.


.. literalinclude:: source/hello_cuda/hello_cuda.cu
    :language: c++
    :linenos:
    :caption: This simple hello world style program allocates an array on the GPU and an array on the CPU, both are initialized to 1. Then dispatch code use HAMR API's to make sure that the data is accessible in CUDA before launching a simple kernel that adds the two arrays. HMAR is used to make the data accessible on the CPU and print the resulkt.

