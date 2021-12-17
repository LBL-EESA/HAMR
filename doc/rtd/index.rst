The HAMR User’s Guide
=====================
HAMR is a library defining an accelerator technology agnostic memory model that
bridges between accelerator technologies (CUDA, HIP, ROCm, OpenMP, Kokos, etc)
and traditional CPUs in heterogeneous computing environments.  HAMR is light
weight and implemented in modern C++.

Source Code
-----------
Source code can be obtained at the `HAMR github repository <https://github.com/LBL-EESA/HAMR>`_.

Online Source Code Documentation
--------------------------------
HAMR's C++ sources are documented via Doxygen at the `HAMR Doxygen site <doxygen/index.html>`_.
The `hamr::buffer <doxygen/classhamr_1_1buffer.html>`_ is a container
that has capabilities similar to `std::vector` and can provide access to data
in different accelerator execution environments.

Introduction
------------

HAMR deals only with memory models and serves as a bridge for moving data
between various low and high level accelerator technologies and the CPU at run
time. HAMR is designed for coupling codes written in different technologies.
For this reason HAMR does not implement an execution environment. The
developer writes their code for the technology of their choice. The
technology's native execution environment is used to run the code.
HAMR manages memory and can be used to couple codes written in different
accelerator technologies and CPU based codes.

When allocating or accessing data, codes declare the environment (CUDA, HIP,
ROCm, OpenMP, Kokos etc) in which data will be accessed. Direct access to
device pointers in that environment is cheap. The data can then be passed to
other codes which may not necessarily be written in the same technology. Those
codes declare the environment in which the data will be accessed. If the data
is not already accessible in that environment, it is moved upon access.

Modern C++ design patterns alleviate the burden for explicit management of
temporary buffers. Lazy movement of data means data can be left in place. This
is an important feature for coupling codes, where it is not known in advance
which computational technology the consumer of the data will make use of.

Technology Agnostic Memory Management
-------------------------------------

hamr::buffer
~~~~~~~~~~~~
The `hamr::buffer <doxygen/classhamr_1_1buffer.html>`_ class is a container
that has capabilities similar to `std::vector` and can provide access to data
in different accelerator execution environments. During construction Producers
of data declare in which environment (CUDA,ROCm, HIP, OpernMP, etc) the data
will initially be accessible in. Access to the data in the declared environment
is essentially free.
When consumers of the data need to access the data, they
declare in which environment access is needed. If the consumers are accessing
in an environment in which the data is in accessible, a temporary allocation is
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
technologies. For instance a code written for the CPU could access the data as
follows:

.. code-block:: c++

    std::shared_ptr<float> spdata = data->get_cpu_accessible();
    float *pdata = spdata.get();

Because the buffer `data` was allocated for use in CUDA, here `spdata` points
to a temporary buffer that has been moved to the CPU. `pdata` is a pointer that
can be used to access the buffers contents on the CPU. Modern C++
`std::shared_ptr` is used to manage the temporary. `pdata` is valid as long as
`spdata` is in scope. In this way the consumer of the data need not know if the
data was moved or accessed in place.

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

