### HAMR
HAMR is a library defining an accelerator technology agnostic memory model that
bridges between accelerator technologies (CUDA, HIP, ROCm, OpenMP, Kokos, etc)
and traditional CPUs in heterogeneous computing environments.  HAMR is light
weight and implemented in modern C++.

### Documentation
The [HAMR User's Guide](https://hamr.readthedocs.io/en/latest/) documents
compiling and use of HAMR and contains simple examples.

The [HAMR Doxygen site](doxygen/index.html) documents the APIs. Most users will
want to start with the [hamr::buffer](doxygen/classhamr_1_1buffer.html), a
container that has capabilities similar to std::vector and can provide access
to data in different accelerator execution environments.
