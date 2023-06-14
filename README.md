### HAMR
HAMR is a library defining an accelerator technology agnostic memory model that
bridges between accelerator technologies (CUDA, HIP, ROCm, OpenMP, Kokos, etc)
and traditional CPUs in heterogeneous computing environments.  HAMR is light
weight and implemented in modern C++. HAMR includes Python integration that
enables zero-copy data transfer between C++ and Python technogies such as Numba
and Cupy.

### Citing
If you've used HAMR in your application please cite us.

[![DOI](https://zenodo.org/badge/429528113.svg)](https://zenodo.org/badge/latestdoi/429528113)

### Source Code
The source code can be obtained at the [HAMR github repository](https://github.com/LBL-EESA/HAMR).

### Documentation
The [HAMR User's Guide](https://hamr.readthedocs.io/en/latest/) documents
compiling and use of HAMR and contains simple examples.

The [HAMR Doxygen site](https://hamr.readthedocs.io/en/latest/doxygen/index.html) documents the APIs. Most users will
want to start with the [hamr::buffer](https://hamr.readthedocs.io/en/latest/doxygen/classhamr_1_1buffer.html), a
container that has capabilities similar to std::vector and can provide access
to data in different accelerator execution environments.

### Regression Testing and CI
![CPU-HAMR build and test](https://github.com/LBL-EESA/hamr/actions/workflows/build_and_test_host.yml/badge.svg)
![CUDA-HAMR build and test](https://github.com/LBL-EESA/hamr/actions/workflows/build_and_test_cuda.yml/badge.svg)
![HIP-HAMR build and test](https://github.com/LBL-EESA/hamr/actions/workflows/build_and_test_hip.yml/badge.svg)
![AMD-OpenMP-HAMR build and test](https://github.com/LBL-EESA/hamr/actions/workflows/build_and_test_amd_openmp.yml/badge.svg)

### License
HAMR's [license](LICENSE) is a BSD license with an ADDED paragraph at the end that makes it easy for us to
accept improvements. See [license](LICENSE) for more information.

## Copyright Notice
HAMR - Heterogeneous Accelerator Memory Resource (HAMR)
Copyright (c) 2022, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any
required approvals from the U.S. Dept. of Energy). All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights.  As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative
works, and perform publicly and display publicly, and to permit others to do so.
