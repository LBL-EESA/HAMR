## HAMR -- The Heterogeneous Accelerator Memory Resource

HAMR is a platform portability library defining an accelerator technology
agnostic memory model that bridges between accelerator technologies (CUDA,
ROCm, OpenMP, Kokos, etc) in heterogeneous computing environments.
HAMR is light weight and implemented in modern C++.

Unlike other platform portability libraries HAMR deals only with the memory
model and serves as a bridge for moving data between technologies at run time.
HAMR does not implemnent execution environments for accellerator technologies.
