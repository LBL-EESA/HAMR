## HAMM -- The Heterogeneous Accelerator Memory Model

HAMM is a platform portability library defining an accelerator technology
agnostic memory model that bridges between accelerator technologies (CUDA,
ROCm, OpenMP, Kokos, etc) in heterogeneous computing environments.
HAMM is light weight and implemented in modern C++.

Unlike other platform portability libraries HAMM deals only with the memory
model and serves as a bridge for moving data between technologies at run time.
HAMM does not implemnent execution environments for accellerator technologies.
