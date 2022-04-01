
HAMR_SOURCE=../../../../
HAMR_BUILD=../../../../build_omp

# NVIDIA HPC Compiler
#CXX=`which nvc++`
#CXX_FLAGS=-mp=gpu -Minfo

# AMD ROCm compiler
CXX=/opt/rocm/llvm/bin/amdclang++
CXX_FLAGS=-target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx1030

all:
	${CXX} ${CXX_FLAGS} hello_openmp.cpp -I${HAMR_SOURCE} -I${HAMR_BUILD} -std=c++14 -L${HAMR_BUILD}/lib/ -lhamr
