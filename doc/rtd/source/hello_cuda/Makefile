
HAMR_SOURCE=../../../../
HAMR_BUILD=../../../../build_cuda

all:
	nvcc hello_cuda.cu -I${HAMR_SOURCE} -I${HAMR_BUILD} -std=c++14 -L${HAMR_BUILD}/lib/ -lhamr
