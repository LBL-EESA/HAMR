from hamr import *
import cupy as cp
import numpy as np
import sys


def add(buf_0, buf_1):
    """ add 2 arrays on the GPU """
    arr_0 = cp.array(buf_0.get_cuda_accessible()) # share data w/ cupy on GPU
    arr_1 = cp.array(buf_1.get_cuda_accessible()) # share data w/ cupy on GPU
    arr_2 = arr_0 + arr_1                         # add on the GPU
    buf_2 = buffer(arr_2)                         # zero-copy from cupy on GPU
    return buf_2

def write(fh, buf):
    """ print the array on the host """
    arr = np.array(buf.get_host_accessible())      # share data w/ numpy on host
    fh.write('%s\n'%(str(arr)))                   # write to the file on host


n_vals = 400
buf_0 = buffer_float(buffer_allocator_cuda, n_vals, 1.0)   # allocate on the host
buf_1 = buffer_float(buffer_allocator_malloc, n_vals, 1.0) # allocate on the GPU

buf_2 = add(buf_0, buf_1)                                  # add the arrays

write(sys.stdout, buf_2)                                   # write the arrays
