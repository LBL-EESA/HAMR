from hamr import *
import cupy as cp
import numpy as np


def add(buf_0, buf_1):
    """ add 2 arrays on the GPU """
    arr_0 = cp.array(buf_0.get_cuda_accessible()) # share data w/ cupy
    arr_1 = cp.array(buf_1.get_cuda_accessible()) # share data w/ cupy
    arr_2 = arr_0 + arr_1                         # add on the GPU
    buf_2 = buffer_float(arr_2)                   # zero-copy from cupy
    return buf_2

def write(buf):
    """ print the array on the CPU """
    arr = np.array(buf.get_cpu_accessible())      # share data w/ numpy
    print(arr)                                    # write to stdout



buf_0 = buffer_float(buffer_allocator_cuda, n_vals, 1.0)   # allocate on the CPU
buf_1 = buffer_float(buffer_allocator_malloc, n_vals, 1.0) # allocate on the GPU

buf_2 = add(buf_0, buf_1)                                  # add the arrays

write(buf_2)                                               # print on the CPU
