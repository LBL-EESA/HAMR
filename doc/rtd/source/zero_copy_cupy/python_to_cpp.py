from hamr import *
import cupy as cp

# create a cupy array on the GPU
n_elem = 16
arr = cp.full((n_elem), 3.1415, dtype='float32')

# zero-copy share the data with C++
buf = buffer(arr)

# modify the cupy array
arr *= 10000

# print the buffer, which should reflect the modification because of the
# zero-copy data sharing
print('buf = %s\n'%(str(buf)))
