from hamr import *
import cupy as cp

# allocate some memory on the GPU
n_elem = 16
buf = buffer_float(buffer_allocator_cuda, n_elem, 3.1415)

# convert to a cupy array
arr = cp.array(buf.get_cuda_accessible(), copy=False)

# modify the cupy array
arr *= 10000

# print the buffer, which should reflect the modification because of the
# zero-copy data sharing
print('buf = %s\n'%(str(buf)))
