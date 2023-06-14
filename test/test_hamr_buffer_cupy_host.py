from hamr import *
import cupy as cp
import sys

stderr = sys.__stderr__

stderr.write('TEST: creating a hamr::buffer host ... \n')
buf = buffer_float(buffer_allocator_malloc, 16, 3.1415)
stderr.write('buf = %s\n'%(str(buf)))
stderr.write('TEST: creating a hamr::buffer host ... OK!\n\n')

stderr.write('TEST: get a handle to the data ... \n')
h = buf.get_cuda_accessible()
stderr.write('TEST: get a handle to the data ... OK!\n\n')

stderr.write('TEST: share the data with Cupy ... \n')
arr = cp.array(h, copy=False)
stderr.write('arr.__cuda_array_interface__ = %s\n'%(arr.__cuda_array_interface__))
stderr.write('TEST: share the data with Cupy ... OK!\n\n')

stderr.write('TEST: deleting the hamr::buffer ... \n')
buf = None
stderr.write('TEST: deleting the hamr::buffer ... OK!\n\n')

stderr.write('TEST: Cupy reads the data ... \n')
stderr.write('arr = %s\n'%(str(arr)))
stderr.write('TEST: Cupy reads the data ... OK!\n\n')

stderr.write('TEST: Cupy modifies the data ... \n')
arr *= 10000
stderr.write('arr = %s\n'%(str(arr)))
stderr.write('TEST: Cupy modifies the data ... OK!\n\n')

stderr.write('TEST: deleting the Cupy array ... \n')
arr = None
stderr.write('TEST: deleting the Cupy array ... OK!\n\n')

sys.exit(0)
