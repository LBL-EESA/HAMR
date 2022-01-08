from hamr import *
import numpy as np
import sys

stderr = sys.__stderr__

stderr.write('TEST: creating a hamr::buffer CPU ... \n')
buf = buffer_float(buffer_allocator_malloc, 16, 3.1415)
stderr.write('buf = %s\n'%(str(buf)))
stderr.write('TEST: creating a hamr::buffer CPU ... OK!\n\n')

stderr.write('TEST: get a handle to the data ... \n')
h = buf.get_cpu_accessible()
stderr.write('TEST: get a handle to the data ... OK!\n\n')

stderr.write('TEST: share the data with Numpy ... \n')
arr = np.array(h, copy=False)
stderr.write('arr.__array_interface__ = %s\n'%(arr.__array_interface__))
stderr.write('TEST: share the data with Numpy ... OK!\n\n')

stderr.write('TEST: deleting the hamr::buffer ... \n')
buf = None
stderr.write('TEST: deleting the hamr::buffer ... OK!\n\n')

stderr.write('TEST: Numpy reads the data ... \n')
stderr.write('arr = %s\n'%(str(arr)))
stderr.write('TEST: Numpy reads the data ... OK!\n\n')

stderr.write('TEST: Numpy modifies the data ... \n')
arr *= 10000
stderr.write('arr = %s\n'%(str(arr)))
stderr.write('TEST: Numpy modifies the data ... OK!\n\n')

stderr.write('TEST: deleting the Numpy array ... \n')
arr = None
stderr.write('TEST: deleting the Numpy array ... OK!\n\n')

sys.exit(0)
