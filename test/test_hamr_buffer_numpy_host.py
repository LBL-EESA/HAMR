from hamr import *
import numpy as np
import sys

stderr = sys.__stderr__

n_elem = 256
init_val = 3.1415
mod_val = 10000
res_val = init_val*mod_val

# send data from C++ to Python
stderr.write('TEST 1 : C++ --> Python\n' \
             '=======================\n')

stderr.write('TEST 1: creating a hamr::buffer host ... \n')
buf = buffer_float(buffer_allocator_malloc, n_elem, init_val)
stderr.write('buf = %s\n'%(str(buf)))
stderr.write('TEST 1: creating a hamr::buffer host ... OK!\n\n')

stderr.write('TEST 1: get a handle to the data ... \n')
h = buf.get_host_accessible()
stderr.write('TEST 1: get a handle to the data ... OK!\n\n')

stderr.write('TEST 1: share the data with Numpy ... \n')
arr = np.array(h, copy=False)
stderr.write('arr.__array_interface__ = %s\n'%(arr.__array_interface__))
stderr.write('TEST 1: share the data with Numpy ... OK!\n\n')

stderr.write('TEST 1: deleting the hamr::buffer ... \n')
buf = None
h = None
stderr.write('TEST 1: deleting the hamr::buffer ... OK!\n\n')

stderr.write('TEST 1: Numpy reads the data ... \n')
stderr.write('arr = %s\n'%(str(arr)))
stderr.write('TEST 1: Numpy reads the data ... OK!\n\n')

stderr.write('TEST 1: Numpy modifies the data ... \n')
arr *= mod_val
stderr.write('arr = %s\n'%(str(arr)))
stderr.write('TEST 1: Numpy modifies the data ... OK!\n\n')

stderr.write('TEST 1: Verify the result ... \n')
if not np.allclose(arr, res_val):
    stderr.write('ERROR: TEST 1 failed!\n')
    sys.exit(-1)
stderr.write('TEST 1: Verify the result ... OK\n\n')

stderr.write('TEST 1: deleting the Numpy array ... \n')
arr = None
stderr.write('TEST 1: deleting the Numpy array ... OK!\n\n')



# send data from Python to C++
stderr.write('TEST 2 : Python --> C++\n' \
             '=======================\n')

stderr.write('TEST 2: creating a Numpy array ... \n')
arr = np.full((n_elem), init_val, dtype='float32')
stderr.write('arr.__array_interface__ = %s\n'%(arr.__array_interface__))
#stderr.write('arr = %s\n'%(str(arr)))
stderr.write('TEST 2: creating a Numpy array ... OK\n\n')

stderr.write('TEST 2: share the data with hamr::buffer ... \n')
buf = buffer(arr)
stderr.write('buf = %s\n'%(str(buf)))
stderr.write('TEST 2: share the data with hamr::buffer ... OK\n\n')

stderr.write('TEST 2: Numpy modifies the data ... \n')
arr *= mod_val
#stderr.write('arr = %s\n'%(str(arr)))
stderr.write('TEST 2: Numpy modifies the data ... OK!\n\n')

stderr.write('TEST 2: deleting the Numpy array ... \n')
arr = None
stderr.write('TEST 2: deleting the Numpy array ... OK!\n\n')

stderr.write('TEST 2: display the modified hamr::buffer ... \n')
stderr.write('buf = %s\n'%(str(buf)))
stderr.write('TEST 2: display the modified hamr::buffer ... OK\n\n')

stderr.write('TEST 2: deleting the hamr::buffer ... \n')
buf = None
stderr.write('TEST 2: deleting the hamr::buffer ... OK!\n\n')

sys.exit(0)
