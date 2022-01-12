from hamr import *
import cupy as cp
import sys

stderr = sys.__stderr__

n_elem = 256
init_val = 3.1415
mod_val = 10000
res_val = init_val*mod_val

# send data from C++ to Python
stderr.write('TEST 1 : C++ --> Python\n' \
             '=======================\n')

stderr.write('TEST 1: creating a hamr::buffer w. CUDA ... \n')
buf = buffer_float(buffer_allocator_cuda, n_elem, init_val)
stderr.write('buf = %s\n'%(str(buf)))
stderr.write('TEST 1: creating a hamr::buffer w. CUDA ... OK!\n\n')

stderr.write('TEST 1: get a handle to the data ... \n')
h = buf.get_cuda_accessible()
stderr.write('TEST 1: get a handle to the data ... OK!\n\n')

stderr.write('TEST 1: share the data with Cupy ... \n')
arr = cp.array(h, copy=False)
stderr.write('arr.__cuda_array_interface__ = %s\n'%(arr.__cuda_array_interface__))
stderr.write('TEST 1: share the data with Cupy ... OK!\n\n')

stderr.write('TEST 1: deleting the hamr::buffer ... \n')
buf = None
h = None
stderr.write('TEST 1: deleting the hamr::buffer ... OK!\n\n')

stderr.write('TEST 1: Cupy modifies the data ... \n')
arr *= mod_val
stderr.write('arr = %s\n'%(str(arr)))
stderr.write('TEST 1: Cupy modifies the data ... OK!\n\n')

stderr.write('TEST 1: Verify the result ... \n')
if not cp.allclose(arr, res_val):
    stderr.write('ERROR: TEST 1 failed!\n')
    sys.exit(-1)
stderr.write('TEST 1: Verify the result ... OK\n\n')

stderr.write('TEST 1: deleting the Cupy array ... \n')
arr = None
stderr.write('TEST 1: deleting the Cupy array ... OK!\n\n')



# send data from Python to C++
stderr.write('TEST 2 : Python --> C++\n' \
             '=======================\n')

stderr.write('TEST 2: creating a Cupy array ... \n')
arr = cp.full((n_elem), init_val, dtype='float32')
stderr.write('arr.__cuda_array_interface__ = %s\n'%(arr.__cuda_array_interface__))
#stderr.write('arr = %s\n'%(str(arr)))
stderr.write('TEST 2: creating a Cupy array ... OK\n\n')

stderr.write('TEST 2: share the data with hamr::buffer ... \n')
buf = buffer(arr)
stderr.write('buf = %s\n'%(str(buf)))
stderr.write('TEST 2: share the data with hamr::buffer ... OK\n\n')

stderr.write('TEST 2: Cupy modifies the data ... \n')
arr *= mod_val
#stderr.write('arr = %s\n'%(str(arr)))
stderr.write('TEST 2: Cupy modifies the data ... OK!\n\n')

stderr.write('TEST 2: deleting the Cupy array ... \n')
arr = None
stderr.write('TEST 2: deleting the Cupy array ... OK!\n\n')

stderr.write('TEST 2: display the modified hamr::buffer ... \n')
stderr.write('buf = %s\n'%(str(buf)))
stderr.write('TEST 2: display the modified hamr::buffer ... OK\n\n')

stderr.write('TEST 2: deleting the hamr::buffer ... \n')
buf = None
stderr.write('TEST 2: deleting the hamr::buffer ... OK!\n\n')

sys.exit(0)
