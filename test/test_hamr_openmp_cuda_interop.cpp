#include "hamr_config.h"
#include "hamr_buffer.h"
#include "hamr_buffer_allocator.h"
#include "hamr_buffer_util.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <omp.h>

#include <iostream>
#include <fstream>

using allocator = hamr::buffer_allocator;
using transfer = hamr::buffer_transfer;

void example1(size_t nElem, int devId)
{
  // OpenMP allocates this array on device memory
  omp_set_default_device(devId);
  double *devPtr = (double*)malloc(nElem*sizeof(double));
  #pragma omp target enter data map(alloc: devPtr[0:nElem])

  // OpenMP initializes the memory on the device
  #pragma omp target teams distribute \
    parallel for map(alloc: devPtr[0:nElem])
  for (size_t i = 0; i < nElem; ++i)
    devPtr[i] = -3.14;

  // zero-copy construct with the device pointer
  hamr::buffer<double> *simData;
  #pragma omp target data use_device_addr(devPtr)
  {
  simData = new hamr::buffer<double>(allocator::openmp, hamr::stream(),
                                     nElem, devId, devPtr, 0);
  }

  // do something with the buffer
  simData->print();

  // delete the array
  delete simData;

  // now it is safe to deallocate the device memory
  #pragma omp target exit data map(release: devPtr[0:nElem])
}


void example2(size_t nElem, int devId)
{
  // allocate device memory
  omp_set_default_device(devId);
  double *devPtr = (double*)omp_target_alloc(nElem*sizeof(double), devId);

  // wrap it in a shared pointer so it is eventually deallocated
  std::shared_ptr<double> spDev(devPtr,
    [devId](double *ptr){ omp_target_free(ptr, devId); });

  // initialize the array on the device
  #pragma omp target teams distribute parallel for is_device_ptr(devPtr)
  for (size_t i = 0; i < nElem; ++i)
    devPtr[i] = -3.14;

  // zero-copy construct with coordinated life cycle management
  auto simData = hamr::buffer<double>(allocator::openmp, hamr::stream(),
                                      nElem, devId, spDev);

  // do something with the buffer
  simData.print();
}

void example3(size_t nElem, int srcDev, int destDev)
{
  // allocate device memory
  omp_set_default_device(srcDev);
  double *devPtr = (double*)omp_target_alloc(nElem*sizeof(double), srcDev);

  // initialize
  #pragma omp target teams distribute parallel for is_device_ptr(devPtr)
  for (size_t i = 0; i < nElem; ++i)
    devPtr[i] = -3.14;

  // zero-copy construct from a device pointer, and take ownership
  auto simData = hamr::buffer<double>(allocator::openmp, hamr::stream(),
                                          nElem, srcDev, devPtr, 1);

  // move to destDev in place
  omp_set_default_device(destDev);
  simData.move(allocator::openmp);

  // do something with the buffer
  simData.print();
}

void example4(size_t nElem, int srcDev, int destDev)
{
  // allocate and value initialize on one device using OpenMP
  omp_set_default_device(srcDev);
  auto simData = hamr::buffer<double>(allocator::openmp, hamr::stream(),
                                          nElem, -3.14);

  // deep-copy to another device using CUDA
  cudaSetDevice(destDev);
  auto dataCpy = hamr::buffer<double>(allocator::openmp, simData);

  // make sure movement is complete before using
  dataCpy.synchronize();

  // do something with the data
  dataCpy.print();
}



hamr::buffer<double>
add_arrays_mp(int dev, hamr::buffer<double> &a1, hamr::buffer<double> &a2)
{
  // get a view of the incoming data on the device we will use
  omp_set_default_device(dev);

#if defined(STRUCTURED_BINDING)
  auto [spa1, pa1] = hamr::get_openmp_accessible(a1);
  auto [spa2, pa2] = hamr::get_openmp_accessible(a2);
#else
  auto spa1 = a1.get_openmp_accessible();
  auto pa1 = spa1.get();

  auto spa2 = a2.get_openmp_accessible();
  auto pa2 = spa2.get();
#endif

  // allocate space for the result
  size_t nElem = a1.size();

  auto a3 = hamr::buffer<double>(allocator::openmp, nElem);

  // direct access to the result since we know it is in place
  auto pa3 = a3.data();

  // do the calculation
  #pragma omp target teams distribute parallel for is_device_ptr(pa1, pa2)
  for (size_t i = 0; i < nElem; ++i)
    pa3[i] = pa2[i] + pa1[i];

  return a3;
}

void example5(size_t nElem, int dev1, int dev2)
{
  // this data is located in host main memory
  auto a1 = hamr::buffer<double>(allocator::malloc, nElem, 1.0);

  // this data is located in device 1 main memory
  omp_set_default_device(dev1);
  auto a2 = hamr::buffer<double>(allocator::openmp, hamr::stream(),
                                 transfer::async, nElem, 2.0);

  // do the calculation on device 2
  auto a3 = add_arrays_mp(dev2, a1, a2);

  // do something with the result
  a3.print();
}


namespace libA {
__global__
void add(double *a3, const double *a1, const double *a2, size_t n)
{
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  if (i >= n) return;
  a3[i] = a1[i] + a2[i];
  //printf("a3[%d]=%g  a1[%d]=%g  a2[%d]=%g \n", i, a3[i], i, a1[i], i, a2[i]);
}

hamr::buffer<double>
Add(int dev, const hamr::buffer<double> &a1, const hamr::buffer<double> &a2)
{
  // use this stream for the calculation
  cudaStream_t strm = hamr::stream();

  // get a view of the incoming data on the device we will use
  cudaSetDevice(dev);

#if defined(STRUCTURED_BINDING)
  auto [spa1, pa1] = hamr::get_cuda_accessible(a1);
  auto [spa2, pa2] = hamr::get_cuda_accessible(a2);
#else
  auto spa1 = a1.get_openmp_accessible();
  auto pa1 = spa1.get();

  auto spa2 = a2.get_openmp_accessible();
  auto pa2 = spa2.get();
#endif

  // allocate space for the result
  size_t nElem = a1.size();

  auto a3 = hamr::buffer<double>(allocator::cuda_async, strm, transfer::async, nElem);

  // direct access to the result since we know it is in place
  auto pa3 = a3.data();

  // make sure the data in flight, if it was moved, has arrived
  a1.synchronize();
  a2.synchronize();

  // do the calculation
  int threads = 128;
  int blocks = nElem / threads + ( nElem % threads ? 1 : 0 );
  add<<<blocks,threads,0,strm>>>(pa3, pa1, pa2, nElem);

  return a3;
}
}

namespace libB {
void Write(std::ofstream &ofs, hamr::buffer<double> &a)
{
  // get a view of the data on the host
  auto [spA, pA] = hamr::get_host_accessible(a);

  // make sure the data if moved has arrived
  a.synchronize();

  // send the data to the file
  size_t nElem = a.size();
  for (size_t i = 0; i < nElem; ++i)
    ofs << pA[i] << " ";
  ofs << std::endl;
}
}

void example6(size_t nElem, int dev1, int dev2)
{
  // this data is located in host memory, initialized to 1
  auto a1 = hamr::buffer<double>(allocator::malloc, hamr::stream(),
                                 transfer::async, nElem, 1.0);

  // this data is located in device 1 memory, unitialized
  omp_set_default_device(dev1);
  auto a2 = hamr::buffer<double>(allocator::openmp, hamr::stream(),
                                 transfer::async, nElem, 1.0);

  // initialize with OpenMP target offload
  auto pA2 = a2.data();

  #pragma omp target teams distribute parallel for is_device_ptr(pA2)
  for (size_t i = 0; i < nElem; ++i)
    pA2[i] = 2.0;

  // pass data to libA for the calculations
  auto a3 = libA::Add(dev2, a1, a2);

  // pass data to libB for I/O
  auto ofile = std::ofstream("data.txt");
  libB::Write(ofile, a1);
  libB::Write(ofile, a2);
  libB::Write(ofile, a3);
  ofile.close();
}





int main(int argc, char **argv)
{
  if (argc != 3)
  {
      std::cerr << "usage: test_openmp_cuda_interop [device id] [device id]" << std::endl;
      return -1;
  }

  size_t nElem = 64;
  int dev = atoi(argv[1]);
  int destDev = atoi(argv[2]);

  std::cerr << "zero-copy construct manual life cycle management ... " << std::endl;
  example1(nElem, dev);

  std::cerr << "zero-copy construct automatic life cycle management ... " << std::endl;
  example2(nElem, dev);

  std::cerr << "move in place ..." << std::endl;
  example3(nElem, dev, destDev);

  std::cerr << "deep copy construct on another device ... " << std::endl;
  example4(nElem, dev, destDev);

  std::cerr << "add two arrays OpenMP ... " << std::endl;
  example5(nElem, dev, destDev);

  std::cerr << "add two arrays OpenMP CUDA interop ... " << std::endl;
  example6(nElem, dev, destDev);

  return 0;
}
