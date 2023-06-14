
#include <iostream>
#include <cstdlib>
#include <cstring>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "hamr_buffer.h"
#include "hamr_cuda_launch.h"

using allocator = hamr::buffer_allocator;
using transfer = hamr::buffer_transfer;

// matrix scalar multiply
template <typename T>
 __global__
void msm(const T *A, const T b, T *C,
    unsigned long n, unsigned long m, unsigned long lda)
{
    unsigned long q = hamr::thread_id_to_array_index();

    if (!hamr::index_is_valid(q, m*lda))
        return;

    unsigned long r = q % lda;
    if (r >= n)
        return;

    C[q] = A[q] * b;
}

// matrix scalar multiply
template <typename T>
int msm(const T *A, const T b, T *C,
    unsigned long n, unsigned long m, unsigned long lda,
    cudaStream_t strm)
{
    unsigned long n_elem = m*lda;

    // get launch parameters
    int device_id = -1;
    int warps_per_block = 8;
    dim3 block_grid;
    int n_blocks = 0;
    dim3 thread_grid = 0;
    if (hamr::partition_thread_blocks(device_id, n_elem,
        warps_per_block, block_grid, n_blocks, thread_grid))
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to determine launch properties." << std::endl;
        return -1;
    }

    // invoke the kernel
    cudaError_t ierr = cudaSuccess;
    msm<<<block_grid, thread_grid, 0, strm>>>(A, b, C, n, m, lda);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        std::cerr << "[" << __FILE__ << ":" << __LINE__ << "] ERROR:"
            " Failed to launch the msm kernel. "
            << cudaGetErrorString(ierr) << std::endl;
        return -1;
    }

    return 0;
}

// *************************************************************************
void print(const char *nm, const hamr::buffer<double> &M, int n)
{
    auto spM = M.get_host_accessible();
    const double *pM = spM.get();

    std::cerr << nm << "(" << n << " x " << n << ") = " << std::endl;
    for (int j = 0; j < n; ++j) // col
    {
        for (int i = 0; i < n; ++i) // row
        {
            std::cerr << pM[j + i*n] << "  ";
        }
        std::cerr << std::endl;
    }
    std::cerr << std::endl;
}

// *************************************************************************
hamr::buffer<double> gen_A(int n)
{
    hamr::buffer<double> tmp(allocator::cuda_host, n*n);
    double *ptmp = tmp.data();

    for (int j = 0; j < n; ++j) // col
    {
        for (int i = 0; i < n; ++i) // row
        {
            ptmp[j*n + i] = i + 1;
        }
    }

    return tmp;
}

// *************************************************************************
hamr::buffer<double> gen_B(int n)
{
    hamr::buffer<double> tmp(allocator::cuda_host, n*n);
    double *ptmp = tmp.data();

    for (int j = 0; j < n; ++j) // col
    {
        for (int i = 0; i < n; ++i) // row
        {
            ptmp[j*n + i] = j + 1;
        }
    }

    return tmp;
}

// --------------------------------------------------------------------------
int gemm(cublasHandle_t h, cudaStream_t strm, transfer sync, int n,
    const hamr::buffer<double> &A, const hamr::buffer<double> &B,
    hamr::buffer<double> &C)
{
    // get A on the GPU (if it's not already there)
    auto spa = A.get_cuda_accessible();

    // get B on the GPU (if it's not already there)
    auto spb = B.get_cuda_accessible();

    // allocate space for the result on the GPU (will use memory allocated in C
    // if it was allocated on the GPU)
    hamr::buffer<double> tmp(allocator::cuda_async, strm, sync, std::move(C));

    double one = 1.0;
    double zero = 0.0;

    // do the matrix multiply
    cublasSetStream(h, strm);

    cublasStatus_t ierr = cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
        n, n, n, &one, spa.get(), n, spb.get(), n, &zero, tmp.data(), n);

    if (ierr != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "ERROR: gemm failed" << std::endl;
        return -1;
    }

    // move the result. this will be zero copy if the allocators are the same.
    C.set_stream(strm, sync);
    C = std::move(tmp);

    return 0;
}




int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cerr << "ERROR: usage" << std::endl
            << "a.out [matrix dim] [transfer mode]" << std::endl;
        return -1;
    }

    // get the matrix dimension
    int n = atoi(argv[1]);

    hamr::buffer_transfer sync = hamr::buffer_transfer::async;

    // get the transfer mode
    cudaStream_t cs1, cs2;
    bool default_stream = false;
    if (strcmp(argv[2], "default") == 0)
    {
        default_stream = true;
        cs1 = cudaStreamLegacy;
        cs2 = cudaStreamLegacy;
    }
    else if (strcmp(argv[2], "sync") == 0)
    {
        sync = transfer::sync;
        cs1 = cudaStreamLegacy;
        cs2 = cudaStreamLegacy;
    }
    else if (strcmp(argv[2], "async") == 0)
    {
        sync = transfer::async;
        cudaStreamCreate(&cs1);
        cudaStreamCreate(&cs2);
    }
    else if (strcmp(argv[2], "sync_host") == 0)
    {
        sync = transfer::sync_host;
        cs1 = cudaStreamPerThread;
        cs2 = cudaStreamPerThread;
    }
    else
    {
        std::cerr << "invalid transfer mode " << argv[2] << std::endl;
        return -1;
    }

    // create the streams
    if (!default_stream)
    {
    }

    // initialize cublas
    cublasHandle_t cbh;
    cublasStatus_t ierr = cublasCreate(&cbh);
    if (ierr != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "ERROR: failed to initialize cuBLAS" << std::endl;
        return -1;
    }

    // allocate buffers
    hamr::buffer<double> A1(allocator::cuda_host, cs1, sync);
    hamr::buffer<double> B1(allocator::cuda_host, cs1, sync);
    hamr::buffer<double> C1(allocator::cuda_async, cs1, sync, n*n, 0.0);

    hamr::buffer<double> A2(allocator::cuda_host, cs2, sync);
    hamr::buffer<double> B2(allocator::cuda_host, cs2, sync);
    hamr::buffer<double> C2(allocator::cuda_async, cs2, sync, n*n, 0.0);

    /// do C = A*B on stream 1
    A1 = gen_A(n);
    B1 = gen_B(n);

    if (gemm(cbh, cs1, sync, n, A1, B1, C1))
        return -1;

    double scale = 1./n;
    if (msm(C1.data(), scale, C1.data(), n, n, n, cs1))
        return -1;

    C1.move(allocator::cuda_host);

    /// do C = A*B on stream 2
    A2 = gen_A(n);
    B2 = gen_B(n);

    if (gemm(cbh, cs2, sync, n, A2, B2, C2))
        return -1;

    if (msm(C2.data(), scale, C2.data(), n, n, n, cs2))
        return -1;

    C2.move(allocator::cuda_host);

    if (n < 32)
    {
        print("A1", A1, n);
        print("B1", B1, n);
    }


    if (!default_stream)
        cudaStreamSynchronize(cs1);

    if (n < 32)
        print("C1", C1, n);

    if (!default_stream)
        cudaStreamSynchronize(cs2);

    if (n < 32)
        print("C2", C2, n);

    // release the memory before the streams!
    C1.free();
    C2.free();

    // finalize cuBLAS
    cublasDestroy(cbh);

    // release the streams
    if (!default_stream)
    {
        cudaStreamDestroy(cs1);
        cudaStreamDestroy(cs2);
    }

    return 0;
}
