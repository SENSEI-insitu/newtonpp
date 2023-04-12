#include <assert.h>
#include <iostream>
#include <cstdlib>
#include <algorithm>

namespace cuda
{

// **************************************************************************
template<bool bZeroPad>
inline __device__ int scanSharedIndex( int tid )
{
    if ( bZeroPad )
    {
        const int warp = tid >> 5;
        const int lane = tid & 31;
        return 49 * warp + 16 + lane;
    }
    else
    {
        return tid;
    }
}

// **************************************************************************
template<typename T, bool bZeroPad>
inline __device__ __host__ int scanSharedMemory( int numThreads )
{
    if ( bZeroPad )
    {
        const int warpcount = numThreads >> 5;
        return (49 * warpcount + 16)*sizeof(T);
    }
    else
    {
        return numThreads*sizeof(T);
    }
}

/*
 * scanWarp - bZeroPadded template parameter specifies whether to conditionally
 * add based on the lane ID.  If we can assume that sPartials[-1..-16] is 0,
 * the routine takes fewer instructions.
 */
// **************************************************************************
template<class T, bool bZeroPadded>
inline __device__ T scanWarp( volatile T *sPartials )
{
    T t = sPartials[0];
    if ( bZeroPadded )
    {
        t += sPartials[- 1]; sPartials[0] = t;
        t += sPartials[- 2]; sPartials[0] = t;
        t += sPartials[- 4]; sPartials[0] = t;
        t += sPartials[- 8]; sPartials[0] = t;
        t += sPartials[-16]; sPartials[0] = t;
    }
    else
    {
        const int tid = threadIdx.x;
        const int lane = tid & 31;
        if ( lane >=  1 ) { t += sPartials[- 1]; sPartials[0] = t; }
        if ( lane >=  2 ) { t += sPartials[- 2]; sPartials[0] = t; }
        if ( lane >=  4 ) { t += sPartials[- 4]; sPartials[0] = t; }
        if ( lane >=  8 ) { t += sPartials[- 8]; sPartials[0] = t; }
        if ( lane >= 16 ) { t += sPartials[-16]; sPartials[0] = t; }
    }
    return t;
}

// **************************************************************************
template<class T, bool bZeroPadded>
inline __device__ T scanBlock( volatile T *sPartials )
{
    extern __shared__ T warpPartials[];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warpid = tid >> 5;

    // Compute this thread's partial sum
    T sum = scanWarp<T,bZeroPadded>( sPartials );

    __syncthreads();

    // Write each warp's reduction to shared memory
    if ( lane == 31 )
    {
        warpPartials[16+warpid] = sum;
    }

    __syncthreads();

    // Have one warp scan reductions
    if ( warpid == 0 )
    {
        scanWarp<T,bZeroPadded>( 16 + warpPartials + tid );
    }
    __syncthreads();

    // Fan out the exclusive scan element (obtained by the conditional and the
    // decrement by 1) to this warp's pending output
    if ( warpid > 0 )
    {
        sum += warpPartials[16+warpid-1];
    }

    __syncthreads();

    // Write this thread's scan output
    *sPartials = sum;

    __syncthreads();

    // The return value will only be used by caller if it contains the spine
    // value (i.e. the reduction of the array we just scanned).
    return sum;
}

// **************************************************************************
template<class T, int numThreads>
__device__ void reduceBlock( T *globalSum, volatile T *shared_sum )
{
    if (numThreads >= 1024)
    {
        if (threadIdx.x < 512)
        {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + 512];
        }
        __syncthreads();
    }

    if (numThreads >= 512)
    {
        if (threadIdx.x < 256)
        {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + 256];
        }
        __syncthreads();
    }

    if (numThreads >= 256)
    {
        if (threadIdx.x < 128)
        {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + 128];
        }
        __syncthreads();
    }

    if (numThreads >= 128)
    {
        if (threadIdx.x <  64)
        {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x +  64];
        }
        __syncthreads();
    }

    // warp synchronous at the end
    if ( threadIdx.x < 32 )
    {
        volatile int *wsSum = shared_sum;
        if (numThreads >=  64) { wsSum[threadIdx.x] += wsSum[threadIdx.x + 32]; }
        if (numThreads >=  32) { wsSum[threadIdx.x] += wsSum[threadIdx.x + 16]; }
        if (numThreads >=  16) { wsSum[threadIdx.x] += wsSum[threadIdx.x +  8]; }
        if (numThreads >=   8) { wsSum[threadIdx.x] += wsSum[threadIdx.x +  4]; }
        if (numThreads >=   4) { wsSum[threadIdx.x] += wsSum[threadIdx.x +  2]; }
        if (numThreads >=   2) { wsSum[threadIdx.x] += wsSum[threadIdx.x +  1]; }
        if ( threadIdx.x == 0 ) *globalSum = wsSum[0];
    }
}

// **************************************************************************
template<class T, int numThreads>
__device__ void predicateReduceSubarray(
    int *gPartials,
    const int *mask, ///< set to 1 if the index should be copied to the output
    size_t iBlock,
    size_t N,
    int elementsPerPartial )
{
    extern volatile __shared__ int sPartials[];

    size_t baseIndex = iBlock * elementsPerPartial;

    int sum = 0;
    for ( int i = threadIdx.x; i < elementsPerPartial; i += blockDim.x )
    {
        size_t index = baseIndex + i;
        if ( index < N )
        {
            sum += mask[index] ;
        }
    }

    sPartials[threadIdx.x] = sum;

    __syncthreads();

    reduceBlock<int,numThreads>( &gPartials[iBlock], sPartials );
}

/** Compute the reductions of each subarray of size elementsPerPartial, and
 * write them to gPartials.
 */
// **************************************************************************
template<class T, int numThreads>
__global__ void
predicateReduceSubarrays( int *gPartials, const int *mask, size_t N, int elementsPerPartial )
{
    extern volatile __shared__ int sPartials[];

    for ( int iBlock = blockIdx.x;
          iBlock*elementsPerPartial < N;
          iBlock += gridDim.x )
    {
        predicateReduceSubarray<T,numThreads>( gPartials, mask, iBlock, N, elementsPerPartial );
    }
}

// **************************************************************************
template<class T>
void predicateReduceSubarrays( int *gPartials, const int *mask, size_t N, int numPartials, int cBlocks, int cThreads )
{
    switch ( cThreads )
    {
        case 128: return predicateReduceSubarrays<T,128><<<cBlocks, 128, 128*sizeof(T)>>>( gPartials, mask, N, numPartials );
        case 256: return predicateReduceSubarrays<T,256><<<cBlocks, 256, 256*sizeof(T)>>>( gPartials, mask, N, numPartials );
        case 512: return predicateReduceSubarrays<T,512><<<cBlocks, 512, 512*sizeof(T)>>>( gPartials, mask, N, numPartials );
        case 1024: return predicateReduceSubarrays<T,1024><<<cBlocks, 1024, 1024*sizeof(T)>>>( gPartials, mask, N, numPartials );
    }

    std::cerr << "ERROR: unexpected number of threads " << cThreads << std::endl;
    abort();
}

// **************************************************************************
template<class T, bool bZeroPad>
__global__
void predicateScan_kernel( T *out, const T *in, size_t N, size_t elementsPerPartial )
{
    extern volatile __shared__ T sPartials[];
    int sIndex = scanSharedIndex<bZeroPad>( threadIdx.x );

    if ( bZeroPad )
    {
        sPartials[sIndex - 16] = 0;
    }

    T base_sum = 0;
    for ( size_t i = 0; i < elementsPerPartial; i += blockDim.x )
    {
        size_t index = blockIdx.x*elementsPerPartial + i + threadIdx.x;
        sPartials[sIndex] = (index < N) ? in[index] : 0;

        __syncthreads();

        scanBlock<T,bZeroPad>( sPartials + sIndex );

        __syncthreads();

        if ( index < N )
        {
            out[index] = sPartials[sIndex] + base_sum;
        }

        __syncthreads();

        // carry forward from this block to the next.
        size_t pidx = scanSharedIndex<bZeroPad>( blockDim.x - 1 );
        base_sum += sPartials[ pidx ];

        __syncthreads();
    }
}

// **************************************************************************
template<class T, bool bZeroPad>
__global__ void
streamCompact_kernel(
    T *pdo_m, T *pdo_x, T *pdo_y, T *pdo_z,
    T *pdo_u, T *pdo_v, T *pdo_w, T *pfo_u, T *pfo_v, T *pfo_w,
    int *outCount, const int *gBaseSums,
    const T *pdi_m, const T *pdi_x, const T *pdi_y, const T *pdi_z,
    const T *pdi_u, const T *pdi_v, const T *pdi_w,
    const T *pfi_u, const T *pfi_v, const T *pfi_w,
    const int *mask,
    size_t N, size_t elementsPerPartial )
{
    extern volatile __shared__ int sPartials[];

    int sIndex = scanSharedIndex<bZeroPad>( threadIdx.x );

    if ( bZeroPad )
    {
        sPartials[sIndex - 16] = 0;
    }

    // exclusive scan element gBaseSums[blockIdx.x]
    int base_sum = 0;
    if ( blockIdx.x && gBaseSums )
    {
        base_sum = gBaseSums[blockIdx.x-1];
    }

    for ( size_t i = 0; i < elementsPerPartial; i += blockDim.x )
    {
        size_t index = blockIdx.x*elementsPerPartial + i + threadIdx.x;

        int mask_val = (index < N ? mask[index] : 0);
        sPartials[sIndex] = mask_val;

        __syncthreads();

        scanBlock<int,bZeroPad>( sPartials + sIndex );

        __syncthreads();

        if ( mask_val )
        {
            int outIndex = base_sum;

            if ( threadIdx.x )
            {
                int pidx = scanSharedIndex<bZeroPad>( threadIdx.x - 1 );
                outIndex += sPartials[pidx];
            }

            // copy values
            pdo_m[outIndex] = pdi_m[index];
            pdo_x[outIndex] = pdi_x[index];
            pdo_y[outIndex] = pdi_y[index];
            pdo_z[outIndex] = pdi_z[index];
            pdo_u[outIndex] = pdi_u[index];
            pdo_v[outIndex] = pdi_v[index];
            pdo_w[outIndex] = pdi_w[index];
            pfo_u[outIndex] = pfi_u[index];
            pfo_v[outIndex] = pfi_v[index];
            pfo_w[outIndex] = pfi_w[index];
        }

        __syncthreads();

        // carry forward from this block to the next.
        int pidx = scanSharedIndex<bZeroPad>( blockDim.x - 1 );
        base_sum += sPartials[ pidx ];

        __syncthreads();
    }

    if ( (threadIdx.x == 0) && (blockIdx.x == 0) )
    {
        if ( gBaseSums )
        {
            *outCount = gBaseSums[ gridDim.x - 1 ];
        }
        else
        {
            int pidx = scanSharedIndex<bZeroPad>( blockDim.x - 1 );
            *outCount = sPartials[ pidx ];
        }
    }
}



#define MAX_PARTIALS 300
__device__ int g_globalPartials[MAX_PARTIALS];


/** streamCompact
 *
 * This sample illustrates how to scan predicates, with an example predicate of
 * testing integers and emitting values that are odd.
 *
 * The algorithm is implemented using the 2-pass scan algorithm, counting the
 * true predicates with a reduction pass; scanning the predicates; then passing
 * over the data again, evaluating the predicates again and using the scanned
 * predicate values as indices to write the output for which the predicate is
 * true.
 *
 * @param[out] pdo_X/pfo_X compacted arrays (device)
 * @param[out] outCount the length of the compacted array (host)
 * @param[in] pdi_X/pfi_X arrays of size N to be compacted (device)
 * @param[in] mask flags that are set to 1 where the input should be copied and zero elsewhere (device)
 * @param[in] nIn the number of arrays to compact (host)
 * @param[in] N the length of the arrays to be compacted (host)
 * @param[in] b the number of threads per block (host)
 * @returns zero if successful
 */
template<class T, bool bZeroPad>
int streamCompact(
    T *pdo_m, T *pdo_x, T *pdo_y, T *pdo_z,
    T *pdo_u, T *pdo_v, T *pdo_w,
    T *pfo_u, T *pfo_v, T *pfo_w,
    int &nOut,
    const T *pdi_m, const T *pdi_x, const T *pdi_y, const T *pdi_z,
    const T *pdi_u, const T *pdi_v, const T *pdi_w,
    const T *pfi_u, const T *pfi_v, const T *pfi_w,
    const int *mask, size_t N, int b )
{
    cudaError_t ierr = cudaSuccess;

    // get a location to return the size of the compacted array
    int *outCount;
    if (((ierr = cudaMalloc(&outCount, sizeof(int))) != cudaSuccess) ||
        ((ierr = cudaMemset(outCount, 0, sizeof(int))) != cudaSuccess))
    {
        std::cerr << "Failed to allocate memory for count" << std::endl;
        return -1;
    }

    // estimate the amount of shared memeory that we need (depends on zero padding)
    int sBytes = scanSharedMemory<int,bZeroPad>( b );

    if ( N <= b )
    {
        // compact on one block with b threads
        streamCompact_kernel<T, bZeroPad><<<1,b,sBytes>>>(
            pdo_m, pdo_x, pdo_y, pdo_z,
            pdo_u, pdo_v, pdo_w,
            pfo_u, pfo_v, pfo_w,
            outCount, 0,
            pdi_m, pdi_x, pdi_y, pdi_z,
            pdi_u, pdi_v, pdi_w,
            pfi_u, pfi_v, pfi_w,
            mask, N, N );

        // fetch the length of the compacted arrays
        if ((ierr = cudaMemcpy(&nOut, outCount, sizeof(int), cudaMemcpyDeviceToHost)) != cudaSuccess)
        {
            std::cerr << "Failed to fetch the output length from the device" << std::endl;
            cudaFree(outCount);
            return -1;
        }

        return 0;
    }

    int *gPartials = 0;
    if ((ierr = cudaGetSymbolAddress(
        (void **) &gPartials, g_globalPartials )) != cudaSuccess)
    {
        std::cerr << "Failed to get the address of the partials array. "
            << cudaGetErrorString(ierr)  << std::endl;
        cudaFree(outCount);
        return -1;
    }

    // ceil(N/b) = number of partial sums to compute
    size_t numPartials = (N + b - 1) / b;

    if ( numPartials > MAX_PARTIALS )
    {
        numPartials = MAX_PARTIALS;
    }

    // elementsPerPartial has to be a multiple of b
    unsigned int elementsPerPartial = (N + numPartials - 1) / numPartials;
    elementsPerPartial = b * ((elementsPerPartial + b - 1) / b);
    numPartials = (N + elementsPerPartial - 1) / elementsPerPartial;

    // number of CUDA threadblocks to use.  The kernels are blocking agnostic,
    // so we can clamp to any number within CUDA's limits and the code will
    // work.
    size_t maxBlocks = MAX_PARTIALS;
    unsigned int numBlocks = std::min( numPartials, maxBlocks );

    // compute block local exclusive prefix scan
    predicateReduceSubarrays<T>( gPartials, mask, N, elementsPerPartial, numBlocks, b );

    // scan across blocks
    predicateScan_kernel<int, bZeroPad><<<1,b,sBytes>>>( gPartials, gPartials, numPartials, numPartials);

    // copy the flagged data
    streamCompact_kernel<T, bZeroPad><<<numBlocks,b,sBytes>>>(
        pdo_m, pdo_x, pdo_y, pdo_z,
        pdo_u, pdo_v, pdo_w,
        pfo_u, pfo_v, pfo_w,
        outCount, gPartials,
        pdi_m, pdi_x, pdi_y, pdi_z,
        pdi_u, pdi_v, pdi_w,
        pfi_u, pfi_v, pfi_w,
        mask, N, elementsPerPartial );

    // fetch the length of the compacted arrays
    if ((ierr = cudaMemcpy(&nOut, outCount, sizeof(int), cudaMemcpyDeviceToHost)) != cudaSuccess)
    {
        std::cerr << "Failed to fetch the output length from the device" << std::endl;
        cudaFree(outCount);
        return -1;
    }

    cudaFree(outCount);

    return 0;
}

// **************************************************************************
int stream_compact(
    double *pdo_m,
    double *pdo_x, double *pdo_y, double *pdo_z,
    double *pdo_u, double *pdo_v, double *pdo_w,
    double *pfo_u, double *pfo_v, double *pfo_w,
    int &outCount,
    const double *pdi_m,
    const double *pdi_x, const double *pdi_y, const double *pdi_z,
    const double *pdi_u, const double *pdi_v, const double *pdi_w,
    const double *pfi_u, const double *pfi_v, const double *pfi_w,
    const int *mask, size_t N, int b )
{
    return streamCompact<double,true>(
        pdo_m, pdo_x, pdo_y, pdo_z,
        pdo_u, pdo_v, pdo_w,
        pfo_u, pfo_v, pfo_w,
        outCount,
        pdi_m, pdi_x, pdi_y, pdi_z,
        pdi_u, pdi_v, pdi_w,
        pfi_u, pfi_v, pfi_w,
        mask, N, b);
}
}
