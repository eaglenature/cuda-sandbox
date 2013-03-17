#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>

template<typename ErrorType>
void check(ErrorType err, const char* const func, const char* const file, const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)


// Get Last component of type T element
template <typename T> struct Last;
// Add the same offset to each component of type T element
template <typename T> struct AddOffset;
// Reduce-sum type T element components
template <typename T> struct RegistersReduce;
// ExlusiveScan type T element components
template <typename T> struct RegistersExlusiveScan;
// InclusiveScan type T element components
template <typename T> struct RegistersInclusiveScan;


template <> struct Last<int>  { __device__ __host__ inline static int get(int a) { return a; } };
template <> struct Last<int2> { __device__ __host__ inline static int get(const int2& a) { return a.y; } };
template <> struct Last<int4> { __device__ __host__ inline static int get(const int4& a) { return a.w; } };

template <> struct AddOffset<int>
{ __device__ __host__ inline static int  apply(int a, int b) { return a + b; } };
template <> struct AddOffset<int2>
{ __device__ __host__ inline static int2 apply(const int2& a, int b) { return make_int2(a.x + b, a.y + b); } };
template <> struct AddOffset<int4>
{ __device__ __host__ inline static int4 apply(const int4& a, int b) { return make_int4(a.x + b, a.y + b, a.z + b, a.w + b); } };

template <> struct RegistersReduce<int>
{ __device__ __host__ inline static int apply(int  a) { return a; } };
template <> struct RegistersReduce<int2>
{ __device__ __host__ inline static int apply(const int2& a) { return a.x + a.y; } };
template <> struct RegistersReduce<int4>
{ __device__ __host__ inline static int apply(const int4& a) { return a.x + a.y + a.z + a.w; } };

template <> struct RegistersExlusiveScan<int>
{ __device__ __host__ inline static int  apply(int a) { return 0; } };
template <> struct RegistersExlusiveScan<int2>
{ __device__ __host__ inline static int2 apply(const int2& a) { return make_int2(0, a.x); } };
template <> struct RegistersExlusiveScan<int4>
{ __device__ __host__ inline static int4 apply(const int4& a) { return make_int4(0, a.x, a.x + a.y, a.x + a.y + a.z); } };

template <> struct RegistersInclusiveScan<int>
{ __device__ __host__ inline static int  apply(int a) { return a; } };
template <> struct RegistersInclusiveScan<int2>
{ __device__ __host__ inline static int2 apply(const int2& a) { return make_int2(a.x, a.x + a.y); } };
template <> struct RegistersInclusiveScan<int4>
{ __device__ __host__ inline static int4 apply(const int4& a) { return make_int4(a.x, a.x + a.y, a.x + a.y + a.z, a.x + a.y + a.z + a.w); } };


// Tested on GTX560Ti - 8 SM cores
const int CTAs = 64;

__device__
void ReduceCTA(volatile int* smem, int CTA_SIZE)
{
    const int tid = threadIdx.x;

    if (CTA_SIZE >= 512) { if (tid < 256) { smem[tid] += smem[tid + 256]; } __syncthreads(); }
    if (CTA_SIZE >= 256) { if (tid < 128) { smem[tid] += smem[tid + 128]; } __syncthreads(); }
    if (CTA_SIZE >= 128) { if (tid <  64) { smem[tid] += smem[tid +  64]; } __syncthreads(); }

    if (tid < 32)
    {
        // warp synchronous instruction, so declare smem volatile to
        // avoid compiler mem access optimization
        volatile int* smemp = smem;
        if (CTA_SIZE >= 64) smemp[tid] += smemp[tid + 32];
        if (CTA_SIZE >= 32) smemp[tid] += smemp[tid + 16];
        if (CTA_SIZE >= 16) smemp[tid] += smemp[tid +  8];
        if (CTA_SIZE >=  8) smemp[tid] += smemp[tid +  4];
        if (CTA_SIZE >=  4) smemp[tid] += smemp[tid +  2];
        if (CTA_SIZE >=  2) smemp[tid] += smemp[tid +  1];
    }
}


#define WARP_SIZE 32
#define NUM_THREADS 256
#define NUM_WARPS (NUM_THREADS / WARP_SIZE)
#define LOG_NUM_THREADS 8
#define LOG_NUM_WARPS (LOG_NUM_THREADS - 5)
#define SCAN_STRIDE (WARP_SIZE + WARP_SIZE / 2 + 1)

//// Original version see: http://www.moderngpu.com/intro/scan.html
//__device__
//void ScanCTA(volatile int* array, volatile int* localSum, int CTA_SIZE, int TOTAL_OFFSET)
//{
//    __shared__ volatile int scan[NUM_WARPS * SCAN_STRIDE];
//
//    int tid = threadIdx.x;
//    int warp = tid / WARP_SIZE;
//    int lane = (WARP_SIZE - 1) & tid;
//
//    volatile int* s = scan + SCAN_STRIDE * warp + lane + WARP_SIZE / 2;
//    s[-16] = 0;
//
//    // Read from global memory to shared memory
//    int x = array[tid];
//    s[0] = x;
//
//    // Run inclusive scan on each warp's data
//    int sum = x;
//    #pragma unroll
//    for (int i = 0; i < 5; ++i)
//    {
//        int offset = 1 << i;
//        sum += s[-offset];
//        s[0] = sum;
//    }
//
//    // Synchronize to make all the totals available to the reduction code
//    __syncthreads();
//
//    __shared__ volatile int totals[NUM_WARPS + NUM_WARPS / 2];
//
//    if (tid < NUM_WARPS)
//    {
//        int total = scan[SCAN_STRIDE * tid + WARP_SIZE / 2 + WARP_SIZE - 1];
//
//        totals[tid] = 0;
//        volatile int* s2 = totals + NUM_WARPS / 2 + tid;
//        int totalsSum = total;
//        s2[0] = total;
//
//        #pragma unroll
//        for (int i = 0; i < LOG_NUM_WARPS; ++i)
//        {
//            int offset = 1 << i;
//            totalsSum += s2[-offset];
//            s2[0] = totalsSum;
//        }
//
//        totals[tid] = totalsSum - total;
//    }
//
//    // Synchronize to make the block scan available to all warps
//    __syncthreads();
//
//    // Add the block scan to the inclusive sum of the block
//    sum += totals[warp];
//
//    // Write the inclusive and excusive scans to global memory
//    // inclusive scan elements
//    int a = sum;
//    int b = localSum[0];
//    __syncthreads();
//
//    sum += TOTAL_OFFSET - x + b;
//    array[tid] = sum;
//
//    if (tid == blockDim.x - 1)
//    {
//        localSum[0] = a + b;
//    }
//    __syncthreads();
//}

template<typename T>
__device__
void ScanCTA(T* array, volatile int* localSum, int CTA_SIZE, int TOTAL_OFFSET)
{
    __shared__ volatile int scan[NUM_WARPS * SCAN_STRIDE];

    int tid = threadIdx.x;
    int warp = tid / WARP_SIZE;
    int lane = (WARP_SIZE - 1) & tid;

    volatile int* s = scan + SCAN_STRIDE * warp + lane + WARP_SIZE / 2;
    s[-16] = 0;

    // Read from global memory and scann in registers
    T data = {0}; // zero structure int/int2/int4
    data = array[tid]; // load from global memory int/int2/int4
    int last = Last<T>::get(data); // remember last component because exclusive scan do not include total sum
    data = RegistersExlusiveScan<T>::apply(data); // apply exclusive scan to structure int/in2/in4
    int x = Last<T>::get(data) + last;   //compute total from data = last from exclusive scanned data + last from original data

    // load total of data into shared memory
    s[0] = x;

    // Run inclusive scan on each warp's data
    int sum = x;
    #pragma unroll
    for (int i = 0; i < 5; ++i)
    {
        int offset = 1 << i;
        sum += s[-offset];
        s[0] = sum;
    }

    // Synchronize to make all the totals available to the reduction code
    __syncthreads();

    __shared__ volatile int totals[NUM_WARPS + NUM_WARPS / 2];

    if (tid < NUM_WARPS)
    {
        int total = scan[SCAN_STRIDE * tid + WARP_SIZE / 2 + WARP_SIZE - 1];

        totals[tid] = 0;
        volatile int* s2 = totals + NUM_WARPS / 2 + tid;
        int totalsSum = total;
        s2[0] = total;

        #pragma unroll
        for (int i = 0; i < LOG_NUM_WARPS; ++i)
        {
            int offset = 1 << i;
            totalsSum += s2[-offset];
            s2[0] = totalsSum;
        }

        totals[tid] = totalsSum - total;
    }

    // Synchronize to make the block scan available to all warps
    __syncthreads();

    // Add the block scan to the inclusive sum of the block
    sum += totals[warp];

    // Write the inclusive and excusive scans to global memory
    // inclusive scan elements
    int a = sum;
    int b = localSum[0];
    __syncthreads();

    sum += TOTAL_OFFSET - x + b;

    // Apply translation by sum to each component od data and store to global memory
    array[tid] = AddOffset<T>::apply(data, sum);

    if (tid == blockDim.x - 1)
    {
        localSum[0] = a + b;
    }
    __syncthreads();
}

__device__
int totals[CTAs];


template<typename T>
__global__
void UpsweepReduceKernel(const T* const array, int N, int B)
{
    const T* tile = array + B * blockDim.x * blockIdx.x;
    const int tid = threadIdx.x;

    int accum = 0;
    int tileCounter = 0;

    T element = {0};

    while (B > tileCounter++)
    {
        element = tile[tid];
        accum += RegistersReduce<T>::apply(element);
        tile += blockDim.x;
    }

    extern __shared__ int threadAccumArray[];

    threadAccumArray[tid] = accum;
    __syncthreads();

    ReduceCTA(threadAccumArray, blockDim.x);

    if (0 == tid)
    {
        accum = threadAccumArray[0];
        totals[blockIdx.x] = accum;
    }
}

__global__
void TotalsScanKernel()
{
    const int totalOffset = 0;
    volatile __shared__ int accumSum[1];

    if (threadIdx.x < CTAs)
    {
        if (0 == threadIdx.x)
            accumSum[0] = 0;
        __syncthreads();

        ScanCTA(totals, accumSum, CTAs, totalOffset);
    }
}

template<typename T>
__global__
void DownsweepScanKernel(T* const array, int N, int B)
{
    const int totalOffset = totals[blockIdx.x];
    T* tile = array + B * blockDim.x * blockIdx.x;

    volatile __shared__ int accumSum[1];

    if (0 == threadIdx.x)
        accumSum[0] = 0;
    __syncthreads();

    int tileCounter = 0;
    while (B > tileCounter++)
    {
        ScanCTA(tile, accumSum, blockDim.x, totalOffset);
        tile += blockDim.x;
    }
}


// Map number of elements per load to type
template<int ELEMENTS_PER_LOAD> struct LoadTraits;
template<> struct LoadTraits<1> { typedef int  Type; };
template<> struct LoadTraits<2> { typedef int2 Type; };
template<> struct LoadTraits<4> { typedef int4 Type; };


template <typename ElementType, int ELEMENTS_PER_LOAD>
void ParallelExclusiveScan(ElementType* const array, int N)
{
    /*
     * Parallel REDUCE-THEN-SCAN
     * Assume that size is dividible by C*T (N = k *(C*T) for some k != 0)
     */
    const int C = CTAs;        // CTAs number
    const int T = 256;         // Tile size
    const int E = ELEMENTS_PER_LOAD; // load 1, 2 or 4 consecutive 4-byte words per thread
    const int B = (N/(T*C))/E; // Tiles per CTA

    const dim3 gridDim(C);
    const dim3 blockDim(T);

    //printf("C:  %d\n", C);
    //printf("T:  %d\n", T);
    //printf("E:  %d\n", E);
    //printf("B:  %d\n", B);
    //printf("GridDim:   (%d %d %d)\n", gridDim.x, gridDim.y, gridDim.z);
    //printf("BlockDim:  (%d %d %d)\n", blockDim.x, blockDim.y, blockDim.z);

    typedef typename LoadTraits<ELEMENTS_PER_LOAD>::Type LoadType;

    UpsweepReduceKernel<LoadType>
        <<<gridDim, blockDim, T * sizeof(ElementType)>>>((const LoadType*)array, N, B);
    //checkCudaErrors(cudaDeviceSynchronize());

    TotalsScanKernel<<<1, T>>>();
    //checkCudaErrors(cudaDeviceSynchronize());

    DownsweepScanKernel<LoadType>
        <<<gridDim, blockDim>>>((LoadType*)array, N, B);
    //checkCudaErrors(cudaDeviceSynchronize());
}


template<typename ElementType>
void CreateSample(std::vector<ElementType>& array)
{
    std::srand(time(0));
    for (int i(0); i < array.size(); ++i)
        array[i] = 1; //static_cast<ElementType>(std::rand() % 100);
}

template<typename ElementType>
void SequentialExclusiveScan(std::vector<ElementType>& dest, const std::vector<ElementType>& src)
{
    ElementType sum = 0;
    for (int i(0); i < dest.size(); ++i)
    {
        dest[i] = sum;
        sum += src[i];
    }
}

template<typename ElementType>
void CompareResults(const std::vector<ElementType>& reference, const std::vector<ElementType>& array)
{
    bool incorrect = false;
    int i = 0;
    for (; i < array.size(); ++i)
    {
        if (reference[i] == array[i]) continue;
        incorrect = true;
        break;
    }
    if (incorrect) { printf("Incorrect at %d:  %d != %d\n", i, reference[i], array[i]); }
    else { printf("Perfectly correct!\n"); }
}

float PeakBandwidth(int devID)
{
    cudaError_t error;
    cudaDeviceProp deviceProp;

    error = cudaGetDevice(&devID);
    if (error != cudaSuccess)
    {
        printf("cudaGetDevice returned error code %d, line(%d)\n", error, __LINE__);
    }
    error = cudaGetDeviceProperties(&deviceProp, devID);
    if (deviceProp.computeMode == cudaComputeModeProhibited)
    {
        fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
        exit(1);
    }
    if (error != cudaSuccess)
    {
        printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
    }
    else
    {
        printf("\nGPUDevice %d:  %s\nCompute cap:  %d.%d\n",
            devID,
            deviceProp.name,
            deviceProp.major,
            deviceProp.minor);
    }
    const int clockRate = deviceProp.memoryClockRate; // [KHz]
    const int memWidth = deviceProp.memoryBusWidth;   // [bits]
    return 2.0 * clockRate * (memWidth/8.0) / 1.0e6;  // [GB/s];
}

int main(int argc, char** argv)
{
    int tilesPerCTA = 4;
    int elementsPerThread = 1;
    if (argc == 3)
    {
        tilesPerCTA = atoi(argv[1]);
        elementsPerThread = atoi(argv[2]);
    }

    int devID = 0;
    const float peakBandwidth = PeakBandwidth(devID);

    const int ARRAY_SIZE = CTAs * 256 * tilesPerCTA;

    /*
        $ ./Scan 1000 4

        GPUDevice 0:  GeForce GTX 560 Ti
        Compute cap:  2.1
        Problem size: 16384000
        CTAs number:  64
        Computation time:         1.950976 [ms]
        Peak bandwidth:         128.256 [GB/s]
        Effective bandwidth:    100.774 [GB/s]  78.573 % of peak!
        Perfectly correct!
    */

    typedef int Element;

    std::vector<Element> h_array(ARRAY_SIZE);
    std::vector<Element> h_sscan(ARRAY_SIZE);

    CreateSample(h_array);
    SequentialExclusiveScan(h_sscan, h_array);

    printf("Problem size: %d\n", ARRAY_SIZE);
    printf("CTAs number:  %d\n", CTAs);

    Element* d_array;
    checkCudaErrors(cudaMalloc((void**) &d_array, sizeof(Element) * ARRAY_SIZE));
    checkCudaErrors(cudaMemcpy(d_array, h_array.data(), sizeof(Element) * ARRAY_SIZE, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start, 0));

    switch (elementsPerThread)
    {
    default:
        printf("ERROR: Chose int/int2/int4\n");
        break;
    case 1:
        ParallelExclusiveScan<Element, 1>(d_array, ARRAY_SIZE);
        break;
    case 2:
        ParallelExclusiveScan<Element, 2>(d_array, ARRAY_SIZE);
        break;
    case 4:
        ParallelExclusiveScan<Element, 4>(d_array, ARRAY_SIZE);
        break;
    }

    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));

    float totalTimeMsec = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&totalTimeMsec, start, stop));

    const size_t loadedBytes = 2 * ARRAY_SIZE * sizeof(Element) + CTAs * sizeof(Element);
    const size_t storedBytes = ARRAY_SIZE * sizeof(Element) + + CTAs * sizeof(Element);
    const float effectiveBandwidth = (loadedBytes + storedBytes)/totalTimeMsec/1.0e6;

    printf("Computation time:         %f [ms]\n", totalTimeMsec);
    printf("Peak bandwidth:         %.3f [GB/s]\n", peakBandwidth);
    printf("Effective bandwidth:    %.3f [GB/s]  %.3f %% of peak!\n",
            effectiveBandwidth,
            (effectiveBandwidth / peakBandwidth) * 100);

    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    checkCudaErrors(cudaMemcpy(h_array.data(), d_array, sizeof(Element) * ARRAY_SIZE, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_array));
    checkCudaErrors(cudaDeviceReset());

    CompareResults(h_sscan, h_array);
}
