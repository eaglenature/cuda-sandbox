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


template <typename T> struct Accum;
template <> struct Accum<int>  { __device__ __host__ inline static int apply(int  a) { return a; } };
template <> struct Accum<int2> { __device__ __host__ inline static int apply(int2 a) { return a.x + a.y; } };
template <> struct Accum<int4> { __device__ __host__ inline static int apply(int4 a) { return a.x + a.y + a.z + a.w; } };

template<typename T>
__global__
void ReduceThreadblocksKernel(const T* const array, int* const totals, int N, int B)
{
    const T* datatile = array + B * blockDim.x * blockIdx.x;
    const int tid = threadIdx.x;

    int accum = 0;
    int tile = 0;

    T element = {0};

    while (tile++ < B)
    {
        element = datatile[tid];
        accum += Accum<T>::apply(element);
        datatile += blockDim.x;
    }

    extern __shared__ int shared[];

    shared[tid] = accum;
    __syncthreads();

    ReduceCTA(shared, blockDim.x);

    if (0 == tid)
    {
        accum = shared[0];
        totals[blockIdx.x] = accum;
    }
}

__global__
void ReduceTotalsKernel(int* const totals)
{
    const int tid = threadIdx.x;

    extern __shared__ int smem[];

    smem[tid] = totals[tid];
    __syncthreads();

    ReduceCTA(smem, blockDim.x);

    if (0 == tid)
        totals[0] = smem[0];
}

/*
 * Minimum number of CTAs to fill GPU GTX560Ti (8 SM cores)
 */
const int CTAs = 64;

// Map number of elements per load to type
template<int ELEMENTS_PER_LOAD> struct LoadTraits;
template<> struct LoadTraits<1> { typedef int  Type; };
template<> struct LoadTraits<2> { typedef int2 Type; };
template<> struct LoadTraits<4> { typedef int4 Type; };


template <typename ElementType, int ELEMENTS_PER_LOAD>
void ParallelReduce(const ElementType* const array, ElementType* const totals, int N)
{
    /*
     * Parallel REDUCE
     * Assume that size is multiply of C*T (N = k *(C*T) for some k > 0)
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

    ReduceThreadblocksKernel<LoadType>
        <<<gridDim, blockDim, T * sizeof(ElementType)>>>((const LoadType*)array, totals, N, B);
    //checkCudaErrors(cudaDeviceSynchronize());

    ReduceTotalsKernel<<<1, C, C * sizeof(ElementType)>>>(totals);
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
ElementType SequentialReduce(const std::vector<ElementType>& src)
{
    ElementType sum = 0;
    for (int i(0); i < src.size(); ++i)
        sum += src[i];
    return sum;
}

template<typename ElementType>
void CompareResults(ElementType hostresult, ElementType deviceresult)
{
    bool incorrect = (hostresult != deviceresult);
    if (incorrect) { printf("Incorrect  %d != %d\n", hostresult, deviceresult); }
    else { printf("Perfectly correct!\nGPU sum reduction: %d\n", deviceresult); }
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
        fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, "
        		"no threads can use ::cudaSetDevice().\n");
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
    int tilesPerCTA = 200;
    int elementsPerThread = 1;
    if (argc > 2)
    {
        tilesPerCTA = atoi(argv[1]);
        elementsPerThread = atoi(argv[2]);
    }

    int devID = 0;
    const float peakBandwidth = PeakBandwidth(devID);
    /*
        const int ARRAY_SIZE = CTAs * 256 * 10000;

        $ ./Reduce 10000 2

        GPUDevice 0:  GeForce GTX 560 Ti
        Compute cap:  2.1
        Problem size: 163840000
        CTAs number:  64
        Computation time:         5.800864 [ms]
        Peak bandwidth:         128.256 [GB/s]
        Effective bandwidth:    112.976 [GB/s]  88.087 % of the peak!
        Perfectly correct!
        GPU sum reduction: 163840000

        const int ARRAY_SIZE = CTAs * 256 * 400; // 82% of peak
        const int ARRAY_SIZE = CTAs * 256 * 800; // 84% of peak
    */

    const int ARRAY_SIZE = CTAs * 256 * tilesPerCTA;

    typedef int Element;

    std::vector<Element> h_array(ARRAY_SIZE);
    std::vector<Element> h_totals(CTAs);

    CreateSample(h_array);
    Element seqresult = SequentialReduce(h_array);

    printf("Problem size: %d\n", ARRAY_SIZE);
    printf("CTAs number:  %d\n", CTAs);

    Element* d_array;
    Element* d_totals;
    checkCudaErrors(cudaMalloc((void**) &d_array,  sizeof(Element) * ARRAY_SIZE));
    checkCudaErrors(cudaMalloc((void**) &d_totals, sizeof(Element) * CTAs));

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
        ParallelReduce<Element, 1>(d_array, d_totals, ARRAY_SIZE);
        break;
    case 2:
        ParallelReduce<Element, 2>(d_array, d_totals, ARRAY_SIZE);
        break;
    case 4:
        ParallelReduce<Element, 4>(d_array, d_totals, ARRAY_SIZE);
        break;
    }

    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));

    float totalTimeMsec = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&totalTimeMsec, start, stop));

    const size_t loadedBytes = ARRAY_SIZE * sizeof(Element) + CTAs * sizeof(Element);
    const size_t storedBytes = 2 * CTAs * sizeof(Element);
    const float effectiveBandwidth = (loadedBytes + storedBytes)/totalTimeMsec/1.0e6;

    printf("Computation time:         %f [ms]\n", totalTimeMsec);
    printf("Peak bandwidth:         %.3f [GB/s]\n", peakBandwidth);
    printf("Effective bandwidth:    %.3f [GB/s]  %.3f %% of the peak!\n",
            effectiveBandwidth,
            (effectiveBandwidth / peakBandwidth) * 100);

    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    checkCudaErrors(cudaMemcpy(h_totals.data(), d_totals, CTAs * sizeof(Element), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_array));
    checkCudaErrors(cudaFree(d_totals));
    checkCudaErrors(cudaDeviceReset());

    CompareResults(seqresult, h_totals.front());
}
