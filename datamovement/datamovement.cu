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


/**************************************************************************************
 * Skeletons for variable-sized grids
 **************************************************************************************/

template<typename T, int LOADS_PER_THREAD>
__global__
void MoveInOutKernel(T* const dest, const T* const source)
{
    const int id = LOADS_PER_THREAD*(blockDim.x * blockIdx.x) + threadIdx.x;

    // Register storage
    T a[LOADS_PER_THREAD];

    // Load from global memory
    #pragma unroll
    for (int i = 0; i < LOADS_PER_THREAD; ++i)
        a[i] = source[id + i*blockDim.x];

    // Store to global memory
    #pragma unroll
    for (int i = 0; i < LOADS_PER_THREAD; ++i)
        dest[id + i*blockDim.x] = a[i];

}

template
<
    typename T,
    int CTA_SIZE_,
    int LOADS_PER_THREAD_
>
struct MoveData
{
    typedef T ElementType;
    static const int CTA_SIZE = CTA_SIZE_;
    static const int LOADS_PER_THREAD = LOADS_PER_THREAD_;
};

template
<
    typename T,
    int CTA_SIZE_,
    int LOADS_PER_THREAD_
>
struct MoveDataInOut : public MoveData<T, CTA_SIZE_, LOADS_PER_THREAD_>
{
    inline void operator() (T* const dest, const T* const source, const dim3& grid)
    {
        MoveInOutKernel<T, LOADS_PER_THREAD_><<<grid, CTA_SIZE_>>>(dest, source);
    }
};


/**************************************************************************************
 * Skeletons for fixed-sized grids
 **************************************************************************************/
const int CTAs = 64;

//TODO Fixed-size grid performance analysis


template<typename MovingSkeleton>
void MeasureInOutVariableGrid(
        MovingSkeleton skeleton,
        float* const d_dest,
        const float* const d_source,
        size_t bytes,
        float peakBandwidth)
{
    typedef typename MovingSkeleton::ElementType ElementType;

    const int N = bytes/sizeof(ElementType);
    const int T = MovingSkeleton::CTA_SIZE;
    const int E = MovingSkeleton::LOADS_PER_THREAD;

    const dim3 grid((N/T)/E);

//    printf("Bytes:   %d\n", bytes);
//    printf("T size:  %d\n", sizeof(ElementType));
//    printf("Problem: %d\n", N);
//    printf("ThBlock: %d\n", T);
//    printf("Grid:    %d\n", grid.x);

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start, 0));

    skeleton((ElementType*)d_dest, (ElementType*)d_source, grid);

    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));

    float totalTimeMsec = 0.0f;

    checkCudaErrors(cudaEventElapsedTime(&totalTimeMsec, start, stop));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    const size_t loadedBytes = bytes;
    const size_t storedBytes = bytes;
    const float effectiveBandwidth = (loadedBytes + storedBytes)/totalTimeMsec/1.0e6;

    printf("IN-OUT-%d   %4d [CTASIZE]    %5d [GRID]    %2zd [Bytes/element]    "
            "%f [ms]    %7.3f [GB/s]    %7.3f [GB/s]    %7.3f %% of peak\n",
            E, T, grid.x, sizeof(ElementType),
            totalTimeMsec, peakBandwidth, effectiveBandwidth, (effectiveBandwidth / peakBandwidth) * 100);
}


void MeasureCudaMemcpy(
        float* const d_dest,
        const float* const d_source,
        size_t bytes,
        float peakBandwidth)
{
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start, 0));

    cudaMemcpy(d_dest, d_source, bytes, cudaMemcpyDeviceToDevice);

    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));

    float totalTimeMsec = 0.0f;

    checkCudaErrors(cudaEventElapsedTime(&totalTimeMsec, start, stop));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    const size_t loadedBytes = bytes;
    const size_t storedBytes = bytes;
    const float effectiveBandwidth = (loadedBytes + storedBytes)/totalTimeMsec/1.0e6;

    printf("MEMCPY     %4d [CTASIZE]    %5d [GRID]    %2zd [Bytes/element]    "
            "%f [ms]    %7.3f [GB/s]    %7.3f [GB/s]    %7.3f %% of peak\n",
            0, 0, sizeof(float),
            totalTimeMsec, peakBandwidth, effectiveBandwidth, (effectiveBandwidth / peakBandwidth) * 100);
}

template<typename ElementType>
void CreateSample(std::vector<ElementType>& array)
{
    std::srand(time(0));
    for (int i(0); i < array.size(); ++i)
        array[i] = static_cast<ElementType>(std::rand() % 100);
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
    int tilesPerCTA = 400;
    if (argc > 1)
    {
        tilesPerCTA = atoi(argv[1]);
    }

    int devID = 0;
    const float peakBandwidth = PeakBandwidth(devID);
    const size_t ARRAY_SIZE = CTAs * 256 * tilesPerCTA;
    const size_t bytes = sizeof(float) * ARRAY_SIZE;

    std::vector<float> h_source(ARRAY_SIZE);
    CreateSample(h_source);

    printf("Problem size: %zd\n", ARRAY_SIZE);

    float* d_source;
    float* d_dest;

    checkCudaErrors(cudaMalloc((void**) &d_source, bytes));
    checkCudaErrors(cudaMalloc((void**) &d_dest, bytes));
    checkCudaErrors(cudaMemcpy(d_source, h_source.data(), bytes, cudaMemcpyHostToDevice));


    printf("======================================================================"
            "===================================================================\n");
    printf("Skeletons for variable-sized grids\n");
    printf("======================================================================"
            "===================================================================\n");

    MeasureInOutVariableGrid(MoveDataInOut< float,  128,  1 >(), d_dest, d_source, bytes, peakBandwidth);
    MeasureInOutVariableGrid(MoveDataInOut< float2, 128,  1 >(), d_dest, d_source, bytes, peakBandwidth);
    MeasureInOutVariableGrid(MoveDataInOut< float4, 128,  1 >(), d_dest, d_source, bytes, peakBandwidth);

    MeasureInOutVariableGrid(MoveDataInOut< float,  256,  1 >(), d_dest, d_source, bytes, peakBandwidth);
    MeasureInOutVariableGrid(MoveDataInOut< float2, 256,  1 >(), d_dest, d_source, bytes, peakBandwidth);
    MeasureInOutVariableGrid(MoveDataInOut< float4, 256,  1 >(), d_dest, d_source, bytes, peakBandwidth);

    MeasureInOutVariableGrid(MoveDataInOut< float,  512,  1 >(), d_dest, d_source, bytes, peakBandwidth);
    MeasureInOutVariableGrid(MoveDataInOut< float2, 512,  1 >(), d_dest, d_source, bytes, peakBandwidth);
    MeasureInOutVariableGrid(MoveDataInOut< float4, 512,  1 >(), d_dest, d_source, bytes, peakBandwidth);

    MeasureInOutVariableGrid(MoveDataInOut< float,  1024, 1 >(), d_dest, d_source, bytes, peakBandwidth);
    MeasureInOutVariableGrid(MoveDataInOut< float2, 1024, 1 >(), d_dest, d_source, bytes, peakBandwidth);
    MeasureInOutVariableGrid(MoveDataInOut< float4, 1024, 1 >(), d_dest, d_source, bytes, peakBandwidth);

    printf("======================================================================"
            "===================================================================\n");

    MeasureInOutVariableGrid(MoveDataInOut< float,  128,  2 >(), d_dest, d_source, bytes, peakBandwidth);
    MeasureInOutVariableGrid(MoveDataInOut< float2, 128,  2 >(), d_dest, d_source, bytes, peakBandwidth);
    MeasureInOutVariableGrid(MoveDataInOut< float4, 128,  2 >(), d_dest, d_source, bytes, peakBandwidth);

    MeasureInOutVariableGrid(MoveDataInOut< float,  256,  2 >(), d_dest, d_source, bytes, peakBandwidth);
    MeasureInOutVariableGrid(MoveDataInOut< float2, 256,  2 >(), d_dest, d_source, bytes, peakBandwidth);
    MeasureInOutVariableGrid(MoveDataInOut< float4, 256,  2 >(), d_dest, d_source, bytes, peakBandwidth);

    MeasureInOutVariableGrid(MoveDataInOut< float,  512,  2 >(), d_dest, d_source, bytes, peakBandwidth);
    MeasureInOutVariableGrid(MoveDataInOut< float2, 512,  2 >(), d_dest, d_source, bytes, peakBandwidth);
    MeasureInOutVariableGrid(MoveDataInOut< float4, 512,  2 >(), d_dest, d_source, bytes, peakBandwidth);

    MeasureInOutVariableGrid(MoveDataInOut< float,  1024, 2 >(), d_dest, d_source, bytes, peakBandwidth);
    MeasureInOutVariableGrid(MoveDataInOut< float2, 1024, 2 >(), d_dest, d_source, bytes, peakBandwidth);
    MeasureInOutVariableGrid(MoveDataInOut< float4, 1024, 2 >(), d_dest, d_source, bytes, peakBandwidth);

    printf("======================================================================"
            "===================================================================\n");

    MeasureInOutVariableGrid(MoveDataInOut< float,  128,  4 >(), d_dest, d_source, bytes, peakBandwidth);
    MeasureInOutVariableGrid(MoveDataInOut< float2, 128,  4 >(), d_dest, d_source, bytes, peakBandwidth);
    MeasureInOutVariableGrid(MoveDataInOut< float4, 128,  4 >(), d_dest, d_source, bytes, peakBandwidth);

    MeasureInOutVariableGrid(MoveDataInOut< float,  256,  4 >(), d_dest, d_source, bytes, peakBandwidth);
    MeasureInOutVariableGrid(MoveDataInOut< float2, 256,  4 >(), d_dest, d_source, bytes, peakBandwidth);
    MeasureInOutVariableGrid(MoveDataInOut< float4, 256,  4 >(), d_dest, d_source, bytes, peakBandwidth);

    MeasureInOutVariableGrid(MoveDataInOut< float,  512,  4 >(), d_dest, d_source, bytes, peakBandwidth);
    MeasureInOutVariableGrid(MoveDataInOut< float2, 512,  4 >(), d_dest, d_source, bytes, peakBandwidth);
    MeasureInOutVariableGrid(MoveDataInOut< float4, 512,  4 >(), d_dest, d_source, bytes, peakBandwidth);

    MeasureInOutVariableGrid(MoveDataInOut< float,  1024, 4 >(), d_dest, d_source, bytes, peakBandwidth);
    MeasureInOutVariableGrid(MoveDataInOut< float2, 1024, 4 >(), d_dest, d_source, bytes, peakBandwidth);
    MeasureInOutVariableGrid(MoveDataInOut< float4, 1024, 4 >(), d_dest, d_source, bytes, peakBandwidth);

    printf("======================================================================"
            "===================================================================\n");

    MeasureCudaMemcpy(d_dest, d_source, bytes, peakBandwidth);

    printf("======================================================================"
            "===================================================================\n");
    printf("Skeletons for fixed-sized grids\n");
    printf("======================================================================"
            "===================================================================\n");

    // TODO Fixed-size grid performance analysis

    printf("======================================================================"
            "===================================================================\n");


    checkCudaErrors(cudaMemcpy(h_source.data(), d_dest, bytes, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(d_source));
    checkCudaErrors(cudaFree(d_dest));
    checkCudaErrors(cudaDeviceReset());
}
