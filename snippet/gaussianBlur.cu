

#define MAX_CONST_ARRAY_FILTER_SIZE 19*19
__constant__ float const_filter[MAX_CONST_ARRAY_FILTER_SIZE];

template<int FilterWidth>
__global__
void gaussian_blur(uchar4 * const outputChannel,
		   const uchar4 * const inputChannel,
		   const size_t numRows,
		   const size_t numCols) {

    //------------------------------------------------------------------------------------------

    // absolute pixel position
    const int2 pos  = make_int2(blockIdx.x * blockDim.x + threadIdx.x, 
                                blockIdx.y * blockDim.y + threadIdx.y);

    const int2 diff = make_int2(blockDim.x - threadIdx.x, 
                                blockDim.y - threadIdx.y);

    const int haloLeft  = (blockIdx.x - 1) * blockDim.x + threadIdx.x;
    const int haloRight = (blockIdx.x + 1) * blockDim.x + threadIdx.x;
    const int haloUp    = (blockIdx.y - 1) * blockDim.y + threadIdx.y;
    const int haloDown  = (blockIdx.y + 1) * blockDim.y + threadIdx.y;

    const int radius = FilterWidth >> 1;
    const int w = blockDim.x + FilterWidth - 1;

    int s = threadIdx.y + radius;
    const int yupp = (s - blockDim.y) * w;
    const int ymid = s * w;
    const int ylow = (s + blockDim.y) * w;

    s = threadIdx.x + radius;
    const int xleft  = s - blockDim.x;
    const int xcent  = s;
    const int xright = s + blockDim.x;

    //-------------------------------------------------------------------------------------------
    // Loading phase: allocate shered memory for pixels (all channels in one uchar3 element [rgb])

    extern __shared__ uchar3 smem[];

    uchar4 t4;
    int x, y;

    // upper left halo region
    if (radius >= diff.x && radius >= diff.y) {
        x = max(haloLeft, 0);
        y = max(haloUp, 0);
        t4 = inputChannel[y*numCols + x];
        smem[yupp + xleft] = make_uchar3(t4.x, t4.y, t4.z);
    }
    // middle halo region
    if (radius >= diff.x) {
        x = max(haloLeft, 0);
        y = min(pos.y, numRows - 1);
        t4 = inputChannel[y*numCols + x];
        smem[ymid + xleft] = make_uchar3(t4.x, t4.y, t4.z);
    }
    // lower left halo region
    if (radius >= diff.x && threadIdx.y < radius) {
        x = max(haloLeft, 0);
        y = min(haloDown, numRows - 1);
        t4 = inputChannel[y*numCols + x];
        smem[ylow + xleft] = make_uchar3(t4.x, t4.y, t4.z);
    }
    // upper central halo region
    if (radius >= diff.y) {
        x = min(pos.x, numCols - 1);
        y = max(haloUp, 0);
        t4 = inputChannel[y*numCols + x];
        smem[yupp + xcent] = make_uchar3(t4.x, t4.y, t4.z);
    }

    // middle central halo region
    x = min(pos.x, numCols - 1);
    y = min(pos.y, numRows - 1);
    t4 = inputChannel[y*numCols + x];
    smem[ymid + xcent] = make_uchar3(t4.x, t4.y, t4.z);

    // lower central halo region
    if (threadIdx.y < radius) {
        x = min(pos.x, numCols - 1);
        y = min(haloDown, numRows - 1);
        t4 = inputChannel[y*numCols + x];
        smem[ylow + xcent] = make_uchar3(t4.x, t4.y, t4.z);
    }
    // upper right halo region
    if (threadIdx.x < radius && radius >= diff.y)  {
        x = min(haloRight, numCols - 1);
        y = max(haloUp, 0);
        t4 = inputChannel[y*numCols + x];
        smem[yupp + xright] = make_uchar3(t4.x, t4.y, t4.z);
    }
    // middle right halo region
    if (threadIdx.x < radius) {
        x = min(haloRight, numCols - 1);
        y = min(pos.y, numRows - 1);
        t4 = inputChannel[y*numCols + x];
        smem[ymid + xright] = make_uchar3(t4.x, t4.y, t4.z);
    }
    // lower right halo region
    if (threadIdx.x < radius && threadIdx.y < radius) {
        x = min(haloRight, numCols - 1);
        y = min(haloDown, numRows - 1);
        t4 = inputChannel[y*numCols + x];
        smem[ylow + xright] = make_uchar3(t4.x, t4.y, t4.z);
    }
    __syncthreads();

    //-------------------------------------------------------------------------------------------
    // Convolution  phase

    uchar3 imagePixel;
    float filterWeight, R, G, B;

#pragma unroll
    for (int j = 0; j < FilterWidth; ++j) {
#pragma unroll
        for (int i = 0; i < FilterWidth; ++i) {
            filterWeight = const_filter[j * FilterWidth + i];
            imagePixel = smem[(threadIdx.y + j) * w + (threadIdx.x + i)];
            R += static_cast<float>(imagePixel.x) * filterWeight;
            G += static_cast<float>(imagePixel.y) * filterWeight;
            B += static_cast<float>(imagePixel.z) * filterWeight;
        }
    }

    if (pos.x < numCols && pos.y < numRows) {
        outputChannel[pos.y * numCols + pos.x] = make_uchar4(static_cast<unsigned char>(R),
                                                             static_cast<unsigned char>(G),
                                                             static_cast<unsigned char>(B),
                                                             255);
    }
}

void gaussianBlur(uchar4 * const d_outputImageRGBA,
		  const uchar4 * const d_inputImageRGBA,
		  const size_t numRows,
		  const size_t numCols,
		  const size_t filterWidth) {

	const dim3 blockSize(64, 8);
	const dim3 gridSize((numCols + blockSize.x - 1)/blockSize.x, (numRows + blockSize.y - 1)/blockSize.y);
	const size_t shmemSize = (blockSize.x + filterWidth - 1) * (blockSize.y + filterWidth - 1) * sizeof(uchar3);

	switch (filterWidth)
	{
	case 3:  gaussian_blur<3><<< gridSize, blockSize, shmemSize>>>(d_outputImageRGBA, d_inputImageRGBA, numRows, numCols); break;
	case 5:  gaussian_blur<5><<< gridSize, blockSize, shmemSize>>>(d_outputImageRGBA, d_inputImageRGBA, numRows, numCols); break;
	case 7:  gaussian_blur<7><<< gridSize, blockSize, shmemSize>>>(d_outputImageRGBA, d_inputImageRGBA, numRows, numCols); break;
	case 9:  gaussian_blur<9><<< gridSize, blockSize, shmemSize>>>(d_outputImageRGBA, d_inputImageRGBA, numRows, numCols); break;
	case 11: gaussian_blur<11><<<gridSize, blockSize, shmemSize>>>(d_outputImageRGBA, d_inputImageRGBA, numRows, numCols); break;
	case 13: gaussian_blur<13><<<gridSize, blockSize, shmemSize>>>(d_outputImageRGBA, d_inputImageRGBA, numRows, numCols); break;
	case 15: gaussian_blur<15><<<gridSize, blockSize, shmemSize>>>(d_outputImageRGBA, d_inputImageRGBA, numRows, numCols); break;
	case 17: gaussian_blur<17><<<gridSize, blockSize, shmemSize>>>(d_outputImageRGBA, d_inputImageRGBA, numRows, numCols); break;
	case 19: gaussian_blur<19><<<gridSize, blockSize, shmemSize>>>(d_outputImageRGBA, d_inputImageRGBA, numRows, numCols); break;
	default:;break;
	}
}
