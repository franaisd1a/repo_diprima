
#include "externalClass.cuh"
/*
void externalClass::squareOnDevice(double *a_h, const int N) {
	double *a_d = new double[N]; // initialize a_d as an array with N double pointer
	size_t size = N * sizeof(double);
	cudaMalloc((void **) &a_d, size);
	cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);

	int block_size = 4;
	int n_blocks = N/block_size + (N%block_size == 0 ? 0:1);
	square_array <<<n_blocks, block_size>>> (a_d, N);
	cudaMemcpy(a_h, a_d, size, cudaMemcpyDeviceToHost);
	cudaFree(a_d);
}
*/

void externalClass::medianCUDAKernel(const cv::gpu::GpuMat &src, cv::gpu::GpuMat &dst, int szK)
{
	dim3 cthreads(16, 16);

	dim3 cblocks(static_cast<int>(std::ceil(src.size().width / 
					static_cast<double>(cthreads.x)))
			   , static_cast<int>(std::ceil(src.size().height / 
					static_cast<double>(cthreads.y))));

	medianKernel<<<cblocks, cthreads>>>(src.ptr(), dst.ptr(), src.cols, src.rows, szK);

}

void externalClass::convolutionCUDAKernel(const cv::gpu::GpuMat &src, cv::gpu::GpuMat &dst, int szK)
{
	dim3 cthreads(16, 16);

	dim3 cblocks(static_cast<int>(std::ceil(src.size().width / 
					static_cast<double>(cthreads.x)))
			   , static_cast<int>(std::ceil(src.size().height / 
					static_cast<double>(cthreads.y))));

  /*unsigned short* arrayValue = 0;
  size_t num_bytes = szK * szK * sizeof(unsigned short);
  cudaMalloc 	( 	(void**)& arrayValue, num_bytes);*/

	convolutionKernel<<<cblocks, cthreads>>>(src.ptr(), dst.ptr(), src.cols, src.rows, szK);

}

void externalClass::convolutionThreshCUDAKernel(const cv::gpu::GpuMat &src, cv::gpu::GpuMat &dst, int szK, int thresh, int maxval)
{
	dim3 cthreads(16, 16);

	dim3 cblocks(static_cast<int>(std::ceil(src.size().width / 
					static_cast<double>(cthreads.x)))
			   , static_cast<int>(std::ceil(src.size().height / 
					static_cast<double>(cthreads.y))));

	convolutionKernelThreshold<<<cblocks, cthreads>>>(src.ptr(), dst.ptr(), src.cols, src.rows, szK, thresh, maxval);                
}

void externalClass::fillImgCUDAKernel(const cv::gpu::GpuMat &mask, cv::gpu::GpuMat &dst, int tlX, int tlY, int brX, int brY)
{
	dim3 cthreads(16, 16);

	dim3 cblocks(static_cast<int>(std::ceil(dst.size().width / 
					static_cast<double>(cthreads.x)))
			   , static_cast<int>(std::ceil(dst.size().height / 
					static_cast<double>(cthreads.y))));

	fillImgKernel<<<cblocks, cthreads>>>(mask.ptr(), dst.ptr(), mask.cols, dst.cols, tlX, tlY, brX, brY);

}


