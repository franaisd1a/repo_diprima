
#include "externalClass.cuh"

#define THREAD_X 16
#define THREAD_Y 16
#define THREAD_LUT 32

//Thread_X 16
//Thread_LUT 32

void externalClass::medianCUDAKernel(const cv::gpu::GpuMat &src, cv::gpu::GpuMat &dst, int szK)
{
	dim3 cthreads(THREAD_X, THREAD_Y);

	dim3 cblocks(static_cast<int>(std::ceil(src.size().width / 
					static_cast<double>(cthreads.x)))
			   , static_cast<int>(std::ceil(src.size().height / 
					static_cast<double>(cthreads.y))));

	medianKernel<<<cblocks, cthreads>>>(src.ptr(), dst.ptr(), src.cols, src.rows, szK);

}

void externalClass::convolutionCUDAKernel(const cv::gpu::GpuMat &src, cv::gpu::GpuMat &dst, int szK)
{
	dim3 cthreads(THREAD_X, THREAD_Y);

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
	dim3 cthreads(THREAD_X, THREAD_Y);

	dim3 cblocks(static_cast<int>(std::ceil(src.size().width / 
					static_cast<double>(cthreads.x)))
			   , static_cast<int>(std::ceil(src.size().height / 
					static_cast<double>(cthreads.y))));

	convolutionKernelThreshold<<<cblocks, cthreads>>>(src.ptr(), dst.ptr(), src.cols, src.rows, szK, thresh, maxval);                
}

void externalClass::fillImgCUDAKernel(const cv::gpu::GpuMat &mask, cv::gpu::GpuMat &dst, int tlX, int tlY, int brX, int brY)
{
	dim3 cthreads(THREAD_X, THREAD_Y);

	dim3 cblocks(static_cast<int>(std::ceil(dst.size().width / 
					static_cast<double>(cthreads.x)))
			   , static_cast<int>(std::ceil(dst.size().height / 
					static_cast<double>(cthreads.y))));

	fillImgKernel<<<cblocks, cthreads>>>(mask.ptr(), dst.ptr(), mask.cols, dst.cols, tlX, tlY, brX, brY);

}

void externalClass::LUT(cv::gpu::GpuMat& lut, const double outByteDepth, const int minValue, const int maxValue)
{
	dim3 cthreads(THREAD_LUT, 1);

	dim3 cblocks(static_cast<int>(std::ceil(lut.size().width / 
					static_cast<double>(cthreads.x)))
			   , static_cast<int>(std::ceil(lut.size().height / 
					static_cast<double>(cthreads.y))));

  lutKernel<<<cblocks, cthreads>>>(lut.ptr<float>(), lut.cols, outByteDepth, minValue, maxValue);
}

void externalClass::stretching(const cv::gpu::GpuMat& src, const cv::gpu::GpuMat& lut, cv::gpu::GpuMat& dst)
{
	dim3 cthreads(THREAD_X, THREAD_Y);

	dim3 cblocks(static_cast<int>(std::ceil(dst.size().width / 
					static_cast<double>(cthreads.x)))
			   , static_cast<int>(std::ceil(dst.size().height / 
					static_cast<double>(cthreads.y))));

  stretchingKernel<<<cblocks, cthreads>>>(src.ptr<ushort>(), lut.ptr<float>(), dst.ptr(), src.cols, src.rows);
}

void externalClass::histogram(const cv::gpu::GpuMat& src, cv::gpu::GpuMat& hist)
{
  int threadPerBlock = 1024;
  int size = src.cols * src.rows;
  
#if 1
  int blockPerGrid = 1;//20;
#else
  /*int blockPerGrid = static_cast<int>(std::ceil(
    static_cast<double>(size)/static_cast<double>(threadPerBlock)));*/
#endif
  
  histKernel<<<blockPerGrid, threadPerBlock>>>(src.ptr<ushort>(), size, hist.ptr<int>());
}

void externalClass::lowerLimKernel(const cv::gpu::GpuMat& hist, const int peakPos, const double thresh, int& minValue)
{
  int threadPerBlock = 1;
  int blockPerGrid = 1;

  int h_val = 0;
  int* d_val;

  cudaMalloc(&d_val, sizeof(int));

  lowerLimit<<<blockPerGrid, threadPerBlock>>>(hist.ptr<int>(), peakPos, thresh, d_val);

  cudaMemcpy(&h_val, d_val, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_val);

  minValue = h_val;
}

void externalClass::upperLimKernel(const cv::gpu::GpuMat& hist, const int peakPos, const double thresh, int& maxValue)
{
  int threadPerBlock = 1;
  int blockPerGrid = 1;

  int h_val = 0;
  int* d_val;

  cudaMalloc(&d_val, sizeof(int));

  upperLimit<<<blockPerGrid, threadPerBlock>>>(hist.ptr<int>(), peakPos, hist.cols, thresh, d_val);

  cudaMemcpy(&h_val, d_val, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_val);

  maxValue = h_val;
}

