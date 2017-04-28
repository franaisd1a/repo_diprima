
#include "externalClass.cuh"

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

void externalClass::LUT(cv::gpu::GpuMat& lut, const double outByteDepth, const int minValue, const int maxValue)
{
	dim3 cthreads(32, 1);

	dim3 cblocks(static_cast<int>(std::ceil(lut.size().width / 
					static_cast<double>(cthreads.x)))
			   , static_cast<int>(std::ceil(lut.size().height / 
					static_cast<double>(cthreads.y))));

  lutKernel<<<cblocks, cthreads>>>(lut.ptr<float>(), lut.cols, outByteDepth, minValue, maxValue);
}

void externalClass::stretching(const cv::gpu::GpuMat& src, const cv::gpu::GpuMat& lut, cv::gpu::GpuMat& dst)
{
	dim3 cthreads(16, 16);

	dim3 cblocks(static_cast<int>(std::ceil(dst.size().width / 
					static_cast<double>(cthreads.x)))
			   , static_cast<int>(std::ceil(dst.size().height / 
					static_cast<double>(cthreads.y))));

  stretchingKernel<<<cblocks, cthreads>>>(src.ptr<ushort>(), lut.ptr<float>(), dst.ptr(), src.cols, src.rows);
}


