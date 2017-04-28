#ifndef EXTERNALCLASS_CUH_
#define EXTERNALCLASS_CUH_

#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv/highgui.h>
#include "opencv2/gpu/gpu.hpp"


#if 1
#include </usr/local/cuda-6.5/include/cuda.h>
#include </usr/local/cuda-6.5/include/cuda_runtime_api.h>
#else
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif

/*
__global__ void square_array(double *a, int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx<N) a[idx] = a[idx] * a[idx];
	printf("GPU PRINT: idx = %d, a = %f\n", idx, a[idx]);
}
*/

__global__ void medianKernel
(const uchar* input, uchar* output, int Image_Width, int Image_Height, int szK)
{
  int numValue = szK*szK;
  int radius = (int)(szK/2);
  int middleValue = (int)(numValue/2);

  unsigned short surround[82];
//unsigned short* surround = new unsigned short[numValue];

  int iterator;

  const int x     = blockDim.x * blockIdx.x + threadIdx.x;
  const int y     = blockDim.y * blockIdx.y + threadIdx.y;

  if( (x >= (Image_Width - radius)) || (y >= Image_Height - radius) || (x < radius) || (y < radius)) return;

  // --- Fill array private to the threads
  iterator = 0;
  for (int r = x - radius; r <= x + radius; r++) {
      for (int c = y - radius; c <= y + radius; c++) {
          surround[iterator] = input[c*Image_Width+r];
          iterator++;
      }
  }

  // --- Sort private array to find the median using Bubble Short
  for (int i=0; i<=middleValue; ++i) {

      // --- Find the position of the minimum element
      int minval=i;
      for (int l=i+1; l<numValue; ++l) if (surround[l] < surround[minval]) minval=l;

      // --- Put found minimum element in its place
      unsigned short temp = surround[i];
      surround[i]=surround[minval];
      surround[minval]=temp;
  }

  // --- Pick the middle one
  output[(y*Image_Width)+x]=surround[middleValue];
  delete[] surround;
}

__global__ void convolutionKernel
(const uchar * input, uchar * output, int imgW, int imgH, int szK)
{
  int radius = (int)(szK/2);  

  const int x     = blockDim.x * blockIdx.x + threadIdx.x;
  const int y     = blockDim.y * blockIdx.y + threadIdx.y;  
 
  if( (x >= (imgW - radius)) || (y >= imgH - radius) || (x == radius) || (y == radius)) return;

  unsigned short sum = 0;
  for (int r = x - radius; r <= x + radius; r++) {
      for (int c = y - radius; c <= y + radius; c++) {
          sum += input[c*imgW+r];
      }
  }
  
  output[(y*imgW)+x]=sum;
}

__global__ void convolutionKernelThreshold
(const uchar * input, uchar * output, int imgW, int imgH, int szK, int thresh, int maxval)
{
  int rad = (int)(szK/2);

  const int x     = blockDim.x * blockIdx.x + threadIdx.x;
  const int y     = blockDim.y * blockIdx.y + threadIdx.y;  
 
  if( (x >= (imgW - rad)) || (y >= imgH - rad) || (x < rad) || (y < rad)) 
  {
  }
  else 
  {
    unsigned short sum = 0;
    for (int r = x - rad; r <= x + rad; r++) {
        for (int c = y - rad; c <= y + rad; c++) {
            sum += input[(c*imgW)+r];
        }
    }
    //non funziona restituisce img nera
    if(sum>thresh*maxval) {
      output[(y*imgW)+x]=maxval;
    }
  }
}

__global__ void fillImgKernel
(const uchar * mask, uchar * dst, int maskW, int dstW, int tlX, int tlY, int brX, int brY)
{
  const int x     = blockDim.x * blockIdx.x + threadIdx.x;
  const int y     = blockDim.y * blockIdx.y + threadIdx.y;  
 
  if ( (x>=tlX) && (x<brX) && (y>=tlY) && (y<brY)  )
  {
    dst[(y*dstW)+x]=mask[(y-tlY)*maskW+(x-tlX)];
  }  
}

__global__ void lutKernel
(float* pLUT, int imgW, const double outputByteDepth
  , const int minValue, const int maxValue)
{
  const int i = blockDim.x * blockIdx.x + threadIdx.x;  
  double scaleFactor = outputByteDepth/(maxValue - minValue);
  if( (i >= imgW) || (i < 0) )
  {}
  else
  {
    if (i < minValue)
    {
      pLUT[i] = 0;
    }
    else if (i > maxValue) {
      pLUT[i] = outputByteDepth;
    }
    else {
      pLUT[i] = (i - minValue)*scaleFactor;
    }
  }  
  return;
}

__global__ void stretchingKernel
(const ushort* pInImg, const float* pLUT, uchar* pDstImg, int imgW, int imgH)
{
  const int x = blockDim.x * blockIdx.x + threadIdx.x;
  const int y = blockDim.y * blockIdx.y + threadIdx.y;  
 
  if( (x >= imgW) || (y >= imgH ) || (x < 0) || (y < 0)) 
  {
  }
  else 
  {
    ushort valIn = pInImg[(y*imgW)+x];
    float valLUT = pLUT[valIn];
    pDstImg[(y*imgW)+x] = (uchar)(valLUT);
  }
}


/******************************************************************************/
/*                            externalClass                                   */
/******************************************************************************/

class externalClass {

public:
	//int GetInt() { return 5; };

	void medianCUDAKernel(const cv::gpu::GpuMat &src, cv::gpu::GpuMat &dst, int szK);
	void convolutionCUDAKernel(const cv::gpu::GpuMat &src, cv::gpu::GpuMat &dst, int szK);
  void convolutionThreshCUDAKernel(const cv::gpu::GpuMat &src, cv::gpu::GpuMat &dst, int szK, int thresh, int maxval);
  void fillImgCUDAKernel(const cv::gpu::GpuMat &mask, cv::gpu::GpuMat &dst, int tlX, int tlY, int brX, int brY);
  void LUT(cv::gpu::GpuMat& lut, const double outByteDepth, const int minValue, const int maxValue);
  void stretching(const cv::gpu::GpuMat& src, const cv::gpu::GpuMat& lut, cv::gpu::GpuMat& dst);
};

#endif /* EXTERNALCLASS_CUH_ */

