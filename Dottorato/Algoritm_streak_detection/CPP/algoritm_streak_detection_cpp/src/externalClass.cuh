#ifndef EXTERNALCLASS_CUH_
#define EXTERNALCLASS_CUH_

#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv/highgui.h>
#include "opencv2/gpu/gpu.hpp"

/*
__global__ void square_array(double *a, int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx<N) a[idx] = a[idx] * a[idx];
	printf("GPU PRINT: idx = %d, a = %f\n", idx, a[idx]);
}
*/

__global__ void medianKernel
(const uchar * input, uchar * output, int Image_Width, int Image_Height)
{
  unsigned short surround[9];

  int iterator;

  const int x     = blockDim.x * blockIdx.x + threadIdx.x;
  const int y     = blockDim.y * blockIdx.y + threadIdx.y;
  //const int tid   = threadIdx.y * blockDim.x + threadIdx.x;   

  if( (x >= (Image_Width - 1)) || (y >= Image_Height - 1) || (x == 0) || (y == 0)) return;

  // --- Fill array private to the threads
  iterator = 0;
  for (int r = x - 1; r <= x + 1; r++) {
      for (int c = y - 1; c <= y + 1; c++) {
          surround[iterator] = input[c*Image_Width+r];
          //surround[iterator] = Input_Image[c*Image_Width+r];
          iterator++;
      }
  }

  // --- Sort private array to find the median using Bubble Short
  for (int i=0; i<5; ++i) {

      // --- Find the position of the minimum element
      int minval=i;
      for (int l=i+1; l<9; ++l) if (surround[l] < surround[minval]) minval=l;

      // --- Put found minimum element in its place
      unsigned short temp = surround[i];
      surround[i]=surround[minval];
      surround[minval]=temp;
  }

  // --- Pick the middle one
  //Output_Image[(y*Image_Width)+x]=surround[4];
  output[(y*Image_Width)+x]=surround[4];
}

class externalClass {

public:
	int GetInt() { return 5; };

	//void squareOnDevice(double *a_h, const int N);
	
	void cudaKernel(const cv::gpu::GpuMat &src, cv::gpu::GpuMat &dst);
};

#endif