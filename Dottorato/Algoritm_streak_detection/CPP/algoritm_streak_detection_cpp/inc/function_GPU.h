/* ==========================================================================
* PK SPACEKIT LIBRARY
* (C) 2016 PLANETEK Italia SRL
* ========================================================================== */

/* ==========================================================================
*   INCLUDE FILE NAME: function.h
* INCLUDE DESCRIPTION: Function for image elaboration.
*       CREATION DATE: 20160727
*             AUTHORS: Francesco Diprima
*        DESIGN ISSUE: None.
*
*             HISTORY: See table below.
*
* 27-Jul-2016 | Francesco Diprima | 0.0 |
* Initial creation of this file.
*
* ========================================================================== */

#ifndef FUNCTION_GPU_H
#define FUNCTION_GPU_H

/* ==========================================================================
* INCLUDE: Basic include file.
* ========================================================================== */
#include <vector>
//#include <cstdint>
#include <stdint.h>
#include <iostream>
#include <math.h>
#include <string.h>
#include <numeric>

#include <opencv2/opencv.hpp>
//#include <opencv\highgui.h>
#include <opencv/highgui.h>
#include "opencv2/gpu/gpu.hpp"

#if 0
#include </usr/local/cuda-6.5/include/cuda.h>
#include </usr/local/cuda-6.5/include/cuda_runtime_api.h>
#endif

#include <cuda.h>
#include <cuda_runtime_api.h>

/* ==========================================================================
* MACROS
* ========================================================================== */

/* ==========================================================================
* CLASS DECLARATION
* ========================================================================== */

/* ==========================================================================
* FUNCTION DECLARATION
* ========================================================================== */

cv::gpu::GpuMat backgroundEstimation(const cv::gpu::GpuMat& imgInOr
  , const cv::Point backCnt, cv::Mat& meanBg, cv::Mat& stdBg);

/**
* gaussianFilter Filter an image using Gaussian lowpass filter
* @param imgIn Input image
* @param hsize Filter size
* @param sigma Filter standard deviation
* @return outImage Filtered image
*/
cv::gpu::GpuMat gaussianFilter(const cv::gpu::GpuMat& imgIn, int hsize[2], double sigma);

/**
* subtractImage Subtraction of image, matrix-matrix difference
* @param imgA Input image A
* @param imgB Input image B
* @return outImage Subtracted image
*/
cv::gpu::GpuMat subtractImage(const cv::gpu::GpuMat& imgA, const cv::gpu::GpuMat& imgB);

cv::gpu::GpuMat addiction(const cv::gpu::GpuMat& imgA, const cv::gpu::GpuMat& imgB);
/**
* morphologyOpen Morphology opening on image with a rectangular kernel rotate of
* an angle. Delete noise and points object in the image and preserve the streaks
* @param imgIn Input image
* @param dimLine Line dimension
* @param teta_streak Line inclination angle
* @return outImage Morphology opening image
*/
cv::gpu::GpuMat morphologyOpen(const cv::gpu::GpuMat& imgIn, int dimLine, double teta_streak);
cv::gpu::GpuMat morphologyOpen(const cv::gpu::GpuMat& imgIn, int rad);

/**
* binarization Image binarization
* @param imgIn Input image
* @param dimLine Line dimension
* @param teta_streak Line inclination angle
* @return outImage Morphology opening image
*/
cv::gpu::GpuMat binarization(const cv::gpu::GpuMat& imgIn);
cv::gpu::GpuMat binarizationZone(const cv::gpu::GpuMat& imgIn
  , const cv::Point zoneCnt, const cv::Mat& level);
#if 0
cv::gpu::GpuMat binarizationDiffTh(cv::gpu::GpuMat& imgIn, int flag);
#endif

std::vector<std::pair<float, int>> hough(const cv::gpu::GpuMat& imgIn);

/**
* convolution Convolution images
* @param imgIn Input image
* @param kernel Convolution kernel
* @param threshold Threshold
* @return outImage Convolution images
*/
cv::gpu::GpuMat convolution(const cv::gpu::GpuMat& imgIn, const cv::Mat& kernel, double threshold);

/**
* callKernel Function for call CUDA kernel
* @param imgA Input image A
* @param imgB Output image B
*/
//void callKernel(const cv::gpu::GpuMat &src, cv::gpu::GpuMat &dst);

/**
* iDivUp Rounded division
* @param a Input a
* @param b Output b
* @return Rounded division
*/
int iDivUp(int a, int b);

//void cudaKernel(const cv::gpu::GpuMat &src, cv::gpu::GpuMat &dst);

#endif /* FUNCTION_GPU_H */
