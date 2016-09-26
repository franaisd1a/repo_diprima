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
#include <cstdint>
#include <iostream>
#include <math.h>
#include <string.h>
#include <numeric>

#include <opencv2/opencv.hpp>
//#include <opencv\highgui.h>
#include <opencv/highgui.h>
#include "opencv2/gpu/gpu.hpp"

/* ==========================================================================
* MACROS
* ========================================================================== */

/* ==========================================================================
* CLASS DECLARATION
* ========================================================================== */

/* ==========================================================================
* FUNCTION DECLARATION
* ========================================================================== */

/**
* gaussianFilter Filter an image using Gaussian lowpass filter
* @param imgIn Input image
* @param hsize Filter size
* @param sigma Filter standard deviation
* @return outImage Filtered image
*/
cv::gpu::GpuMat gaussianFilter(cv::gpu::GpuMat& imgIn, int hsize[2], double sigma);

/**
* subtractImage Subtraction of image, matrix-matrix difference
* @param imgA Input image A
* @param imgB Input image B
* @return outImage Subtracted image
*/
cv::gpu::GpuMat subtractImage(cv::gpu::GpuMat& imgA, cv::gpu::GpuMat& imgB);

/**
* morphologyOpen Morphology opening on image with a rectangular kernel rotate of
* an angle. Delete noise and points object in the image and preserve the streaks
* @param imgIn Input image
* @param dimLine Line dimension
* @param teta_streak Line inclination angle
* @return outImage Morphology opening image
*/
cv::gpu::GpuMat morphologyOpen(cv::gpu::GpuMat& imgIn, int dimLine, double teta_streak);

/**
* binarization Image binarization
* @param imgIn Input image
* @param dimLine Line dimension
* @param teta_streak Line inclination angle
* @return outImage Morphology opening image
*/
cv::gpu::GpuMat binarization(cv::gpu::GpuMat& imgIn);
#if 0
cv::gpu::GpuMat binarizationDiffTh(cv::gpu::GpuMat& imgIn, int flag);
#endif

/**
* convolution Convolution images
* @param imgIn Input image
* @param kernel Convolution kernel
* @param threshold Threshold
* @return outImage Convolution images
*/
cv::gpu::GpuMat convolution(cv::gpu::GpuMat& imgIn, cv::Mat& kernel, double threshold);

#if 0
/**
* connectedComponents Found connected components
* @param imgIn Input image
* @param borders Image borders
* @return 
*/
std::vector< cv::Vec<int, 3> > connectedComponents
(
  cv::Mat& imgIn
  , cv::Vec<int, 4>& borders
  , std::vector< cv::Vec<int, 3> >& POINTS
  , std::vector< cv::Vec<int, 3> >& STREAKS
);

/**
* connectedComponents Found centroid of circular connected components
* @param imgIn Input image
* @param contours Contours found by findContours function
* @param borders Image borders
* @return Vector with circular connect componets coordinates
*/
std::vector< cv::Vec<int, 3> > connectedComponentsPoints
(
  cv::Mat& imgIn
  , std::vector<std::vector<cv::Point> >& contours
  , cv::Vec<int, 4>& borders
);

/**
* connectedComponentsStreaks Found centroid of Streaks
* @param imgIn Input image
* @param contours Contours found by findContours function
* @param borders Image borders
* @return Vector with streaks coordinates
*/
std::vector< cv::Vec<int, 3> > connectedComponentsStreaks
(
  cv::Mat& imgIn
  , std::vector<std::vector<cv::Point> >& contours
  , cv::Vec<int, 4>& borders
);

/**
* hough Hough transform
* @param imgIn Input image
* @return outImage 
*/
cv::Mat hough(cv::Mat& imgIn);
#endif

#endif /* FUNCTION_GPU_H */

