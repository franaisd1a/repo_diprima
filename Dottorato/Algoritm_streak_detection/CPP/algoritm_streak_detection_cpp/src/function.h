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

#ifndef FUNCTION_H
#define FUNCTION_H

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
#include <time.h>

#include <opencv2/opencv.hpp>
//#include <opencv\highgui.h>
#include <opencv/highgui.h>

#include <stdlib.h>
#include <fitsio.h>

/* ==========================================================================
* MACROS
* ========================================================================== */
/*#define FIGURE 1U
#define FIGURE_1 1U
#define FILE_READ 1U
#define CLEAR 0U
#define BACKGROUND_SUBTRACTION 1U
#define DIFFERENT_THRESHOLD 1U
#define FIT 1U
#define DILATE 1U*/

/* ==========================================================================
* CLASS DECLARATION
* ========================================================================== */

/* ==========================================================================
* FUNCTION DECLARATION
* ========================================================================== */

/**
* fileExt Get file extension
* @param nameFile Input file name
* @param img cv::Mat image
*/
char* fileExt(const char* nameFile);

/**
* readFit Read .fit file and write in opencv Mat
* @param nameFile Input file name
* @param img cv::Mat image
*/
void readFit(char* nameFile, cv::Mat& img);

/**
* histogramStretching Histogram Stretching
* @param imgIn Input image
*/
cv::Mat histogramStretching(cv::Mat& imgIn);

/**
* gaussianFilter Filter an image using Gaussian lowpass filter
* @param imgIn Input image
* @param hsize Filter size
* @param sigma Filter standard deviation
* @return outImage Filtered image
*/
cv::Mat gaussianFilter(cv::Mat& imgIn, int hsize[2], double sigma);

/**
* gaussianFilter Filter an image using median filter
* @param imgIn Input image
* @param kerlen Little Kernel
* @return outImage Filtered image
*/
cv::Mat medianFilter(cv::Mat& imgIn, int kerlen);

/**
* gaussianFilter Filter an image using the difference between two median filter
* @param imgIn Input image
* @param littleKerlen Little Kernel
* @param bigKerlen Big Kernel
* @return outImage Filtered image
*/
cv::Mat medianFilter(cv::Mat& imgIn, int littleKerlen, int bigKerlen);

/**
* morphologyOpen Morphology opening on image with a rectangular kernel rotate of
* an angle. Delete noise and points object in the image and preserve the streaks
* @param imgIn Input image
* @param dimLine Line dimension
* @param teta_streak Line inclination angle
* @return outImage Morphology opening image
*/
cv::Mat morphologyOpen(cv::Mat& imgIn, int dimLine, double teta_streak);
cv::Mat morphologyOpen(cv::Mat& imgIn, int rad);

/**
* binarization Image binarization
* @param imgIn Input image
* @param dimLine Line dimension
* @param teta_streak Line inclination angle
* @return outImage Morphology opening image
*/
cv::Mat binarization(cv::Mat& imgIn);
cv::Mat binarization(cv::Mat& imgIn, double level);
cv::Mat binarizationDiffTh(cv::Mat& imgIn, int flag);

/**
* convolution Convolution images
* @param imgIn Input image
* @param kernel Convolution kernel
* @param threshold Threshold
* @return outImage Convolution images
*/
cv::Mat convolution(cv::Mat& imgIn, cv::Mat& kernel, double threshold);

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
std::vector<std::pair<float, int>> hough(cv::Mat& imgIn);

/**
* timeElapsed Compute elapsed time
* @param start Start reference time
* @param strName String to plot
* @return Elapsed time
*/
void timeElapsed(clock_t start, const char* strName);

#endif /* FUNCTION_H */


