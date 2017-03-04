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
#include <limits>

#include <opencv2/opencv.hpp>
//#include <opencv\highgui.h>
#include <opencv/highgui.h>

#include <stdlib.h>
#include <fitsio.h>

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
* fileExt Get file extension
* @param nameFile Input file name
* @return File extension
*/
std::vector<char*> fileExt(const char* strN);

/**
* readFit Read .fit file and copy in opencv Mat
* @param nameFile Input file name
* @param stream Output stream
* @param img Output cv::Mat image
*/
void readFit(const char* nameFile, std::ostream& stream, cv::Mat& img);

/**
* histogramStretching Histogram Stretching
* @param imgIn Input image
* @return Output stretched image
*/
cv::Mat histogramStretching(const cv::Mat& imgIn);

cv::Mat subtraction(const cv::Mat& imgA, const cv::Mat& imgB);

/**
* gaussianFilter Filter an image using Gaussian lowpass filter
* @param imgIn Input image
* @param hsize Filter size
* @param sigma Filter standard deviation
* @return outImage Filtered image
*/
cv::Mat gaussianFilter(const cv::Mat& imgIn, int hsize[2], double sigma);

/**
* medianFilter Filter an image using median filter
* @param imgIn Input image
* @param kerlen Kernel size
* @return outImage Filtered image
*/
cv::Mat medianFilter(const cv::Mat& imgIn, int kerlen);

/**
* medianFilter Filter an image using the difference between two median filter
* @param imgIn Input image
* @param littleKerlen Little Kernel size
* @param bigKerlen Big Kernel size
* @return outImage Filtered image
*/
cv::Mat medianFilter(const cv::Mat& imgIn, int littleKerlen, int bigKerlen);

/**
* morphologyOpen Morphology opening on image with a rectangular kernel rotate of
* an angle. Delete noise and points object in the image and preserve the streaks
* @param imgIn Input image
* @param dimLine Line dimension
* @param teta_streak Line inclination angle
* @return outImage Morphology opening image
*/
cv::Mat morphologyOpen(const cv::Mat& imgIn, int dimLine, double teta);
cv::Mat morphologyOpen2(const cv::Mat& imgIn, int dimLine, double teta);

/**
* morphologyOpen Morphology opening on image with a circular kernel. 
* @param imgIn Input image
* @param rad Kernel radius
* @return outImage Morphology opening image
*/
cv::Mat morphologyOpen(const cv::Mat& imgIn, int rad);

/**
* backgroundEstimation 
* @param imgIn Input image
* @param backSz 
* @return outImage 
*/
cv::Mat backgroundEstimation
(
  const cv::Mat& imgIn
  , const cv::Point backCnt
  , cv::Mat& meanBg
  , cv::Mat& stdBg
);

cv::Mat binarizationZone
(
  const cv::Mat& imgIn
  , const cv::Point zoneCnt
  , const cv::Mat& level
);

/**
* binarization Image binarization using Otsu method
* @param imgIn Input image
* @return Binary image
*/
cv::Mat binarization(const cv::Mat& imgIn);

/**
* binarization Image binarization using user threshold
* @param imgIn Input image
* @param level User threshold
* @return Binary image
*/
cv::Mat binarization(const cv::Mat& imgIn, double level);
cv::Mat binarizationDiffTh(const cv::Mat& imgIn, int flag);

/**
* convolution Image convolution
* @param imgIn Input image
* @param kernel Convolution kernel
* @param threshold Threshold
* @return outImage Convolution images
*/
cv::Mat convolution(const cv::Mat& imgIn, const cv::Mat& kernel, double threshold);

cv::Mat distTransform(const cv::Mat& imgIn);

/**
* connectedComponents Found connected components
* @param imgPoints Input image for points detection
* @param imgStreaks Input image for streaks detection
* @param Img_input Original image
* @param borders Image borders
* @param POINTS Vector with points centroid
* @param STREAKS Vector with streaks centroid
*/
void connectedComponents
(
  const cv::Mat& imgPoints
  , const cv::Mat& imgStreaks
  , const cv::Mat& Img_input
  , const cv::Vec<int, 4>& borders  
  , std::vector< cv::Vec<float, 3> >& POINTS
  , std::vector< cv::Vec<float, 3> >& STREAKS
);
void connectedComponents2
(
  const cv::Mat& imgPoints
  , const cv::Mat& imgStreaks
  , const cv::Mat& Img_input
  , const cv::Vec<int, 4>& borders
  , std::vector< cv::Vec<float, 3> >& POINTS
  , std::vector< cv::Vec<float, 3> >& STREAKS
);

/**
* connectedComponentsPoints Found centroid of circular connected components
* @param max_img_sz Max image dimension
* @param contours Contours found by findContours
* @param borders Image borders
* @param outContoursRes Correct contours
* @return Vector with points centroid
*/
std::vector< cv::Vec<int, 3> > connectedComponentsPoints
(
  const float max_img_sz
  , const std::vector<std::vector<cv::Point > >& contours
  , const cv::Vec<int, 4>& borders
  , std::vector<std::vector<cv::Point > >& outContoursRes
);
std::vector< cv::Vec<int, 3> > connectedComponentsPoints2
(
  const float max_img_sz
  , const std::vector<std::vector<cv::Point > >& contours
  , const cv::Vec<int, 4>& borders
  , std::vector<std::vector<cv::Point > >& outContoursRes
);

/**
* connectedComponentsStreaks Found centroid of Streaks
* @param max_img_sz Max image dimension
* @param contours Contours found by findContours
* @param borders Image borders
* @param outContoursRes Correct contours
* @return Vector with streaks centroid
*/
std::vector< cv::Vec<int, 3> > connectedComponentsStreaks
(
  const float max_img_sz
  , const std::vector<std::vector<cv::Point > >& contours
  , const cv::Vec<int, 4>& borders
  , std::vector<std::vector<cv::Point > >& outContoursRes
);
std::vector< cv::Vec<int, 3> > connectedComponentsStreaks2
(
  const std::vector<std::vector<cv::Point > >& contours
  , const cv::Vec<int, 4>& borders
  , std::vector<std::vector<cv::Point > >& outContoursRes
  , std::vector< cv::RotatedRect >& rotRectV
);

/**
* deleteOverlapping Delete overlapping objects
* @param imgSz Image dimensions
* @param inPOINTS Input vector with points centroid
* @param inSTREAKS Input vector with streaks centroid
* @param contoursP Input points contours
* @param contoursS Input streaks contours
* @param outPOINTS Output vector with points centroid
* @param outSTREAKS Output vector with streaks centroid
*/
void deleteOverlapping
(
  const cv::Point imgSz
  , std::vector< cv::Vec<int, 3> >& inPOINTS
  , std::vector< cv::Vec<int, 3> >& inSTREAKS
  , const std::vector<std::vector<cv::Point > >& inContoursP
  , const std::vector<std::vector<cv::Point > >& inContoursS
  , std::vector< cv::Vec<int, 3> >& outPOINTS
  , std::vector< cv::Vec<int, 3> >& outSTREAKS
  , std::vector<std::vector<cv::Point > >& outContoursP
  , std::vector<std::vector<cv::Point> >& outContoursS
);


void preciseCentroid
(
  const cv::Mat& img
  , const std::vector<std::vector<cv::Point > >& contours
  , std::vector< cv::Vec<float, 3> >& center
);

/**
* rayCasting Ray Casting algorithm
* @param poly Input polygon
* @param poly Input point
* @return Boolean value true if the point is inside the polygon
*/
bool rayCasting
(
  const std::vector<cv::Point> & poly
  , const cv::Point& p
);

/**
* barycentre Compute baricentre position
* @param pixelIdList Object's pixel list
* @param p Object's baricentre
*/
void barycentre
(
  const cv::Mat& img
  , const std::vector<cv::Point> pixelIdList
  , cv::Point2f& p
);

/**
* hough Hough transform
* @param imgIn Input image
* @return Vector with lines incination angle
*/
std::vector<std::pair<float, int>> hough(const cv::Mat& imgIn);

/**
* timeElapsed Compute elapsed time
* @param stream Output stream
* @param start Start reference time
* @param strName String to plot
*/
void timeElapsed(std::ostream& stream, clock_t start, const char* strName);

/**
* linearKernel Create linear structural element
* @param dimLine Line dimension
* @param teta Line inclination angle
* @return Kernel with linear structural element
*/
cv::Mat linearKernel(int dimLine, double teta);

/**
* stamp Print on file and console the input string
* @param stream Output stream
* @param strName String to write
*/
void stamp(std::ostream& stream, const char* strName);

/**
* writeResult Print on file and console the result points and streaks centroid
* @param stream Output stream
* @param POINTS Vector with points centroid
* @param STREAKS Vector with streaks centroid
*/
void writeResult
(
  std::ostream& stream
  , const std::vector< cv::Vec<float, 3> >& STREAKS
  , const std::vector< cv::Vec<float, 3> >& POINTS
);

void plotResult
(
  const cv::Mat& imgIn
  , const std::vector< cv::Vec<float, 3> >& POINTS
  , const std::vector< cv::Vec<float, 3> >& STREAKS
  , const std::vector<char *>& input
);

void sigmaClipProcessing
(
  const cv::Mat& histStretch
  , const cv::Mat& Img_input
  , std::ostream& infoFile
  , std::vector< cv::Vec<float, 3> >& POINTS
  , std::vector< cv::Vec<float, 3> >& STREAKS
);



#endif /* FUNCTION_H */
