/* ==========================================================================
* ALGO STREAK DETECTION
* ========================================================================== */

/* ==========================================================================
* MODULE FILE NAME: main_GPU_cuda.cpp
*      MODULE TYPE:
*
*         FUNCTION: Detect streaks and points.
*          PURPOSE:
*    CREATION DATE: 20160727
*          AUTHORS: Francesco Diprima
*     DESIGN ISSUE: None
*       INTERFACES: None
*     SUBORDINATES: None.
*
*          HISTORY: See table below.
*
* 27-Jul-2016 | Francesco Diprima | 0.0 |
* Initial creation of this file.
*
* ========================================================================== */

/* ==========================================================================
* INCLUDES
* ========================================================================== */
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv/highgui.h>
#include "opencv2/gpu/gpu.hpp"
//#include "externalClass.cu" // important to include .cu file, not header file

//#include "../inc/function_GPU.h"
#include "../inc/function_GPU.cuh"
#include "../inc/function.h"

using namespace cv;
using namespace std;

void main_sigmaClipBig_GPU
(
  cv::gpu::GpuMat& histStretchGPU
  , const cv::Mat& Img_input
  , std::ostream& infoFile
  , std::vector< cv::Vec<float, 3> >& POINTS
  , std::vector< cv::Vec<float, 3> >& STREAKS
)
{
/* ======================================================================= *
 * Initializations                                                         *
 * ======================================================================= */

  double bordersThick = 0.015;  
  cv::Point imgSz ( Img_input.cols, Img_input.rows );
  int imgType = histStretchGPU.type();
  cv::Point_<double> borders ( bordersThick, 1 - bordersThick );
  cv::Vec<int, 4> imgBorders = { static_cast<int>(ceil(borders.x * imgSz.x))
                          , static_cast<int>(ceil(borders.x * imgSz.y))
                          , static_cast<int>(floor(borders.y * imgSz.x))
                          , static_cast<int>(floor(borders.y * imgSz.y)) };


/* ======================================================================= *
 * Points detection                                                        *
 * ======================================================================= */
  
/* ----------------------------------------------------------------------- *
 * Median filter                                                           *
 * ----------------------------------------------------------------------- */

  clock_t start = clock();

  int kerlen = 3;

  gpu::GpuMat medianImgGPU = medianFIlterK(histStretchGPU, kerlen);

	timeElapsed(infoFile, start, "Median filter");
  
  histStretchGPU.release();
  
/* ----------------------------------------------------------------------- *
 * Background estimation                                                   *
 * ----------------------------------------------------------------------- */

  start = clock();

  size_t maxColdim = 512;
  size_t maxRowdim = 512;

  int regionNumR = static_cast<int>(::round(static_cast<float>(imgSz.y / maxRowdim)));
  int regionNumC = static_cast<int>(::round(static_cast<float>(imgSz.x / maxColdim)));

  cv::Point backCnt (regionNumC, regionNumR);
  cv::Mat meanBg = cv::Mat::zeros(backCnt.y, backCnt.x, CV_64F);
  cv::Mat  stdBg = cv::Mat::zeros(backCnt.y, backCnt.x, CV_64F);

  cv::gpu::GpuMat backgroungImg = 
    backgroundEstimation(medianImgGPU, backCnt, meanBg, stdBg);

  timeElapsed(infoFile, start, "Background estimation");


/* ----------------------------------------------------------------------- *
 * Background subtraction                                                  *
 * ----------------------------------------------------------------------- */

  start = clock();

  cv::gpu::GpuMat bgSubtracImg = subtractImage(medianImgGPU, backgroungImg);
    
  timeElapsed(infoFile, start, "Background subtraction");
  
  medianImgGPU.release();
  backgroungImg.release();

/* ----------------------------------------------------------------------- *
 * Median filter                                                           *
 * ----------------------------------------------------------------------- */

  start = clock();

  gpu::GpuMat medianBgSubImg = medianFIlterK(bgSubtracImg, kerlen);
  bgSubtracImg.release();

	timeElapsed(infoFile, start, "Median filter");


/* ----------------------------------------------------------------------- *
 * Binarization for points detection                                       *
 * ----------------------------------------------------------------------- */

  start = clock();

  cv::Mat level = cv::Mat::zeros(backCnt.y, backCnt.x, CV_64F);
  level = meanBg + 3.5*stdBg;
  
  gpu::GpuMat binaryImgPnt = binarizationZone(medianBgSubImg, backCnt, level);
  
  timeElapsed(infoFile, start, "Binarization for points detection");


/* ----------------------------------------------------------------------- *
 * Binarization for streaks detection                                      *
 * ----------------------------------------------------------------------- */

  start = clock();

  cv::Mat levelStk = cv::Mat::zeros(backCnt.y, backCnt.x, CV_64F);
  levelStk = meanBg + 1*stdBg;//2.8
  
  gpu::GpuMat binaryImgStk = binarizationZone(medianBgSubImg, backCnt, levelStk);
  medianBgSubImg.release();

  timeElapsed(infoFile, start, "Binarization for streaks detection");
  

/* ----------------------------------------------------------------------- *
 * Distance transformation for streaks detection                           *
 * ----------------------------------------------------------------------- */

  start = clock();
    
  cv::Mat binaryImgStk_cpu;
  binaryImgStk.download(binaryImgStk_cpu);

  cv::Mat distStk = distTransform(binaryImgStk_cpu);
  binaryImgStk.release();
  binaryImgStk_cpu.release();


/* ----------------------------------------------------------------------- *
 * Convolution kernel for points detection                                 *
 * ----------------------------------------------------------------------- */

  start = clock();

  int szKernel = 3;
  int threshConv = 7;//6
  int maxvalConv = 255;

  gpu::GpuMat convImgPnt = convolution(binaryImgPnt, szKernel, threshConv, maxvalConv);
  binaryImgPnt.release();

  timeElapsed(infoFile, start, "Convolution for points detection");


/* ----------------------------------------------------------------------- *
 * Morphology opening                                                      *
 * ----------------------------------------------------------------------- */

  start = clock();

  int radDisk = 6;  
  gpu::GpuMat openImg = morphologyOpen(convImgPnt, radDisk);
  convImgPnt.release();

  timeElapsed(infoFile, start, "Morphology opening");
    

/* ======================================================================= *
 * Streaks detection                                                       *
 * ======================================================================= */
  
/* ----------------------------------------------------------------------- *
 * Hough transform                                                         *
 * ----------------------------------------------------------------------- */

  start = clock();

  cv::Mat resizeImg;
  double f = 0.5;
  cv::Size dsize ( 0, 0 );
  resize(distStk, resizeImg, dsize, f, f, cv::INTER_LINEAR);

  std::vector<std::pair<float, int>> angle = hough(resizeImg);
  resizeImg.release();
  
  std::string s_nH = "Number of inclination angles: " + std::to_string(angle.size());
  stamp(infoFile, s_nH.c_str());

  for (size_t a = 0; a < angle.size(); ++a)
  {
    std::string s_vA = "Angle: " + std::to_string(angle.at(a).first) 
      + " " + std::to_string(angle.at(a).second);
    stamp(infoFile, s_vA.c_str());
  }

  timeElapsed(infoFile, start, "Hough transform");
  
#if 0
/*************************************************/
  //Move data on GPU
  cv::gpu::GpuMat resizeGPU = gpu::createContinuous(resizeImg.rows
    , resizeImg.cols, resizeImg.type());
  
  resizeGPU.upload(resizeImg);

  std::vector<std::pair<float, int>> asd = hough(resizeGPU);
  resizeGPU.release();
/*************************************************/
#endif
/* ----------------------------------------------------------------------- *
 * Sum streaks binary image                                                *
 * ----------------------------------------------------------------------- */

  start = clock();

  gpu::GpuMat sumStrRemImg = gpu::createContinuous(imgSz.y, imgSz.x, imgType);

  for (int i = 0; i < angle.size(); ++i)
  {

/* ----------------------------------------------------------------------- *
 * Morphology opening with linear kernel for remove streaks                *
 * ----------------------------------------------------------------------- */
    
    int dimLineRem = 60;

    gpu::GpuMat morpOpLinRem = morphologyOpen(openImg, dimLineRem, angle.at(i).first);
    

/* ----------------------------------------------------------------------- *
 * Binary image with streaks                                               *
 * ----------------------------------------------------------------------- */

    sumStrRemImg = addiction(sumStrRemImg, morpOpLinRem);
    morpOpLinRem.release();
        
#if SPD_FIGURE_1
    cv::Mat sumStrRemImg_host;
    sumStrRemImg.download(sumStrRemImg_host);
    namedWindow("sumStrRemImg", cv::WINDOW_NORMAL);
    imshow("sumStrRemImg", sumStrRemImg_host);
    cv::waitKey(0);
#endif
  }
  
  timeElapsed(infoFile, start, "Sum remove streaks binary");


/* ----------------------------------------------------------------------- *
 * Binary image without streaks                                            *
 * ----------------------------------------------------------------------- */
  
  start = clock();

  gpu::GpuMat onlyPoints = subtractImage(openImg, sumStrRemImg);
  sumStrRemImg.release();
  openImg.release();

  timeElapsed(infoFile, start, "Subtract image");


/* ----------------------------------------------------------------------- *
 * Convolution kernel remove streaks                                       *
 * ----------------------------------------------------------------------- */

  start = clock();
  
  int threshConvRm = 8;
   
  gpu::GpuMat convImgRms_GPU = convolution(onlyPoints, szKernel, threshConvRm, maxvalConv);
  onlyPoints.release();

  timeElapsed(infoFile, start, "Convolution remove streaks");
    

  /* ----------------------------------------------------------------------- *
 * Connected components                                                    *
 * ----------------------------------------------------------------------- */

  start = clock();

  std::vector<std::vector<cv::Point > > contoursP;
  std::vector<std::vector<cv::Point > > contoursS;

  cv::Mat convImgRms;
  convImgRms_GPU.download(convImgRms);
  convImgRms_GPU.release();

  connectedComponents2(convImgRms, distStk, Img_input
    , POINTS, STREAKS, contoursP, contoursS);

  distStk.release();
  convImgRms.release();

  timeElapsed(infoFile, start, "Connected components");


/* ----------------------------------------------------------------------- *
 * Light curve study                                                       *
 * ----------------------------------------------------------------------- */

  //lightCurve(Img_input, STREAKS, contoursS);

}
