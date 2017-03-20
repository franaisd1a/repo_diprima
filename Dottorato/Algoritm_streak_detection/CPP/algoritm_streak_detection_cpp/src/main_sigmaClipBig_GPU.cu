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
#include "externalClass.cu" // important to include .cu file, not header file

#include "../inc/function_GPU.h"
#include "../inc/function.h"
#include "../inc/macros.h"

using namespace cv;
using namespace std;

void main_sigmaClipBig_GPU
(
  const cv::Mat& histStretch
  , const cv::Mat& Img_input
  , std::ostream& infoFile
  , std::vector< cv::Vec<float, 3> >& POINTS
  , std::vector< cv::Vec<float, 3> >& STREAKS
)
{

/* ======================================================================= *
 * GPU initializations and informations                                    *
 * ======================================================================= */

  int deviceCount = gpu::getCudaEnabledDeviceCount();

  //gpu::setDevice(deviceCount-1);
  gpu::setDevice(deviceCount);
    
  // --- CUDA warm up
  gpu::GpuMat warmUp = gpu::createContinuous(2, 2, 0);

  /* Open file */
  FILE * pFile;
  pFile = fopen ("consoleGPU.txt","w");
   
  fprintf(pFile, "Device number %d\n", deviceCount);
    
  //Move data on GPU
  gpu::GpuMat histStretchGPU = gpu::createContinuous(histStretch.rows
    , histStretch.cols, histStretch.type());
  
  histStretchGPU.upload(histStretch);

  cv::Point_<int> I_input_size = { Img_input.cols, Img_input.rows };
  double bordersThick = 0.015;
  cv::Point_<double> borders = { bordersThick, 1 - bordersThick };
  cv::Vec<int, 4> imgBorders = { static_cast<int>(ceil(borders.x * I_input_size.x))
                          , static_cast<int>(ceil(borders.x * I_input_size.y))
                          , static_cast<int>(floor(borders.y * I_input_size.x))
                          , static_cast<int>(floor(borders.y * I_input_size.y)) };


/* ======================================================================= *
 * Points detection                                                        *
 * ======================================================================= */
  
/* ----------------------------------------------------------------------- *
 * Median filter                                                           *
 * ----------------------------------------------------------------------- */

  clock_t start = clock();
  
  gpu::GpuMat medianImgGPU = gpu::createContinuous(histStretch.rows
    , histStretch.cols, histStretch.type());
  
  int kerlen = 3;

  externalClass kernelCUDA;
  kernelCUDA.medianCUDAKernel(histStretchGPU, medianImgGPU, kerlen);
    
#if SPD_FIGURE_1
  cv::Mat result_hostMedian;
  medianImgGPU.download(result_hostMedian);
  // Create a window for display.
  namedWindow("Median filter GPU", cv::WINDOW_NORMAL);
  imshow("Median filter GPU", result_hostMedian);
#endif

	timeElapsed(infoFile, start, "Median filter");
  cv::waitKey(0);

  
/* ----------------------------------------------------------------------- *
 * Background estimation                                                   *
 * ----------------------------------------------------------------------- */

  start = clock();

  size_t maxColdim = 512;
  size_t maxRowdim = 512;

  int regionNumR = static_cast<int>(::round(histStretch.rows / maxRowdim));
  int regionNumC = static_cast<int>(::round(histStretch.cols / maxColdim));

  cv::Point backCnt = {regionNumC, regionNumR};
  cv::Mat meanBg = cv::Mat::zeros(backCnt.y, backCnt.x, CV_64F);
  cv::Mat  stdBg = cv::Mat::zeros(backCnt.y, backCnt.x, CV_64F);

  cv::gpu::GpuMat backgroungImg = 
    backgroundEstimation(medianImgGPU, backCnt, meanBg, stdBg);

  timeElapsed(infoFile, start, "Background estimation");
  cv::waitKey(0);


/* ----------------------------------------------------------------------- *
 * Background subtraction                                                  *
 * ----------------------------------------------------------------------- */

  start = clock();

  cv::gpu::GpuMat bgSubtracImg = subtractImage(medianImgGPU, backgroungImg);
    
  medianImgGPU.release();
  backgroungImg.release();
  
  timeElapsed(infoFile, start, "Background subtraction");

#if SPD_FIGURE_1
    namedWindow("Background subtraction", cv::WINDOW_NORMAL);
    imshow("Background subtraction", bgSubtracImg);
    cv::waitKey(0);
#endif
  

/* ----------------------------------------------------------------------- *
 * Median filter                                                           *
 * ----------------------------------------------------------------------- */

  clock_t start = clock();
  
  gpu::GpuMat medianBgSubImg = gpu::createContinuous(histStretch.rows
    , histStretch.cols, histStretch.type());
  
  kernelCUDA.medianCUDAKernel(bgSubtracImg, medianBgSubImg, kerlen);
    
#if SPD_FIGURE_1
  cv::Mat result_hostMedian;
  medianImgGPU.download(result_hostMedian);
  // Create a window for display.
  namedWindow("Median filter GPU", cv::WINDOW_NORMAL);
  imshow("Median filter GPU", result_hostMedian);
#endif

	timeElapsed(infoFile, start, "Median filter");
  cv::waitKey(0);


/* ----------------------------------------------------------------------- *
 * Binarization for points detection                                       *
 * ----------------------------------------------------------------------- */

  start = clock();

  cv::Mat level = cv::Mat::zeros(backCnt.y, backCnt.x, CV_64F);
  level = meanBg + 3.5*stdBg;
  
  gpu::GpuMat binaryImgPnt = binarizationZone(medianBgSubImg, backCnt, level);
  
  timeElapsed(infoFile, start, "Binarization");
  cv::waitKey(0);


/* ----------------------------------------------------------------------- *
 * Binarization for streaks detection                                      *
 * ----------------------------------------------------------------------- */

  start = clock();

  cv::Mat levelStk = cv::Mat::zeros(backCnt.y, backCnt.x, CV_64F);
  levelStk = meanBg + 1*stdBg;//2.8
  
  gpu::GpuMat binaryImgStk = binarizationZone(medianBgSubImg, backCnt, levelStk);
  medianBgSubImg.release();

  timeElapsed(infoFile, start, "Binarization for streaks detection");
  cv::waitKey(0);
  

/* ----------------------------------------------------------------------- *
 * Distance transformation for streaks detection                           *
 * ----------------------------------------------------------------------- */

  start = clock();
    
  cv::Mat binaryImgStk_cpu;
  binaryImgStk.download(binaryImgStk_cpu);

  cv::Mat distStk = distTransform(binaryImgStk_cpu);
  binaryImgStk.release();


/* ----------------------------------------------------------------------- *
 * Convolution kernel for points detection                                 *
 * ----------------------------------------------------------------------- */

  start = clock();

  int szKernel = 3;
  cv::Mat kernel = cv::Mat::ones(szKernel, szKernel, CV_8U);
  double threshConv = 7;//6
  
  gpu::GpuMat convImgPnt = convolution(binaryImgPnt, kernel, threshConv);
  binaryImgPnt.release();
  timeElapsed(infoFile, start, "Convolution for points detection");
  cv::waitKey(0);


/* ----------------------------------------------------------------------- *
 * Morphology opening                                                      *
 * ----------------------------------------------------------------------- */

  start = clock();

  int radDisk = 6;  
  gpu::GpuMat openImg = morphologyOpen(convImgPnt, radDisk);

  timeElapsed(infoFile, start, "Morphology opening");
  
  cv::waitKey(0);


/* ======================================================================= *
 * Streaks detection                                                       *
 * ======================================================================= */
  
/* ----------------------------------------------------------------------- *
 * Hough transform                                                         *
 * ----------------------------------------------------------------------- */

  start = clock();

  cv::Mat resizeImg;
  double f = 0.5;
  cv::Size dsize = { 0, 0 };
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
  cv::waitKey(0);


/* ----------------------------------------------------------------------- *
 * Sum streaks binary image                                                *
 * ----------------------------------------------------------------------- */

  start = clock();

  gpu::GpuMat sumStrRemImg = gpu::createContinuous(histStretch.rows
    , histStretch.cols, histStretch.type());
  //cv::Mat::zeros(histStretch.rows, histStretch.cols, CV_8U);

  for (int i = 0; i < angle.size(); ++i)
  {

/* ----------------------------------------------------------------------- *
 * Morphology opening with linear kernel for remove streaks                *
 * ----------------------------------------------------------------------- */
    
    int dimLineRem = 60;

    gpu::GpuMat morpOpLinRem = morphologyOpen(openImg, dimLineRem, angle.at(i).first);
    cv::waitKey(0);


/* ----------------------------------------------------------------------- *
 * Binary image with streaks                                               *
 * ----------------------------------------------------------------------- */

    sumStrRemImg = addiction(sumStrRemImg, morpOpLinRem);
    morpOpLinRem.release();
        
#if SPD_FIGURE_1
    namedWindow("sumStrRemImg", cv::WINDOW_NORMAL);
    imshow("sumStrRemImg", sumStrRemImg);
    cv::waitKey(0);
#endif
  }
  
  timeElapsed(infoFile, start, "Sum remove streaks binary");


/* ----------------------------------------------------------------------- *
 * Binary image without streaks                                            *
 * ----------------------------------------------------------------------- */
  
  gpu::GpuMat onlyPoints = subtractImage(openImg, sumStrRemImg);
  sumStrRemImg.release();
  openImg.release();

#if SPD_FIGURE_1
  namedWindow("onlyPoints", cv::WINDOW_NORMAL);
  imshow("onlyPoints", onlyPoints);
  cv::waitKey(0);
#endif


/* ----------------------------------------------------------------------- *
 * Convolution kernel remove streaks                                       *
 * ----------------------------------------------------------------------- */

  start = clock();

  cv::Mat kernelRm = cv::Mat::ones(szKernel, szKernel, CV_8U);
  double threshConvRm = 8;
  
  gpu::GpuMat convImgRms_GPU = gpu::createContinuous(histStretch.rows
    , histStretch.cols, histStretch.type());

  convImgRms_GPU = convolution(onlyPoints, kernelRm, threshConvRm);
  kernelRm.release();
  onlyPoints.release();
  cv::waitKey(0);

  timeElapsed(infoFile, start, "Convolution");
    

  /* ----------------------------------------------------------------------- *
 * Connected components                                                    *
 * ----------------------------------------------------------------------- */

  start = clock();

  std::vector<std::vector<cv::Point > > contoursP;
  std::vector<std::vector<cv::Point > > contoursS;

  cv::Mat convImgRms;
  convImgRms_GPU.download(convImgRms);

  connectedComponents2(convImgRms, distStk, Img_input
    , POINTS, STREAKS, contoursP, contoursS);

  distStk.release();
  convImgRms.release();

  timeElapsed(infoFile, start, "Connected components");


/* ----------------------------------------------------------------------- *
 * Light curve study                                                       *
 * ----------------------------------------------------------------------- */

  lightCurve(Img_input, STREAKS, contoursS);

}
