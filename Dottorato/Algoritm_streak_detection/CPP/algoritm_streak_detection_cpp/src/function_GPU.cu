/* ==========================================================================
* ALGO STREAK DETECTION
* ========================================================================== */

/* ==========================================================================
* MODULE FILE NAME: function.cpp
*      MODULE TYPE: 
*
*         FUNCTION: Function for image elaboration.
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
/*#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv/highgui.h>
#include "opencv2/gpu/gpu.hpp"*/


#include "../inc/function_GPU.cuh"
#include "../inc/macros.h"
#include "externalClass.cu" // important to include .cu file, not header file

/* ==========================================================================
* MODULE PRIVATE MACROS
* ========================================================================== */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

/* ==========================================================================
* MODULE PRIVATE TYPE DECLARATIONS
* ========================================================================== */

/* ==========================================================================
* STATIC VARIABLES FOR MODULE
* ========================================================================== */

/* ==========================================================================
* STATIC MEMBERS
* ========================================================================== */

/* ==========================================================================
* NAME SPACE
* ========================================================================== */

/* ==========================================================================
*        FUNCTION NAME: histogramStreching
* FUNCTION DESCRIPTION: Histogram streching on GPU
*        CREATION DATE: 20170422
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
cv::gpu::GpuMat histogramStreching
(
  const cv::Mat& imgIn
)
{
  //int depth = imgIn.depth();
  int type = imgIn.type();
  double outByteDepth = 255.0;

  int color = 0;

  if (0 == type) {
    color = static_cast<int>(::pow(2, 8));
  } else if (2 == type) {
    color = static_cast<int>(::pow(2, 16));
  } else {
    printf("Error. Unsupported pixel type.\n");
  }

  externalClass kernelCUDA;
  int imgRows = imgIn.rows;
  int imgCols = imgIn.cols;
  
  /*************************** Histogram computation **************************/
  
  cv::gpu::GpuMat imgInGPU = cv::gpu::createContinuous(imgRows, imgCols, imgIn.type());
  
  imgInGPU.upload(imgIn);

  cv::gpu::GpuMat hist = cv::gpu::createContinuous(1, color, CV_32S);

  size_t maxColdim = 2048;
  size_t maxRowdim = 2048;

  size_t regionNumR = static_cast<size_t>(::round(static_cast<float>(imgRows / maxRowdim)));
  if (0 == regionNumR) { regionNumR = 1; }
  size_t regionNumC = static_cast<size_t>(::round(static_cast<float>(imgCols / maxColdim)));
  if (0 == regionNumC) { regionNumC = 1; }

  size_t regionDimR = static_cast<size_t>(::round(static_cast<float>(imgRows / regionNumR)));
  size_t regionDimC = static_cast<size_t>(::round(static_cast<float>(imgCols / regionNumC)));

  std::vector<int> vRegRow;
  std::vector<int> vRegCol;

  for (size_t o = 0; o < regionNumR; ++o)
  {
    vRegRow.push_back(regionDimR*o);
  }
  vRegRow.push_back(imgRows);

  for (size_t o = 0; o < regionNumC; ++o)
  {
    vRegCol.push_back(regionDimC*o);
  }
  vRegCol.push_back(imgCols);

  for (size_t i = 0; i < regionNumR; ++i)
  {
    for (size_t j = 0; j < regionNumC; ++j)
    {
      const cv::Point ptTL ( vRegCol.at(j), vRegRow.at(i) );
      const cv::Point ptBR ( vRegCol.at(j + 1)-1, vRegRow.at(i + 1)-1);

      cv::Rect region_of_interest = cv::Rect(ptTL, ptBR);      
      cv::gpu::GpuMat imgInGPUPart = imgInGPU(region_of_interest);

      kernelCUDA.histogram(imgInGPUPart, hist);
    }
  }
  
#if SPD_DEBUG
  cv::Mat histHost;
  hist.download(histHost);
  //Print hist value
  const int* pLine = histHost.ptr<int>(0);
  for (int row = 0; row < 10240; ++row)
  {
    printf("BIN=%d value=%d\n", row, pLine[row]);
  }
#endif

  /*********************** Compute limits *************************************/

  double maxHistValue = 0, minHistValue = 0;
  cv::Point minLocHistValue (0, 0);
  cv::Point maxLocHistValue (0, 0);
  
  {
    cv::gpu::GpuMat buf;
    cv::gpu::minMaxLoc(imgInGPU, &minHistValue, &maxHistValue, &minLocHistValue, &maxLocHistValue, buf);
    buf.release();
  }

  double peakMax = 0, peakMin = 0;
  cv::Point peakMinLoc (0, 0);
  cv::Point peakMaxLoc (0, 0);

  {
    cv::gpu::GpuMat buf;
    cv::gpu::minMaxLoc(hist, &peakMin, &peakMax, &peakMinLoc, &peakMaxLoc, buf);
    buf.release();
  }
  
  const double percentile[2] = { 0.432506, (1 - 0.97725) };
  double  lowThresh = peakMax * percentile[0];
  double highThresh = peakMax * percentile[1];

  int minValue = 0;
  int maxValue = 0;

  kernelCUDA.lowerLimKernel(hist, peakMaxLoc.x, lowThresh, minValue);

  kernelCUDA.upperLimKernel(hist, peakMaxLoc.x, highThresh, maxValue);

  hist.release();
  
  //printf("Lower limit=%d Upper limit=%d\n", minValue, maxValue);

  /*************************** Stretching *************************************/

  cv::gpu::GpuMat LUT = cv::gpu::createContinuous(1, color, CV_32F);

  kernelCUDA.LUT(LUT, outByteDepth, minValue, maxValue);

  cv::gpu::GpuMat imgOut = cv::gpu::createContinuous(imgIn.rows, imgIn.cols, CV_8U);

  kernelCUDA.stretching(imgInGPU, LUT, imgOut);

  imgInGPU.release();
  LUT.release();
  
#if SPD_FIGURE_1
  cv::Mat imgOut_host;
  imgOut.download(imgOut_host);
  // Create a window for display.
  namedWindow("Histogram streching GPU", cv::WINDOW_NORMAL);
  imshow("Histogram streching GPU", imgOut_host);
  cv::waitKey(0);
#endif

  return imgOut;
}

/* ==========================================================================
*        FUNCTION NAME: histogram
* FUNCTION DESCRIPTION: Histogram on GPU
*        CREATION DATE: 20170422
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
cv::Mat histogram
(
  const cv::Mat& imgIn
)
{
  cv::gpu::GpuMat imgInGPU = cv::gpu::createContinuous(imgIn.rows, imgIn.cols, imgIn.type());
  
  imgInGPU.upload(imgIn);

  int histSz = static_cast<int>(::pow(2,16));
  cv::gpu::GpuMat hist = cv::gpu::createContinuous(1, histSz, CV_32S);

  externalClass kernelCUDA;
  kernelCUDA.histogram(imgInGPU, hist);
  
  cv::Mat histHost;
  hist.download(histHost);

#if 0
  //Print matrix value
  const int* pLine = histHost.ptr<int>(0);
  for (int row = 0; row < 10240; ++row)
  {
    printf("BIN=%d value=%d\n", row, pLine[row]);
  }
#endif

  imgInGPU.release();

  return histHost;
}

/* ==========================================================================
*        FUNCTION NAME: streching
* FUNCTION DESCRIPTION: Histogram streching on GPU
*        CREATION DATE: 20170422
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
cv::gpu::GpuMat streching
(
  const cv::Mat& imgIn
  , const cv::Mat& hist
  , const double outByteDepth
  , const int minValue
  , const int maxValue)
{
//  double outputByteDepth = 255.0;
  cv::gpu::GpuMat LUT = cv::gpu::createContinuous(1, hist.cols, hist.type());

  externalClass kernelCUDA;
  kernelCUDA.LUT(LUT, outByteDepth, minValue, maxValue);

  cv::gpu::GpuMat imgInGPU = cv::gpu::createContinuous(imgIn.rows, imgIn.cols, imgIn.type());
  
  imgInGPU.upload(imgIn);

  cv::gpu::GpuMat imgOut = cv::gpu::createContinuous(imgIn.rows, imgIn.cols, CV_8U);

  kernelCUDA.stretching(imgInGPU, LUT, imgOut);

  imgInGPU.release();
  LUT.release();
  
#if SPD_FIGURE_1
  cv::Mat imgOut_host;
  imgOut.download(imgOut_host);
  // Create a window for display.
  namedWindow("Histogram streching GPU", cv::WINDOW_NORMAL);
  imshow("Histogram streching GPU", imgOut_host);
  cv::waitKey(0);
#endif

  return imgOut;
}

/* ==========================================================================
*        FUNCTION NAME: medianFIlterK
* FUNCTION DESCRIPTION: Median filter on GPU
*        CREATION DATE: 20160727
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
cv::gpu::GpuMat medianFIlterK(const cv::gpu::GpuMat& imgIn, int kerlen)
{
  cv::gpu::GpuMat imgOut = cv::gpu::createContinuous(imgIn.rows, imgIn.cols
    , imgIn.type());

  externalClass kernelCUDA;
  kernelCUDA.medianCUDAKernel(imgIn, imgOut, kerlen);

#if SPD_FIGURE_1
  cv::Mat imgOut_host;
  imgOut.download(imgOut_host);
  // Create a window for display.
  namedWindow("Median filter GPU", cv::WINDOW_NORMAL);
  imshow("Median filter GPU", imgOut_host);
  cv::waitKey(0);
#endif

  return imgOut;
} 


/* ==========================================================================
*        FUNCTION NAME: backgroundEstimation
* FUNCTION DESCRIPTION: Background Estimation
*        CREATION DATE: 20160727
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
cv::gpu::GpuMat backgroundEstimation(const cv::gpu::GpuMat& imgIn, const cv::Point backCnt, cv::Mat& meanBg, cv::Mat& stdBg)
{
  //cv::gpu::GpuMat imgIn = imgInOr.clone();
  size_t backSzR = static_cast<size_t>(::round(static_cast<float>(imgIn.rows / backCnt.y)));
  size_t backSzC = static_cast<size_t>(::round(static_cast<float>(imgIn.cols / backCnt.x)));
  
  std::vector<int> vBackSrow;
  std::vector<int> vBackScol;

  for (size_t o = 0; o < backCnt.y; ++o)
  {
    vBackSrow.push_back(backSzR*o);
  }
  vBackSrow.push_back(imgIn.rows);

  for (size_t o = 0; o < backCnt.x; ++o)
  {
    vBackScol.push_back(backSzC*o);
  }
  vBackScol.push_back(imgIn.cols);
    
  cv::gpu::GpuMat outImg = cv::gpu::createContinuous(imgIn.rows, imgIn.cols, imgIn.type());

  for (size_t i = 0; i < backCnt.y; ++i)
  {
    for (size_t j = 0; j < backCnt.x; ++j)
    {
      const cv::Point ptTL (vBackScol.at(j), vBackSrow.at(i));
      const cv::Point ptBR (vBackScol.at(j+1), vBackSrow.at(i+1));

      cv::Rect region_of_interest = cv::Rect(ptTL, ptBR);
      cv::gpu::GpuMat imgPart = imgIn(region_of_interest);

      cv::gpu::GpuMat imgPartTh = cv::gpu::createContinuous(imgPart.rows, imgPart.cols, imgPart.type());
      //cv::gpu::GpuMat imgPart2 = cv::gpu::createContinuous(imgPart.rows, imgPart.cols, imgPart.type());
      
      float oldStd=0;
      float diffPercStd = 1;

      cv::Scalar bg_mean = 0;
      cv::Scalar bg_std = 0;
      cv::gpu::meanStdDev(imgPart, bg_mean, bg_std);

      meanBg.at<double>(i, j) = bg_mean.val[0];
      //meanBg.at<double>(i,j) = *(cv::mean(imgPart, cv::noArray()).val); Not exist mean for gpu

      while (diffPercStd>0.2f)
      {
        cv::Scalar meanBGmod = 0;
        cv::Scalar stdBGs = 0;
        cv::gpu::meanStdDev(imgPart, meanBGmod, stdBGs);
       
        stdBg.at<double>(i,j) = stdBGs.val[0];//*(stdBGs.val);

        double threshH = meanBg.at<double>(i,j)+2.5*stdBg.at<double>(i,j);//3

        double maxval = 1.0;
        double asdf = cv::gpu::threshold(imgPart, imgPartTh, threshH, maxval, cv::THRESH_TOZERO_INV);

/*        int dType = -1;
        cv::gpu::multiply(imgPart, imgPartTh, imgPart2, maxval, dType, cv::gpu::Stream::Null());*/

        diffPercStd = ::abs((stdBg.at<double>(i,j)-oldStd)/stdBg.at<double>(i,j));
        oldStd=stdBg.at<double>(i,j);        
      }
      
      externalClass kernelCUDA;
      kernelCUDA.fillImgCUDAKernel(imgPartTh, outImg, ptTL.x, ptTL.y, ptBR.x, ptBR.y);//imgPart2
    }
  }

#if SPD_FIGURE_1
  cv::Mat result_host;
  outImg.download(result_host);
  // Create a window for display.
  namedWindow("Background estimation", cv::WINDOW_NORMAL);
  imshow("Background estimation", result_host);
  cv::waitKey(0);
#endif

  return outImg;
}

/* ==========================================================================
*        FUNCTION NAME: gaussianFilter
* FUNCTION DESCRIPTION: Gaussian lowpass filter
*        CREATION DATE: 20160727
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
cv::gpu::GpuMat gaussianFilter(const cv::gpu::GpuMat& imgIn, int hsize[2], double sigma)
{
  //cv::gpu::GpuMat imgOut;
  cv::gpu::GpuMat imgOut = 
    cv::gpu::createContinuous(imgIn.rows, imgIn.cols, imgIn.type());

  cv::Size h ( hsize[0], hsize[1] );

  int columnBorderType=-1;
  cv::gpu::GaussianBlur(imgIn, imgOut, h, sigma, sigma, cv::BORDER_DEFAULT, columnBorderType);
  
#if SPD_FIGURE_1
  cv::Mat result_host;
  imgOut.download(result_host);
  // Create a window for display.
  namedWindow("Gaussain filter GPU", cv::WINDOW_NORMAL);
  imshow("Gaussain filter GPU", result_host);
  cv::waitKey(0);
#endif

  return imgOut;
}

/* ==========================================================================
*        FUNCTION NAME: subtractImage
* FUNCTION DESCRIPTION: Subtraction of images, matrix-matrix difference.
*        CREATION DATE: 20160727
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
cv::gpu::GpuMat subtractImage(const cv::gpu::GpuMat& imgA, const cv::gpu::GpuMat& imgB)
{
  cv::gpu::GpuMat imgOut = 
    cv::gpu::createContinuous(imgA.rows, imgA.cols, imgA.type());
      
  cv::gpu::subtract(imgA, imgB, imgOut);
  
#if SPD_FIGURE_1
  cv::Mat result_host;
  imgOut.download(result_host);
  // Create a window for display.
  namedWindow("Subtracted image GPU", cv::WINDOW_NORMAL);
  imshow("Subtracted image GPU", result_host);
  cv::waitKey(0);
#endif
  
  return imgOut;
}

/* ==========================================================================
*        FUNCTION NAME: addiction
* FUNCTION DESCRIPTION: Addiction of images, matrix-matrix addiction.
*        CREATION DATE: 20160727
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
cv::gpu::GpuMat addiction(const cv::gpu::GpuMat& imgA, const cv::gpu::GpuMat& imgB)
{
  cv::gpu::GpuMat imgOut = 
    cv::gpu::createContinuous(imgA.rows, imgA.cols, imgA.type());
      
  cv::gpu::add(imgA, imgB, imgOut);
  
#if SPD_FIGURE_1
  cv::Mat result_host;
  imgOut.download(result_host);
  // Create a window for display.
  namedWindow("Sum image GPU", cv::WINDOW_NORMAL);
  imshow("Sum image GPU", result_host);
  cv::waitKey(0);
#endif
  
  return imgOut;
}

/* ==========================================================================
*        FUNCTION NAME: morphologyOpen
* FUNCTION DESCRIPTION: Morphology opening
*        CREATION DATE: 20160727
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
cv::gpu::GpuMat morphologyOpen(const cv::gpu::GpuMat& imgIn, int dimLine, double teta_streak)
{
  //cv::gpu::GpuMat imgOut;
  cv::gpu::GpuMat imgOut = 
    cv::gpu::createContinuous(imgIn.rows, imgIn.cols, imgIn.type());

  int iter = 1;
  cv::Point anchor = cv::Point(-1, -1);

  //InputArray kernel;
  cv::Mat horizontalStructure = getStructuringElement(cv::MORPH_RECT, cv::Size(dimLine, 1), anchor);

  cv::gpu::morphologyEx(imgIn, imgOut, cv::MORPH_OPEN, horizontalStructure, anchor, iter);
    
#if SPD_FIGURE_1
  cv::Mat result_host;
  imgOut.download(result_host);
  // Create a window for display.
  namedWindow("Morphology opening with rectangular kernel GPU", cv::WINDOW_NORMAL);
  imshow("Morphology opening with rectangular kernel GPU", result_host);
  cv::waitKey(0);
#endif

  return imgOut;
}

/* ==========================================================================
*        FUNCTION NAME: morphologyOpen
* FUNCTION DESCRIPTION: Morphology opening with circular structuring element
*        CREATION DATE: 20160727
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
cv::gpu::GpuMat morphologyOpen(const cv::gpu::GpuMat& imgIn, int rad)
{
  cv::gpu::GpuMat imgOut =
    cv::gpu::createContinuous(imgIn.rows, imgIn.cols, imgIn.type());

  int iter = 1;
  cv::Point anchor = cv::Point(-1, -1);
  cv::Size size = cv::Size(rad, rad);

  //InputArray kernel;
  cv::Mat structElement = cv::getStructuringElement(cv::MORPH_ELLIPSE, size, anchor);

  cv::gpu::morphologyEx(imgIn, imgOut, cv::MORPH_OPEN, structElement, anchor, iter);

#if SPD_FIGURE_1
  cv::Mat result_host;
  imgOut.download(result_host);
  // Create a window for display.
  namedWindow("Morphology opening with circular kernel", cv::WINDOW_NORMAL);
  imshow("Morphology opening with circular kernel", result_host);
  cv::waitKey(0);
#endif

  return imgOut;
}

/* ==========================================================================
*        FUNCTION NAME: binarization
* FUNCTION DESCRIPTION: Image binarization
*        CREATION DATE: 20160727
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
cv::gpu::GpuMat binarization(const cv::gpu::GpuMat& imgIn)
{
  //cv::gpu::GpuMat imgOut, binImg;
  
  cv::gpu::GpuMat imgOut = 
    cv::gpu::createContinuous(imgIn.rows, imgIn.cols, imgIn.type());
    
  cv::gpu::GpuMat binImg = 
    cv::gpu::createContinuous(imgIn.rows, imgIn.cols, imgIn.type());      
    
  double maxval = 255.0;
  double level = 0.0;
  
  level = cv::gpu::threshold(imgIn, binImg, cv::THRESH_OTSU, maxval, cv::THRESH_BINARY);
  
  level = level * 2.5;//1.5
  
  cv::gpu::threshold(imgIn, imgOut, level, maxval, cv::THRESH_BINARY);
  
#if SPD_FIGURE_1
  /* Create a window for display.
  namedWindow("Binary image", WINDOW_NORMAL);
  imshow("Binary image", binImg);*/

  cv::Mat result_host;
  imgOut.download(result_host);
  // Create a window for display.
  namedWindow("Binary image Otsu threshold GPU", cv::WINDOW_NORMAL);
  imshow("Binary image Otsu threshold GPU", result_host);
  cv::waitKey(0);
#endif

  return imgOut;
}

/* ==========================================================================
*        FUNCTION NAME: binarizationZone
* FUNCTION DESCRIPTION: Image binarization using user threshold
*        CREATION DATE: 20160727
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
cv::gpu::GpuMat binarizationZone(const cv::gpu::GpuMat& imgIn, const cv::Point zoneCnt, const cv::Mat& level)
{
  size_t zoneSzR = static_cast<size_t>(::round(static_cast<float>(imgIn.rows / zoneCnt.y)));
  size_t zoneSzC = static_cast<size_t>(::round(static_cast<float>(imgIn.cols / zoneCnt.x)));
  
  std::vector<int> vBackSrow;
  std::vector<int> vBackScol;

  for (size_t o = 0; o < zoneCnt.y; ++o)
  {
    vBackSrow.push_back(zoneSzR*o);
  }
  vBackSrow.push_back(imgIn.rows);

  for (size_t o = 0; o < zoneCnt.x; ++o)
  {
    vBackScol.push_back(zoneSzC*o);
  }
  vBackScol.push_back(imgIn.cols);
    
  cv::gpu::GpuMat outImg = cv::gpu::createContinuous(imgIn.rows, imgIn.cols, imgIn.type());
  
  for (size_t i = 0; i < zoneCnt.y; ++i)
  {
    for (size_t j = 0; j < zoneCnt.x; ++j)
    {
      const cv::Point ptTL ( vBackScol.at(j), vBackSrow.at(i) );
      const cv::Point ptBR ( vBackScol.at(j + 1), vBackSrow.at(i + 1) );

      cv::Rect region_of_interest = cv::Rect(ptTL, ptBR);
      cv::gpu::GpuMat imgPart = imgIn(region_of_interest);
      cv::gpu::GpuMat imgPartTh = cv::gpu::createContinuous(imgPart.rows, imgPart.cols, imgPart.type());

      double maxval = 255.0;
      double asdf = cv::gpu::threshold(imgPart, imgPartTh, level.at<double>(i,j), maxval, cv::THRESH_BINARY);
      
      cv::gpu::GpuMat img_RoI = outImg(region_of_interest);
      imgPart.copyTo(img_RoI);
    }
  }
  
#if SPD_FIGURE_1
  cv::Mat result_host;
  outImg.download(result_host);
  namedWindow("Binary image user thresholdZones", cv::WINDOW_NORMAL);
  imshow("Binary image user thresholdZones", result_host);
  cv::waitKey(0);
#endif

  return outImg;
}

#if 0
/* ==========================================================================
*        FUNCTION NAME: binarizationDiffTh
* FUNCTION DESCRIPTION: Image binarization
*        CREATION DATE: 20160727
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
cv::gpu::GpuMat binarizationDiffTh(cv::gpu::GpuMat& imgIn, int flag)
{
  cv::gpu::GpuMat imgOut, binImg;
  cv::gpu::GpuMat subBImgTL, subBImgTR, subBImgBL, subBImgBR;

  cv::Point imgSz ( imgIn.rows, imgIn.cols );

  /*int dims[] = { 5, 1 };
  cv::Mat level(2, dims, CV_64F);*/

  cv::gpu::GpuMat subImageTL(imgIn, cv::Rect(0, 0, imgIn.cols/2, imgIn.rows/2));
  cv::gpu::GpuMat subImageTR(imgIn, cv::Rect(0, imgIn.rows/2, imgIn.cols/2, imgIn.rows/2));
  cv::gpu::GpuMat subImageBL(imgIn, cv::Rect(0, imgIn.rows/2, imgIn.cols/2, imgIn.rows/2));
  cv::gpu::GpuMat subImageBR(imgIn, cv::Rect(imgIn.cols/2, imgIn.rows/2, imgIn.cols/2, imgIn.rows/2));

  
  double maxval = 1.0;
  double level1 = 0.0;
  double level2 = 0.0;
  double level3 = 0.0;
  double level4 = 0.0;
  double level5 = 0.0;

  level1 = cv::gpu::threshold(subImageTL, subBImgTL, cv::THRESH_OTSU, maxval, cv::THRESH_BINARY);
  level2 = cv::gpu::threshold(subImageTR, subBImgTR, cv::THRESH_OTSU, maxval, cv::THRESH_BINARY);
  level3 = cv::gpu::threshold(subImageBL, subBImgBL, cv::THRESH_OTSU, maxval, cv::THRESH_BINARY);
  level4 = cv::gpu::threshold(subImageBR, subBImgBR, cv::THRESH_OTSU, maxval, cv::THRESH_BINARY);
  level5 = cv::gpu::threshold(binImg    ,    imgOut, cv::THRESH_OTSU, maxval, cv::THRESH_BINARY);

  level1 = level1 *1.5;
  level2 = level2 *1.5;
  level3 = level3 *1.5;
  level4 = level4 *1.5;
  level5 = level5 *1.5;

  /*media mediana ordinamento */

  /*da completare*/

  if (SPD_FIGURE_1)
  {
    // Create a window for display.
    namedWindow("Binary image", cv::WINDOW_NORMAL);
    imshow("Binary image", imgOut);
  }

  return imgOut;
}
#endif

/* ==========================================================================
*        FUNCTION NAME: hough
* FUNCTION DESCRIPTION: Hough transform
*        CREATION DATE: 20160911
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
std::vector<std::pair<float, int>> hough(const cv::gpu::GpuMat& imgIn)
{
  double rho = 1; //0.5;
  double theta = 2*CV_PI / 180; //CV_PI / (2*180); r:pi=g:180
  int threshold = 60;
  
  //std::vector<cv::Vec2f> houghVal;
  cv::gpu::GpuMat houghVal;
  bool doSort = true; 
  int maxLines = 5;

  cv::gpu::HoughLines(imgIn, houghVal, rho, theta, threshold, doSort, maxLines);
  
  cv::Mat houghValCPU;
  cv::gpu::HoughLinesDownload(houghVal, houghValCPU);

  std::cout << "hough val " << houghValCPU << std::endl;

#if 0
  // Select the inclination angles
  std::vector<float> angle;
  for (size_t i = 0; i < houghVal.size(); ++i)
  {
    angle.push_back(houghVal.at(i)[1]);
  }
  angle.push_back(CV_PI / 2); //Force research at 0\B0
  
  int count = 0;
  std::vector<std::pair<float, int>> countAngle;
  for (size_t i = 0; i < houghVal.size(); ++i)
  {
    int a = std::count(angle.begin(), angle.end(), angle.at(i));
    countAngle.push_back(std::make_pair(angle.at(i), a));
    count = count + a;
    if (houghVal.size() == count) break;
  }
#else
  std::vector<std::pair<float, int>> countAngle;
#endif

#if SPD_FIGURE_1
    cv::Mat color_dst;
    cvtColor( imgIn, color_dst, CV_GRAY2BGR );
    double minLineLength = 20;
    double maxLineGap = 1;
    std::vector<cv::Vec4i> lines;
    HoughLinesP(imgIn, lines, rho, theta, threshold, minLineLength, maxLineGap);

    for (size_t i = 0; i < lines.size(); i++) {
      line(color_dst, cv::Point(lines[i][0], lines[i][1]),
        cv::Point(lines[i][2], lines[i][3]), cv::Scalar(0, 0, 255), 3, 8);
    }

    // Create a window for display.
    namedWindow("Hough transform", cv::WINDOW_NORMAL);
    imshow("Hough transform", color_dst);
    cv::waitKey(0);
#endif
  
  return countAngle;
}

/* ==========================================================================
*        FUNCTION NAME: convolution
* FUNCTION DESCRIPTION: Image convolution
*        CREATION DATE: 20160727
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
cv::gpu::GpuMat convolution(const cv::gpu::GpuMat& imgIn, int szK, int thresh, int maxval)
{
  cv::gpu::GpuMat imgOut = 
    cv::gpu::createContinuous(imgIn.rows, imgIn.cols, imgIn.type());
  
  externalClass kernelCUDA;
  kernelCUDA.convolutionThreshCUDAKernel(imgIn, imgOut, szK, thresh, maxval);  
  
#if SPD_FIGURE_1
    cv::Mat result_host;
    imgOut.download(result_host);
    // Create a window for display.
    namedWindow("Convolution image GPU", cv::WINDOW_NORMAL);
    imshow("Convolution image GPU", result_host);
#endif

  return imgOut;
}

/* ==========================================================================
*        FUNCTION NAME: iDivUp
* FUNCTION DESCRIPTION: Rounded division 
*        CREATION DATE: 20160727
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
int iDivUp(int a, int b)
{ 
  return ((a % b) != 0) ? (a / b + 1) : (a / b); 
}

/* ==========================================================================
*        FUNCTION NAME: gpuErrchk
* FUNCTION DESCRIPTION: CUDA error check
*        CREATION DATE: 20160727
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
//#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
