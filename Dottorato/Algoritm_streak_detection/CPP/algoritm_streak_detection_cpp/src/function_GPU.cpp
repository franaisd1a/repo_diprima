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
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv/highgui.h>
#include "opencv2/gpu/gpu.hpp"
#include "externalClass.cu" // important to include .cu file, not header file

#include "function_GPU.h"
#include "macros.h"


/* ==========================================================================
* MODULE PRIVATE MACROS
* ========================================================================== */

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
*        FUNCTION NAME: backgroundEstimation
* FUNCTION DESCRIPTION: Background Estimation
*        CREATION DATE: 20160727
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
cv::gpu::GpuMat backgroundEstimation(const cv::gpu::GpuMat& imgInOr, const cv::Point backCnt, cv::Mat& meanBg, cv::Mat& stdBg)
{
  cv::gpu::GpuMat imgIn = imgInOr.clone();
  size_t backSzR = static_cast<size_t>(::round(imgIn.rows / backCnt.y));
  size_t backSzC = static_cast<size_t>(::round(imgIn.cols / backCnt.x));
  
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
  //in cpu braso a zero

  for (size_t i = 0; i < backCnt.y; ++i)
  {
    for (size_t j = 0; j < backCnt.x; ++j)
    {
      const cv::Point ptTL = {vBackScol.at(j), vBackSrow.at(i)};
      const cv::Point ptBR = {vBackScol.at(j+1), vBackSrow.at(i+1)};

      cv::Rect region_of_interest = cv::Rect(ptTL, ptBR);
      cv::gpu::GpuMat imgPart = imgIn(region_of_interest);
      cv::gpu::GpuMat imgPartTh = cv::gpu::createContinuous(imgPart.rows, imgPart.cols, imgPart.type());
      //in cpu braso a zero
      cv::gpu::GpuMat imgPart2 = cv::gpu::createContinuous(imgPart.rows, imgPart.cols, imgPart.type());
      
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
        double asdf = cv::gpu::threshold(imgPart, imgPartTh, threshH, maxval, cv::THRESH_BINARY_INV);

        cv::gpu::multiply(imgPart, imgPartTh, imgPart2, cv::gpu::Stream::Null());

        diffPercStd = ::abs((stdBg.at<double>(i,j)-oldStd)/stdBg.at<double>(i,j));
        oldStd=stdBg.at<double>(i,j);        
      }
      
      /*cv::gpu::GpuMat img_RoI = outImg(region_of_interest);
      imgPart2.copyTo(img_RoI);
      //imgPart2.copyTo(outImg(region_of_interest).ptr());*/

      /*externalClass kernelCUDA;
      kernelCUDA.fillImgCUDAKernel(imgPart2, outImg, ptTL.x, ptTL.y, ptBR.x, ptBR.y);*/

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

  cv::Size h = { hsize[0], hsize[1] };

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
  namedWindow("Subtracted image GPU", cv::WINDOW_NORMAL);
  imshow("Subtracted image GPU", result_host);
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
  cv::Mat horizontalStructure = getStructuringElement(cv::MORPH_RECT, cv::Size(dimLine, 1));

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
  cv::Mat horizontalStructure = getStructuringElement(cv::MORPH_ELLIPSE, size, anchor);

  cv::gpu::morphologyEx(imgIn, imgOut, cv::MORPH_OPEN, horizontalStructure, anchor, iter);

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
  size_t zoneSzR = static_cast<size_t>(::round(imgIn.rows / zoneCnt.y));
  size_t zoneSzC = static_cast<size_t>(::round(imgIn.cols / zoneCnt.x));
  
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
      const cv::Point ptTL = { vBackScol.at(j), vBackSrow.at(i) };
      const cv::Point ptBR = { vBackScol.at(j + 1), vBackSrow.at(i + 1) };

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

  cv::Point imgSz = { imgIn.rows, imgIn.cols };

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
cv::gpu::GpuMat convolution(const cv::gpu::GpuMat& imgIn, const cv::Mat& kernel, double thresh)
{
  //cv::gpu::GpuMat imgOut, convImg;
  cv::gpu::GpuMat imgOut = 
    cv::gpu::createContinuous(imgIn.rows, imgIn.cols, imgIn.type());
    
  /*cv::gpu::GpuMat convImg = 
    cv::gpu::createContinuous(imgIn.rows, imgIn.cols, imgIn.type());*/
  cv::gpu::GpuMat convImg;      
    
  /*kernel_size = 3 + 2 * (ind % 5);
  kernel = Mat::ones(kernel_size, kernel_size, CV_32F) / (float)(kernel_size*kernel_size);*/

  int ddepth = -1;
  cv::Point anchor = cv::Point(-1, -1);
  cv::gpu::Stream stream;
  stream.Null();
  
  std::cout << "prima di filter2d " << std::endl;
  cv::gpu::filter2D(imgIn, convImg, ddepth, kernel, anchor, cv::BORDER_DEFAULT);
  std::cout << "dopo di filter2d " << std::endl;
/*thresh=1;
  cv::gpu::boxFilter(const_cast<const cv::gpu::GpuMat&>(imgIn), convImg, ddepth, kernel, anchor, stream);
*/
  
/*cv::Size ksize(3, 3);
  cv::gpu::blur(const_cast<const cv::gpu::GpuMat&>(imgIn), convImg, ksize, anchor, stream);
*/  
  
  double maxval = 255.0;
    
  cv::gpu::threshold(convImg, imgOut, thresh, maxval, cv::THRESH_BINARY);
//cv::gpu::threshold(imgIn, imgOut, thresh, maxval, cv::THRESH_BINARY);
  
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
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
