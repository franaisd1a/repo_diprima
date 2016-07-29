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
#include "function.h"

/* ==========================================================================
* MODULE PRIVATE MACROS
* ========================================================================== */
#define FIGURE 1U
#define FIGURE_1 0U

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
using namespace cv;
using namespace std;

/* ==========================================================================
*        FUNCTION NAME: gaussianFilter
* FUNCTION DESCRIPTION: Gaussian lowpass filter
*        CREATION DATE: 20160727
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
Mat gaussianFilter(Mat& imgIn, int hsize[2], double sigma)
{
  Mat imgOut;

  Size h = { hsize[0], hsize[1] };

  GaussianBlur(imgIn, imgOut, h, sigma, sigma, BORDER_DEFAULT);
  
  return imgOut;
}

/* ==========================================================================
*        FUNCTION NAME: medianFilter
* FUNCTION DESCRIPTION: Median filter
*        CREATION DATE: 20160727
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
Mat medianFilter(cv::Mat& imgIn, int kerlen)
{
  Mat imgOut;

  medianBlur(imgIn, imgOut, kerlen);

  return imgOut;
}

/* ==========================================================================
*        FUNCTION NAME: medianFilter
* FUNCTION DESCRIPTION: Median filter
*        CREATION DATE: 20160727
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
Mat medianFilter(cv::Mat& imgIn, int littleKerlen, int bigKerlen)
{
  Mat imgOut, imgBigKer;

  medianBlur(imgIn, imgOut, littleKerlen);
  medianBlur(imgIn, imgBigKer, bigKerlen);

  imgOut = imgOut - imgBigKer;

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
Mat morphologyOpen(cv::Mat& imgIn, int dimLine, double teta_streak)
{
  Mat imgOut;

  int iter = 1;
  Point anchor = Point(-1, -1);

  //InputArray kernel;
  Mat horizontalStructure = getStructuringElement(MORPH_RECT, Size(dimLine, 1));

  morphologyEx(imgIn, imgOut, MORPH_OPEN, horizontalStructure, anchor, iter
    , BORDER_CONSTANT, morphologyDefaultBorderValue());

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
Mat binarization(cv::Mat& imgIn, int flag)
{
  Mat imgOut, binImg, subBImgTL, subBImgTR, subBImgBL, subBImgBR;

  Point imgSz = { imgIn.rows, imgIn.cols };

  /*int dims[] = { 5, 1 };
  cv::Mat level(2, dims, CV_64F);*/

  Mat subImageTL(imgIn, Rect(0, 0, imgIn.cols/2, imgIn.rows/2));
  Mat subImageTR(imgIn, Rect(0, imgIn.rows/2, imgIn.cols/2, imgIn.rows/2));
  Mat subImageBL(imgIn, Rect(0, imgIn.rows/2, imgIn.cols/2, imgIn.rows/2));
  Mat subImageBR(imgIn, Rect(imgIn.cols/2, imgIn.rows/2, imgIn.cols/2, imgIn.rows/2));

  
  double maxval = 0.0;
  double level1 = 0.0;
  double level2 = 0.0;
  double level3 = 0.0;
  double level4 = 0.0;
  double level5 = 0.0;

  level1 = threshold(subImageTL, subBImgTL, THRESH_OTSU, maxval, THRESH_BINARY);
  level2 = threshold(subImageTR, subBImgTR, THRESH_OTSU, maxval, THRESH_BINARY);
  level3 = threshold(subImageBL, subBImgBL, THRESH_OTSU, maxval, THRESH_BINARY);
  level4 = threshold(subImageBR, subBImgBR, THRESH_OTSU, maxval, THRESH_BINARY);
  level5 = threshold(binImg    ,    imgOut, THRESH_OTSU, maxval, THRESH_BINARY);

  level1 = level1 *1.5;
  level2 = level2 *1.5;
  level3 = level3 *1.5;
  level4 = level4 *1.5;
  level5 = level5 *1.5;

  /*media mediana ordinamento */

  /*da completare*/

  return imgOut;
}

/* ==========================================================================
*        FUNCTION NAME: convolution
* FUNCTION DESCRIPTION: Image convolution
*        CREATION DATE: 20160727
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
Mat convolution(cv::Mat& imgIn, cv::Mat& kernel, int threshold)
{
  Mat imgOut;
  /*kernel_size = 3 + 2 * (ind % 5);
  kernel = Mat::ones(kernel_size, kernel_size, CV_32F) / (float)(kernel_size*kernel_size);*/

  int ddepth = -1;
  Point anchor = Point(-1, -1);
  double delta = 0;

  filter2D(imgIn, imgOut, ddepth, kernel, anchor, delta, BORDER_DEFAULT);

  return imgOut;
}

