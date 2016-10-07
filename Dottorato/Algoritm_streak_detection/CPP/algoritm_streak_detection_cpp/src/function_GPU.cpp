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
*        FUNCTION NAME: gaussianFilter
* FUNCTION DESCRIPTION: Gaussian lowpass filter
*        CREATION DATE: 20160727
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
cv::gpu::GpuMat gaussianFilter(cv::gpu::GpuMat& imgIn, int hsize[2], double sigma)
{
  //cv::gpu::GpuMat imgOut;
  cv::gpu::GpuMat imgOut = 
    cv::gpu::createContinuous(imgIn.rows, imgIn.cols, imgIn.type());  

  cv::Size h = { hsize[0], hsize[1] };

  int columnBorderType=-1;
  cv::gpu::GaussianBlur(imgIn, imgOut, h, sigma, sigma, cv::BORDER_DEFAULT, columnBorderType);
  
  if (FIGURE_1)
  {
    cv::Mat result_host;
    imgOut.download(result_host);
    // Create a window for display.
    namedWindow("Gaussain filter GPU", cv::WINDOW_NORMAL);
    imshow("Gaussain filter GPU", result_host);
  }

  return imgOut;
}

/* ==========================================================================
*        FUNCTION NAME: subtractImage
* FUNCTION DESCRIPTION: Subtraction of image, matrix-matrix difference.
*        CREATION DATE: 20160727
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
cv::gpu::GpuMat subtractImage(cv::gpu::GpuMat& imgA, cv::gpu::GpuMat& imgB)
{
  //cv::gpu::GpuMat imgOut;
  cv::gpu::GpuMat imgOut = 
    cv::gpu::createContinuous(imgA.rows, imgA.cols, imgA.type());
      
  cv::gpu::subtract(imgA, imgB, imgOut);
  
  if (FIGURE_1)
  {
    cv::Mat result_host;
    imgOut.download(result_host);
    // Create a window for display.
    namedWindow("Subtracted image GPU", cv::WINDOW_NORMAL);
    imshow("Subtracted image GPU", result_host);
  }
  
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
cv::gpu::GpuMat morphologyOpen(cv::gpu::GpuMat& imgIn, int dimLine, double teta_streak)
{
  //cv::gpu::GpuMat imgOut;
  cv::gpu::GpuMat imgOut = 
    cv::gpu::createContinuous(imgIn.rows, imgIn.cols, imgIn.type());  

  int iter = 1;
  cv::Point anchor = cv::Point(-1, -1);

  //InputArray kernel;
  cv::Mat horizontalStructure = getStructuringElement(cv::MORPH_RECT, cv::Size(dimLine, 1));

  cv::gpu::morphologyEx(imgIn, imgOut, cv::MORPH_OPEN, horizontalStructure, anchor, iter);
    
  if (FIGURE_1)
  {
    cv::Mat result_host;
    imgOut.download(result_host);
    // Create a window for display.
    namedWindow("Morphology opening with rectangular kernel GPU", cv::WINDOW_NORMAL);
    imshow("Morphology opening with rectangular kernel GPU", result_host);
  }

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
cv::gpu::GpuMat binarization(cv::gpu::GpuMat& imgIn)
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
  
  if (FIGURE_1)
  {
    /* Create a window for display.
    namedWindow("Binary image", WINDOW_NORMAL);
    imshow("Binary image", binImg);*/

    cv::Mat result_host;
    imgOut.download(result_host);
    // Create a window for display.
    namedWindow("Binary image Otsu threshold GPU", cv::WINDOW_NORMAL);
    imshow("Binary image Otsu threshold GPU", result_host);
  }

  return imgOut;
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

  if (FIGURE_1)
  {
    // Create a window for display.
    namedWindow("Binary image", cv::WINDOW_NORMAL);
    imshow("Binary image", imgOut);
  }

  return imgOut;
}
#endif

/* ==========================================================================
*        FUNCTION NAME: convolution
* FUNCTION DESCRIPTION: Image convolution
*        CREATION DATE: 20160727
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
cv::gpu::GpuMat convolution(cv::gpu::GpuMat& imgIn, const cv::Mat& kernel, double thresh)
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
  
  if (FIGURE_1)
  {
    cv::Mat result_host;
    imgOut.download(result_host);
    // Create a window for display.
    namedWindow("Convolution image GPU", cv::WINDOW_NORMAL);
    imshow("Convolution image GPU", result_host);

  }
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


