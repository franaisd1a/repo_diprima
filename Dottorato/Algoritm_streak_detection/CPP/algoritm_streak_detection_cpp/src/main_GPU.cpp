/* ==========================================================================
* ALGO STREAK DETECTION
* ========================================================================== */

/* ==========================================================================
* MODULE FILE NAME: main_GPU.cpp
*      MODULE TYPE:
*
*         FUNCTION: Detect streaks and points on GPU.
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
#include "function.h"
#include "macros.h"

#include <stdio.h>
#include <iostream>
#include <fstream>

/* ==========================================================================
* MODULE PRIVATE MACROS
* ========================================================================== */
#define MEDIAN_FILTER_CPU 0U
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
*        FUNCTION NAME: main_GPU
* FUNCTION DESCRIPTION:
*        CREATION DATE: 20160727
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
int main_GPU(char* name_file)
{
  std::ofstream infoFile(stdout);

  cout << "GPU algorithms ." << std::endl;
  /* Open file */
  FILE * pFile;
  pFile = fopen ("consoleGPU.txt","w");
   
  // Read file
  Mat Img_input = imread(name_file, CV_LOAD_IMAGE_GRAYSCALE );
    
  // Check for invalid file
  if (!Img_input.data)  {
    cout << "Error: could not open or find the image." << std::endl;
    return -1;
  }
  
  int channels = Img_input.channels();
  int depth = Img_input.depth();
  //cv::Point_<int> I_input_size = { Img_input.cols, Img_input.rows  };
  cv::Point_<int> I_input_size(Img_input.cols, Img_input.rows);
  
  cv::Point_<double> borders = { 0.015, 0.985 };
  Vec<int, 4> imgBorders = {static_cast<int>(ceil( borders.x * I_input_size.x))
                          , static_cast<int>(ceil( borders.x * I_input_size.y))
                          , static_cast<int>(floor(borders.y * I_input_size.x))
                          , static_cast<int>(floor(borders.y * I_input_size.y))};

  fprintf(pFile, "Image channels: %d\n", channels);
  fprintf(pFile, "Image depth bit: %d\n", depth);

/* ======================================================================= *
 * GPU initializations and informations                                    *
 * ======================================================================= */

  int deviceCount = gpu::getCudaEnabledDeviceCount();

  fprintf(pFile, "Device number %d\n", deviceCount);

  //Move data on GPU 
  gpu::GpuMat srcImgGPU = 
    gpu::createContinuous(Img_input.rows, Img_input.cols, Img_input.type());
    
  srcImgGPU.upload(Img_input);
    
  gpu::GpuMat dstImgGPU = 
    gpu::createContinuous(Img_input.rows, Img_input.cols, Img_input.type());
  
    
/* ======================================================================= *
 * Big Points detection                                                    *
 * ======================================================================= */

/* ----------------------------------------------------------------------- *
 * Gaussian filter                                                         *
 * ----------------------------------------------------------------------- */

  int hsize[2] = {31, 31};//{101, 101};
  double sigma = 30;
  clock_t start = clock();

  gpu::GpuMat gaussImg = gaussianFilter(srcImgGPU, hsize, sigma);

  timeElapsed(infoFile, start, "Gaussian filter");

  fprintf(pFile, "End Gaussian filter\n");

/* ----------------------------------------------------------------------- *
 * Background subtraction                                                  *
 * ----------------------------------------------------------------------- */

  start = clock();

  gpu::GpuMat backgroundSub = subtractImage(srcImgGPU, gaussImg);
  
  timeElapsed(infoFile, start, "Background subtraction");

  gaussImg.release();

  fprintf(pFile, "End Background subtraction\n");

/* ----------------------------------------------------------------------- *
 * Median filter                                                           *
 * ----------------------------------------------------------------------- */

  int kerlen = 11;

  /* CPU version of median filter */
  //Download data from GPU 
  cv::Mat bgSub;
  backgroundSub.download(bgSub);
  backgroundSub.release();
  
  Mat medianImg = medianFilter(bgSub, kerlen);

  fprintf(pFile, "End Median filter\n");

/* ----------------------------------------------------------------------- *
 * Binarization                                                            *
 * ----------------------------------------------------------------------- */

  //Move data on GPU 
  gpu::GpuMat medianImgGPU;
  medianImgGPU.upload(medianImg);
  

  start = clock();
  gpu::GpuMat binaryImgGPU = binarization(medianImgGPU);

  timeElapsed(infoFile, start, "Binarization");

  medianImgGPU.release();

  fprintf(pFile, "End Binarization\n");


/* ----------------------------------------------------------------------- *
 * Convolution kernel                                                      *
 * ----------------------------------------------------------------------- */

  int szKernel = 3;
  double threshConv = szKernel*szKernel;
  Mat kernel = Mat::ones(szKernel, szKernel, CV_8U);
  
  start = clock();
  gpu::GpuMat convImgGPU = convolution(binaryImgGPU, kernel, threshConv);
  
  timeElapsed(infoFile, start, "Convolution");

  fprintf(pFile, "End Convolution kernel\n");

  //Download data from GPU 
  cv::Mat convImg;
  convImgGPU.download(convImg);
  convImgGPU.release();

/* ----------------------------------------------------------------------- *
 * Hough transform                                                         *
 * ----------------------------------------------------------------------- */
#if 0
  Mat houghImg = hough(medianImg);
#endif

/* ----------------------------------------------------------------------- *
 * Connected components                                                    *
 * ----------------------------------------------------------------------- */
      
  std::vector< cv::Vec<float, 3> > POINTS;
  std::vector< cv::Vec<float, 3> > STREAKS;
  
  connectedComponents(convImg, convImg, Img_input, imgBorders, POINTS, STREAKS);

  Mat color_Img_input;
  cvtColor( Img_input, color_Img_input, CV_GRAY2BGR );

  int radius = 5;
  Scalar colorP = {0,255,0};
  Scalar colorS = {0,0,255};
  int thickness = -1;
  int lineType = 8;
  int shift = 0;

cout << "points sz " << POINTS.size() << std::endl;
  for (size_t i = 0; i < POINTS.size(); ++i)
  {
    cout << "points " << i << std::endl;
    Point center = { static_cast<int>(POINTS.at(i)[0]), static_cast<int>(POINTS.at(i)[1]) };
    circle(color_Img_input, center, radius, colorP, thickness, lineType, shift);

    /*center = { STREAKS.at(i)[0], STREAKS.at(i)[1] };
    circle(color_Img_input, center, radius, color, thickness, lineType, shift);*/
  }
cout << "STREAKS sz " << STREAKS.size() << std::endl;
  for (size_t i = 0; i < STREAKS.size(); ++i)
  {
    cout << "STREAKS " << i << std::endl;
    Point center = { static_cast<int>(STREAKS.at(i)[0]), static_cast<int>(STREAKS.at(i)[1]) };
    circle(color_Img_input, center, radius, colorS, thickness, lineType, shift);
  }


/* ----------------------------------------------------------------------- *
 * Morphology opening                                                      *
 * ----------------------------------------------------------------------- */

  if (SPD_FIGURE)
  {
    // Create a window for display.
    namedWindow("Algo gpu", WINDOW_NORMAL);
    imshow("Algo gpu", color_Img_input);
  }


#if 0
  Mat result_host;
  srcImgGPU.download(result_host);

  namedWindow("Display window2", WINDOW_NORMAL);
  imshow("Display window2", result_host);
#endif

  cv::waitKey(0);
  
  std::fclose(pFile);
      
  return 1;
}
