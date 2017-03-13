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

int main_GPU_cuda(char* name_file)
{
  //cout << "GPU algorithms with CUDA kernel." << std::endl;
  std::ofstream infoFile(stdout);

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
  
  // Read file
  Mat Img_input = imread(name_file, CV_LOAD_IMAGE_GRAYSCALE );
    
  // Check for invalid file
  if (!Img_input.data)  {
    cout << "Error: could not open or find the image." << std::endl;
    return -1;
  }
  
  int channels = Img_input.channels();
  int depth = Img_input.depth();
  cv::Point_<int> I_input_size ( Img_input.cols, Img_input.rows  );
  cv::Point_<double> borders ( 0.015, 0.985 );
  Vec<int, 4> imgBorders (static_cast<int>(ceil( borders.x * I_input_size.x))
                          , static_cast<int>(ceil( borders.x * I_input_size.y))
                          , static_cast<int>(floor(borders.y * I_input_size.x))
                          , static_cast<int>(floor(borders.y * I_input_size.y)));

  fprintf(pFile, "Image channels: %d\n", channels);
  fprintf(pFile, "Image depth bit: %d\n", depth);

  //Move data on GPU
  gpu::GpuMat srcImgGPU = 
    gpu::createContinuous(Img_input.rows, Img_input.cols, Img_input.type());
  
  srcImgGPU.upload(Img_input);
  
/* ----------------------------------------------------------------------- *
 * Gaussian filter                                                         *
 * ----------------------------------------------------------------------- */

  int hsize[2] = {31, 31};//{101, 101};
  double sigma = 30;
  clock_t start = clock();

  gpu::GpuMat gaussImg = gaussianFilter(srcImgGPU, hsize, sigma);

  if (SPD_STAMP) {
    timeElapsed(infoFile, start, "Gaussian filter");}

  fprintf(pFile, "End Gaussian filter\n");
//cv::waitKey(0);
/* ----------------------------------------------------------------------- *
 * Background subtraction                                                  *
 * ----------------------------------------------------------------------- */

  start = clock();

  gpu::GpuMat backgroundSub = subtractImage(srcImgGPU, gaussImg);
  
  srcImgGPU.release();
  gaussImg.release();

  if (SPD_STAMP) {
    timeElapsed(infoFile, start, "Background subtraction");}

  gaussImg.release();

  fprintf(pFile, "End Background subtraction\n");
//cv::waitKey(0);
/* ----------------------------------------------------------------------- *
 * Median filter                                                           *
 * ----------------------------------------------------------------------- */
	
  start = clock();
  
  gpu::GpuMat medianImgGPU = 
    gpu::createContinuous(Img_input.rows, Img_input.cols, Img_input.type());
  
  int kerlen = 9;

  externalClass kernelCUDA;
  kernelCUDA.medianCUDAKernel(backgroundSub, medianImgGPU, kerlen);
  
  backgroundSub.release();

  if (SPD_STAMP) {
    timeElapsed(infoFile, start, "Median filter ");}

  if (SPD_FIGURE_1)
  {
    cv::Mat result_hostMedian;
    medianImgGPU.download(result_hostMedian);
    // Create a window for display.
    namedWindow("Median filter GPU", cv::WINDOW_NORMAL);
    imshow("Median filter GPU", result_hostMedian);
  }
	//cv::waitKey(0);

/* ----------------------------------------------------------------------- *
 * Binarization                                                            *
 * ----------------------------------------------------------------------- */

  start = clock();
  gpu::GpuMat binaryImgGPU = binarization(medianImgGPU);

  if (SPD_STAMP) {
    timeElapsed(infoFile, start, "Binarization");}

  medianImgGPU.release();

  fprintf(pFile, "End Binarization\n");
//cv::waitKey(0);
/* ----------------------------------------------------------------------- *
 * Convolution kernel                                                      *
 * ----------------------------------------------------------------------- */

  int szKernel = 3;
  double threshConv = szKernel*szKernel;
  const Mat kernel = Mat::ones(szKernel, szKernel, CV_8U);
  
  gpu::GpuMat convImgGPU = 
    gpu::createContinuous(Img_input.rows, Img_input.cols, Img_input.type());
	
  gpu::GpuMat convBinGPU = 
    gpu::createContinuous(Img_input.rows, Img_input.cols, Img_input.type());
	
  start = clock();
  //gpu::GpuMat convImgGPU = convolution(binaryImgGPU, kernel, threshConv);
    
  kernelCUDA.convolutionCUDAKernel(binaryImgGPU, convImgGPU, szKernel);
  
  binaryImgGPU.release();

  if (SPD_FIGURE_1)
  {
    cv::Mat result_hostcon;
    convImgGPU.download(result_hostcon);
    // Create a window for display.
    namedWindow("Convolution filter GPU", cv::WINDOW_NORMAL);
    imshow("Convolution filter GPU", result_hostcon);
  }
  
  double maxval = 255.0;
  cv::gpu::threshold(convImgGPU, convBinGPU, threshConv, maxval, cv::THRESH_BINARY);
    
  convImgGPU.release();

  if (SPD_STAMP) {
    timeElapsed(infoFile, start, "Convolution");}

  if (SPD_FIGURE_1)
  {
    cv::Mat result_hostthcon;
    convBinGPU.download(result_hostthcon);
    // Create a window for display.
    namedWindow("Convolution threshold filter GPU", cv::WINDOW_NORMAL);
    imshow("Convolution threshold filter GPU", result_hostthcon);
  }
  
  fprintf(pFile, "End Convolution kernel\n");
//cv::waitKey(0);
/* ----------------------------------------------------------------------- *
 * Connected components                                                    *
 * ----------------------------------------------------------------------- */

  //Download data from GPU 
  cv::Mat convImg;
  convBinGPU.download(convImg);
  convBinGPU.release();
  
  std::vector< cv::Vec<float, 3> > POINTS;
  std::vector< cv::Vec<float, 3> > STREAKS;
  
  connectedComponents(convImg, convImg, Img_input, imgBorders, POINTS, STREAKS);

  
/* ----------------------------------------------------------------------- *
 * Plot Result                                                             *
 * ----------------------------------------------------------------------- */

  if (SPD_FIGURE)
  {
    Mat color_Img_input;
    cvtColor( Img_input, color_Img_input, CV_GRAY2BGR );
    
    Img_input.release();

    int radius = 5;
    Scalar colorP (0,255,0);
    Scalar colorS (0,0,255);
    int thickness = -1;
    int lineType = 8;
    int shift = 0;

    for (size_t i = 0; i < POINTS.size(); ++i)
    {
      Point center ( POINTS.at(i)[0], POINTS.at(i)[1] );
	
      circle(color_Img_input, center, radius, colorP, thickness, lineType, shift);

      /*center = { STREAKS.at(i)[0], STREAKS.at(i)[1] };
      circle(color_Img_input, center, radius, color, thickness, lineType, shift);*/
    }

    for (size_t i = 0; i < STREAKS.size(); ++i)
    {
      Point center ( STREAKS.at(i)[0], STREAKS.at(i)[1] );
      circle(color_Img_input, center, radius, colorS, thickness, lineType, shift);
    }

    // Create a window for display.
    namedWindow("CUDA kernel", WINDOW_NORMAL);
    imshow("CUDA kernel", color_Img_input);
  }

  fclose(pFile);
        
  //cv::waitKey(0);
  
  return 1;
}
