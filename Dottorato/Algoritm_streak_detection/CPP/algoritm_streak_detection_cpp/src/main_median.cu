#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv/highgui.h>
#include "opencv2/gpu/gpu.hpp"
#include "externalClass.cu" // important to include .cu file, not header file

#include "function_GPU.h"
#include "function.h"
#include "macros.h"

using namespace cv;
using namespace std;

int main_median(char* name_file)
{
  cout << "GPU algorithms median." << std::endl;
  
/* ======================================================================= *
 * GPU initializations and informations                                    *
 * ======================================================================= */

  int deviceCount = gpu::getCudaEnabledDeviceCount();

  gpu::setDevice(deviceCount-1);
    
  // --- CUDA warm up
  /*unsigned short *forFirstCudaMalloc; gpuErrchk(cudaMalloc((void**)&forFirstCudaMalloc, dataLength * sizeof(unsigned short)));
  gpuErrchk(cudaFree(forFirstCudaMalloc));*/
  
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

  timeElapsed(start, "Gaussian filter");

  fprintf(pFile, "End Gaussian filter\n");

/* ----------------------------------------------------------------------- *
 * Background subtraction                                                  *
 * ----------------------------------------------------------------------- */

  start = clock();

  gpu::GpuMat backgroundSub = subtractImage(srcImgGPU, gaussImg);
  
  timeElapsed(start, "Background subtraction");

  gaussImg.release();

  fprintf(pFile, "End Background subtraction\n");

/* ----------------------------------------------------------------------- *
 * Median filter                                                           *
 * ----------------------------------------------------------------------- */
	
  start = clock();
  
  gpu::GpuMat medianImgGPU = 
    gpu::createContinuous(Img_input.rows, Img_input.cols, Img_input.type());
  
  externalClass medianFilter;
  medianFilter.cudaKernel(srcImgGPU, medianImgGPU);
  
  timeElapsed(start, "Median filter ");
	
/* ----------------------------------------------------------------------- *
 * Binarization                                                            *
 * ----------------------------------------------------------------------- */

  start = clock();
  gpu::GpuMat binaryImgGPU = binarization(medianImgGPU);

  timeElapsed(start, "Binarization");

  medianImgGPU.release();

  fprintf(pFile, "End Binarization\n");

/* ----------------------------------------------------------------------- *
 * Convolution kernel                                                      *
 * ----------------------------------------------------------------------- */

  int szKernel = 3;
  double threshConv = szKernel*szKernel;
  const Mat kernel = Mat::ones(szKernel, szKernel, CV_8U);
  
  start = clock();
  gpu::GpuMat convImgGPU = convolution(binaryImgGPU, kernel, threshConv);
  
  timeElapsed(start, "Convolution");

  fprintf(pFile, "End Convolution kernel\n");

/* ----------------------------------------------------------------------- *
 * Connected components                                                    *
 * ----------------------------------------------------------------------- */

  //Download data from GPU 
  cv::Mat convImg;
  convImgGPU.download(convImg);
  convImgGPU.release();
  
  std::vector< cv::Vec<int, 3> > POINTS;

  std::vector< cv::Vec<int, 3> > STREAKS;
  
  connectedComponents(convImg, imgBorders, POINTS, STREAKS);

  Mat color_Img_input;
  cvtColor( Img_input, color_Img_input, CV_GRAY2BGR );

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


/* ----------------------------------------------------------------------- *
 * End                                                                     *
 * ----------------------------------------------------------------------- */

  if (FIGURE)
  {
    // Create a window for display.
    namedWindow("Display window", WINDOW_NORMAL);
    imshow("Display window", color_Img_input);
  }

  fclose(pFile);
        
  return 1;
}