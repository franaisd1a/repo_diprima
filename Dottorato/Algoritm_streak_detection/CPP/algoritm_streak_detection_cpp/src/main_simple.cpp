/* ==========================================================================
* ALGO STREAK DETECTION
* ========================================================================== */

/* ==========================================================================
* MODULE FILE NAME: main.cpp
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
#include "function.h"
#include "macros.h"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <time.h>
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
using namespace cv;
using namespace std;

/* ==========================================================================
*        FUNCTION NAME: main_simple
* FUNCTION DESCRIPTION:
*        CREATION DATE: 20160727
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
int main_simple(char* nameFile)
{
  //cout << "CPU algorithms." << std::endl;

  /* Open log file */
  FILE * pFile;
  pFile = fopen ("consoleSimple.txt","w");
  

  /* Read file extension */

  char* ext = fileExt(nameFile);
  const char* extJPG = "jpg";
  const char* extFIT = "fit";

  /* Read image */

  Mat Img_input;

  if (0==strcmp(ext, extJPG))
  {
    // Read file
    Img_input = imread(nameFile, CV_LOAD_IMAGE_GRAYSCALE);

    // Check for invalid file
    if (!Img_input.data) {
      cout << "Error: could not open or find the image." << std::endl;
      return -1;
    }
  }
  else if (0==strcmp(ext, extFIT))
  {
    readFit(nameFile, Img_input);
  }
  else
  {
    printf("Error in reading process.\n");
    fprintf(pFile, "Error in reading process.\n");
    return -1;
  }


  int channels = Img_input.channels();
  int depth = Img_input.depth();
  fprintf(pFile, "Image channels: %d\n", channels);
  fprintf(pFile, "Image depth bit: %d\n", depth);

  cv::Point_<int> I_input_size = { Img_input.cols, Img_input.rows  };
  cv::Point_<double> borders = { 0.015, 0.985 };
  Vec<int, 4> imgBorders = {static_cast<int>(ceil( borders.x * I_input_size.x))
                          , static_cast<int>(ceil( borders.x * I_input_size.y))
                          , static_cast<int>(floor(borders.y * I_input_size.x))
                          , static_cast<int>(floor(borders.y * I_input_size.y))};  
  //printf("imgBorders: %d %d %d %d", imgBorders[0], imgBorders[1], imgBorders[2], imgBorders[3]);
      
  
/* ----------------------------------------------------------------------- *
 * Histogram Stretching                                                    *
 * ----------------------------------------------------------------------- */

  Mat histStretch;
  if (0 == strcmp(ext, extJPG))
  {
    histStretch = Img_input;
  }
  else if (0 == strcmp(ext, extFIT))
  {
    histStretch = histogramStretching(Img_input);
  }

  int depth2 = histStretch.depth();
  int type = histStretch.type();

/* ----------------------------------------------------------------------- *
 * Set image borders with zero                                             *
 * ----------------------------------------------------------------------- */
  
  cv::Rect border(cv::Point(0, 0), Img_input.size());
  cv::Scalar color(0, 0, 0);
  int thickness = max(imgBorders[0], imgBorders[1]);

  cv::rectangle(histStretch, border, color, thickness);  

  if (FIGURE_1) {
    // Create a window for display.
    namedWindow("img", cv::WINDOW_NORMAL);
    imshow("img", histStretch);
  }
  
/* ======================================================================= *
 * Points detection                                                        *
 * ======================================================================= */

  clock_t start = clock();

/* ----------------------------------------------------------------------- *
 * Median filter                                                           *
 * ----------------------------------------------------------------------- */

  start = clock();
  
  int kerlen = 3;
  Mat medianImg = medianFilter(histStretch, kerlen);
  //backgroundSub.release();

  if (TIME_STAMP) timeElapsed(start, "Median filter");
  fprintf(pFile, "End Median filter\n");


/* ----------------------------------------------------------------------- *
 * Binarization                                                            *
 * ----------------------------------------------------------------------- */

  start = clock();

  double level = 150;//150 / 255
  Mat binaryImg = binarization(medianImg, level);
  medianImg.release();

  if (TIME_STAMP) timeElapsed(start, "Binarization");
  fprintf(pFile, "End Binarization\n");


/* ----------------------------------------------------------------------- *
 * Convolution kernel                                                      *
 * ----------------------------------------------------------------------- */

  start = clock();

  int szKernel = 3;
  Mat kernel = Mat::ones(szKernel, szKernel, CV_8U);
  double threshConv = 6;// szKernel*szKernel;
  
  Mat convImg = convolution(binaryImg, kernel, threshConv);
  binaryImg.release();

  if (TIME_STAMP) timeElapsed(start, "Convolution");
  fprintf(pFile, "End Convolution kernel\n");
  

/* ----------------------------------------------------------------------- *
 * Morphology opening                                                      *
 * ----------------------------------------------------------------------- */

  start = clock();

  int radDisk = 6;  
  Mat openImg = morphologyOpen(convImg, radDisk);

  if (TIME_STAMP) timeElapsed(start, "Morphology opening");
  fprintf(pFile, "End Morphology opening kernel\n");
  
  
/* ======================================================================= *
 * Streaks detection                                                       *
 * ======================================================================= */
  
/* ----------------------------------------------------------------------- *
 * Hough transform                                                         *
 * ----------------------------------------------------------------------- */
#if 1
  start = clock();

  Mat resizeImg;
  double f = 0.5;
  Size dsize = { 0, 0 };
  resize(convImg, resizeImg, dsize, f, f, INTER_LINEAR);
  
  std::vector<std::pair<float, int>> countAngle = hough(resizeImg);
  convImg.release();
#endif

/* ----------------------------------------------------------------------- *
 * Connected components                                                    *
 * ----------------------------------------------------------------------- */

  std::vector< cv::Vec<int, 3> > POINTS;

  std::vector< cv::Vec<int, 3> > STREAKS;
  
  connectedComponents(openImg, imgBorders, POINTS, STREAKS);

/* ----------------------------------------------------------------------- *
 * Plot result                                                             *
 * ----------------------------------------------------------------------- */

  if (FIGURE)
  {
    Mat color_Img_input;
    cvtColor( histStretch, color_Img_input, CV_GRAY2BGR );

    Img_input.release();

    int radius = 15;
    Scalar colorP = {0,255,0};
    Scalar colorS = {0,0,255};
    int thickness = -1;
    int lineType = 8;
    int shift = 0;

    std::cout << "Detected points: " << POINTS.size() << std::endl;
    for (size_t i = 0; i < POINTS.size(); ++i)
    {
      Point center = { POINTS.at(i)[0], POINTS.at(i)[1] };
      circle(color_Img_input, center, radius, colorP, thickness, lineType, shift);

      /*center = { STREAKS.at(i)[0], STREAKS.at(i)[1] };
      circle(color_Img_input, center, radius, color, thickness, lineType, shift);*/
    }

    std::cout << "Detected streaks: " << STREAKS.size() << std::endl;
    for (size_t i = 0; i < STREAKS.size(); ++i)
    {
      Point center = { STREAKS.at(i)[0], STREAKS.at(i)[1] };
      circle(color_Img_input, center, radius, colorS, thickness, lineType, shift);
      std::cout << "Centroid streaks: " << STREAKS.at(i)[0] << " " << STREAKS.at(i)[1] << std::endl;
    }

    // Create a window for display.
    namedWindow("Algo simple", WINDOW_NORMAL);
    imshow("Algo simple", color_Img_input);
  }

  fclose(pFile);
  
  //cv::waitKey(0);
  
  return 0;
}
