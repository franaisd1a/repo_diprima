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

/* ----------------------------------------------------------------------- *
 * Open and read file                                                      *
 * ----------------------------------------------------------------------- */

  clock_t start = clock();
      
  /* Read file extension */
  char* fileName[256];
  std::vector<char*> vec = fileExt(nameFile);
  char* ext = vec.at(1);
  const char* extjpg = "jpg";
  const char* extJPG = "JPG";
  const char* extfit = "fit";
  const char* extFIT = "FIT";
  
  /* Open log file */
#if SPD_STAMP
# if SPD_STAMP_FILE_INFO
  char s_infoFileName[256];
  strcpy (s_infoFileName, vec.at(0));
  strcat ( s_infoFileName, "_info.txt" );
  std::ofstream infoFile(s_infoFileName);
# else
  std::ofstream infoFile(stdout);
# endif
#else
  std::ofstream infoFile(stderr);
#endif
  
  /* Read image */
  Mat Img_input;

  if ( (0==strcmp(ext, extJPG)) || (0==strcmp(ext, extjpg)) ) {
    // Read file
    Img_input = imread(nameFile, CV_LOAD_IMAGE_GRAYSCALE);

    // Check for invalid file
    if (!Img_input.data) {
      printf("Error: could not open or find the image.");
      return -1;
    }
  }
  else if ( (0==strcmp(ext, extFIT)) || (0==strcmp(ext, extfit)) ) {
    readFit(nameFile, infoFile, Img_input);
  } else {
    printf("Error in reading process.");
    return -1;
  }

  int channels = Img_input.channels();
  int depth = Img_input.depth();
#if SPD_STAMP
  std::string s_Ch = "Image channels: " + std::to_string(channels);
  stamp(infoFile, s_Ch.c_str());
  std::string s_Dp = "Image depth bit: " + std::to_string(depth);
  stamp(infoFile, s_Dp.c_str());
#endif

  cv::Point_<int> I_input_size = { Img_input.cols, Img_input.rows  };
  double bordersThick = 0.015;
  cv::Point_<double> borders = { bordersThick, 1-bordersThick };
  Vec<int, 4> imgBorders = {static_cast<int>(ceil( borders.x * I_input_size.x))
                          , static_cast<int>(ceil( borders.x * I_input_size.y))
                          , static_cast<int>(floor(borders.y * I_input_size.x))
                          , static_cast<int>(floor(borders.y * I_input_size.y))};  
    
#if SPD_STAMP
  timeElapsed(infoFile, start, "Open and read file");
#endif


/* ----------------------------------------------------------------------- *
 * Histogram Stretching                                                    *
 * ----------------------------------------------------------------------- */

  start = clock();

  Mat histStretch;
  if ( (0==strcmp(ext, extJPG)) || (0==strcmp(ext, extjpg)) )
  {
    histStretch = Img_input;
  }
  else if ( (0==strcmp(ext, extFIT)) || (0==strcmp(ext, extfit)) )
  {
    histStretch = histogramStretching(Img_input);
  }
  Img_input.release();

#if SPD_STAMP
  timeElapsed(infoFile, start, "Histogram Stretching");
#endif


/* ----------------------------------------------------------------------- *
 * Set image borders with zero                                             *
 * ----------------------------------------------------------------------- */
#if 0
  cv::Rect border(cv::Point(0, 0), Img_input.size());
  cv::Scalar color(0, 0, 0);
  int thickness = max(imgBorders[0], imgBorders[1]);

  cv::rectangle(histStretch, border, color, thickness);  

#if SPD_FIGURE_1
    // Create a window for display.
    namedWindow("img", cv::WINDOW_NORMAL);
    imshow("img", histStretch);
#endif
#endif

/* ======================================================================= *
 * Points detection                                                        *
 * ======================================================================= */
  
/* ----------------------------------------------------------------------- *
 * Median filter                                                           *
 * ----------------------------------------------------------------------- */

  start = clock();
  
  int kerlen = 3;
  Mat medianImg = medianFilter(histStretch, kerlen);
  //backgroundSub.release();

#if SPD_STAMP
  timeElapsed(infoFile, start, "Median filter");
#endif


/* ----------------------------------------------------------------------- *
 * Binarization                                                            *
 * ----------------------------------------------------------------------- */

  start = clock();

  double level = 150;
  Mat binaryImg = binarization(medianImg, level);
  medianImg.release();

#if SPD_STAMP
  timeElapsed(infoFile, start, "Binarization");
#endif


/* ----------------------------------------------------------------------- *
 * Convolution kernel                                                      *
 * ----------------------------------------------------------------------- */

  start = clock();

  int szKernel = 3;
  Mat kernel = Mat::ones(szKernel, szKernel, CV_8U);
  double threshConv = 6;
  
  Mat convImg = convolution(binaryImg, kernel, threshConv);
  binaryImg.release();

#if SPD_STAMP
  timeElapsed(infoFile, start, "Convolution");
#endif


/* ----------------------------------------------------------------------- *
 * Morphology opening                                                      *
 * ----------------------------------------------------------------------- */

  start = clock();

  int radDisk = 6;  
  Mat openImg = morphologyOpen(convImg, radDisk);

#if SPD_STAMP
  timeElapsed(infoFile, start, "Morphology opening");
#endif
  

/* ======================================================================= *
 * Streaks detection                                                       *
 * ======================================================================= */
  
/* ----------------------------------------------------------------------- *
 * Hough transform                                                         *
 * ----------------------------------------------------------------------- */

  start = clock();

  Mat resizeImg;
  double f = 0.5;
  Size dsize = { 0, 0 };
  resize(convImg, resizeImg, dsize, f, f, INTER_LINEAR);
  
  std::vector<std::pair<float, int>> angle = hough(resizeImg);
  resizeImg.release();
  
#if SPD_STAMP
  std::string s_nH = "Number of inclination angles: " + std::to_string(angle.size());
  stamp(infoFile, s_nH.c_str());
  timeElapsed(infoFile, start, "Hough transform");
#endif


/* ----------------------------------------------------------------------- *
 * Sum streaks binary image                                                *
 * ----------------------------------------------------------------------- */

  start = clock();
  printf("pre crea matrice\n");
  cv::Mat sumStrImg = cv::Mat::zeros(histStretch.rows, histStretch.cols, CV_8U);
  printf("post crea matrice\n");
  for (int i = 0; i < angle.size(); ++i)
  {
/* ----------------------------------------------------------------------- *
 * Morphology opening with linear kernel                                   *
 * ----------------------------------------------------------------------- */
    printf("ciclo %d\n\n",i);
    int dimLine = 21;

    Mat morpOpLin = morphologyOpen(convImg, dimLine, angle.at(i).first);

/* ----------------------------------------------------------------------- *
 * Convolution with linear kernel                                          *
 * ----------------------------------------------------------------------- */

    Mat kernelL = linearKernel(dimLine, angle.at(i).first);
    double threshConvL = 9;

    Mat convStreak = convolution(morpOpLin, kernelL, threshConvL);

    sumStrImg = sumStrImg + convStreak;
  }

#if SPD_FIGURE_1
  namedWindow("Final binary image", cv::WINDOW_NORMAL);
  imshow("Final binary image", sumStrImg);
#endif
  
#if SPD_STAMP
  timeElapsed(infoFile, start, "Sum streaks binary");
#endif


/* ----------------------------------------------------------------------- *
 * Connected components                                                    *
 * ----------------------------------------------------------------------- */

  start = clock();

  std::vector< cv::Vec<int, 3> > POINTS;
  std::vector< cv::Vec<int, 3> > STREAKS;
  
  connectedComponents(openImg, sumStrImg, imgBorders, POINTS, STREAKS);
  openImg.release();
  sumStrImg.release();

#if SPD_STAMP
  timeElapsed(infoFile, start, "Connected components");
#endif


/* ----------------------------------------------------------------------- *
 * Write result                                                             *
 * ----------------------------------------------------------------------- */

#if SPD_STAMP
# if SPD_STAMP_FILE_RESULT
  /* Open result file */
  char s_resFileName[256];
  strcpy (s_resFileName, vec.at(0));
  strcat ( s_resFileName, ".txt" );
  std::ofstream resFile(s_resFileName);
# else
  std::ofstream resFile(stdout);
# endif
  writeResult(resFile, POINTS, STREAKS);
#endif
  
/* ----------------------------------------------------------------------- *
 * Plot result                                                             *
 * ----------------------------------------------------------------------- */

  if (SPD_FIGURE)
  {
    Mat color_Img_input;
    cvtColor( histStretch, color_Img_input, CV_GRAY2BGR );//histStretch

    Img_input.release();

    int radius = 11;
    Scalar colorP = {0,255,0};
    Scalar colorS = {0,0,255};
    int thickness = -1;
    int lineType = 8;
    int shift = 0;

    for (size_t i = 0; i < POINTS.size(); ++i) {
      Point center = { POINTS.at(i)[0], POINTS.at(i)[1] };
      circle(color_Img_input, center, radius, colorP, thickness, lineType, shift);
    }
    for (size_t i = 0; i < STREAKS.size(); ++i) {
      Point center = { STREAKS.at(i)[0], STREAKS.at(i)[1] };
      circle(color_Img_input, center, radius, colorS, thickness, lineType, shift);
    }

#if SPD_FIGURE
    namedWindow("Algo simple", WINDOW_NORMAL);
    imshow("Algo simple", color_Img_input);
#endif
#if SPD_FILE
    char s_imgName[256];
    strcpy(s_imgName, vec.at(0));
    strcat(s_imgName, ".jpg");
    imwrite( s_imgName, color_Img_input );
#endif
  }
  return 0;
}
