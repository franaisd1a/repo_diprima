/* ==========================================================================
* ALGO STREAK DETECTION
* ========================================================================== */

/* ==========================================================================
* MODULE FILE NAME: main_sigmaClip.cpp
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
*        FUNCTION NAME: main_sigmaClip
* FUNCTION DESCRIPTION:
*        CREATION DATE: 20160727
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
int main_sigmaClipSimple(const std::vector<char *>& input)
{
#if 0
	int as0d = system ("pwd");

#ifdef _WIN32
	std::string astroScript = "launcherWIN.bat ";
#else
  std::string astroScript = "./astrometricReduction.sh ";
#endif
	std::string command = astroScript + input.at(0) + " " + input.at(4) + " " + input.at(1);
	printf("\n%s\n", command.c_str());
	int asd = system (command.c_str());

	printf("fine astrometry \d\n", asd);
#endif


/* ----------------------------------------------------------------------- *
 * Open and read file                                                      *
 * ----------------------------------------------------------------------- */

  clock_t start = clock();

  /* Read file extension */
  const char* extjpg = "jpg";
  const char* extJPG = "JPG";
  const char* extfit = "fit";
  const char* extFIT = "FIT";
  
  /* Open log file */
# if SPD_STAMP_FILE_INFO
  char s_infoFileName[1024];
  strcpy (s_infoFileName, input.at(4));
  strcat (s_infoFileName, input.at(1));
  strcat (s_infoFileName, "_info.txt" );
  std::ofstream infoFile(s_infoFileName);
# else
  std::ofstream infoFile(stdout);
# endif

  {
    std::string s_Ch = "File name: ";
    s_Ch += input.at(0);
    stamp(infoFile, s_Ch.c_str());
  }  
  {
    std::string s_Ch = "Result folder path: ";
    s_Ch += input.at(4);
    stamp(infoFile, s_Ch.c_str());
    stamp(infoFile, "\n");
  }

  /* Read raw image */
  Mat Img_input;

  if ( (0==strcmp(input.at(2), extJPG)) || (0==strcmp(input.at(2), extjpg)) ) {
    // Read file
    Img_input = imread(input.at(0), CV_LOAD_IMAGE_GRAYSCALE);

    // Check for invalid file
    if (!Img_input.data) {
      printf("Error: could not open or find the image.");
      return -1;
    }
  }
  else if ( (0==strcmp(input.at(2), extFIT)) || (0==strcmp(input.at(2), extfit)) ) {
    readFit(input.at(0), infoFile, Img_input);
  } else {
    printf("Error in reading process.");
    return -1;
  }

  int channels = Img_input.channels();
  int depth = Img_input.depth();

  std::string s_Ch = "Image channels: " + std::to_string(channels);
  stamp(infoFile, s_Ch.c_str());
  std::string s_Dp = "Image depth bit: " + std::to_string(depth);
  stamp(infoFile, s_Dp.c_str());

  timeElapsed(infoFile, start, "Open and read file");


/* ----------------------------------------------------------------------- *
 * Histogram Stretching                                                    *
 * ----------------------------------------------------------------------- */

  start = clock();

  Mat histStretch;
  if ( (0==strcmp(input.at(2), extJPG)) || (0==strcmp(input.at(2), extjpg)) )
  {
    histStretch = Img_input;
  }
  else if ( (0==strcmp(input.at(2), extFIT)) || (0==strcmp(input.at(2), extfit)) )
  {
    histStretch = histogramStretching(Img_input);
  }
  
  timeElapsed(infoFile, start, "Histogram Stretching");
  cv::waitKey(0);


/***************************************************************************/
/*                               Processing                                */
/***************************************************************************/
    

/* ======================================================================= *
 * Points detection                                                        *
 * ======================================================================= */
  
/* ----------------------------------------------------------------------- *
 * Median filter                                                           *
 * ----------------------------------------------------------------------- */

  start = clock();
  
  int kerlenSz = 3;
  Mat medianImg = medianFilter(histStretch, kerlenSz);

  timeElapsed(infoFile, start, "Median filter");
  cv::waitKey(0);


/* ----------------------------------------------------------------------- *
 * Background estimation                                                   *
 * ----------------------------------------------------------------------- */

  start = clock();

  int backCnt = 5;
  const cv::Point vBackCnt = {backCnt, backCnt};
  cv::Mat meanBg = cv::Mat::zeros(backCnt, backCnt, CV_64F);
  cv::Mat  stdBg = cv::Mat::zeros(backCnt, backCnt, CV_64F);

  cv::Mat backgroungImg = 
    backgroundEstimation(medianImg, vBackCnt, meanBg, stdBg);

  timeElapsed(infoFile, start, "Background estimation");
  cv::waitKey(0);
  
/* ----------------------------------------------------------------------- *
 * Background subtraction                                                  *
 * ----------------------------------------------------------------------- */

  start = clock();

  cv::Mat bgSubtracImg = subtraction(medianImg, backgroungImg);
    
  medianImg.release();
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

  start = clock();
  
  Mat medianBgSubImg = medianFilter(bgSubtracImg, kerlenSz);
  bgSubtracImg.release();

  timeElapsed(infoFile, start, "Median filter");
  cv::waitKey(0);
#if 0
  {
    char s_imgName[256];
    strcpy(s_imgName, input.at(4));
    strcat(s_imgName, input.at(1));
    strcat(s_imgName, "medianBgSubImg");
    strcat(s_imgName, ".jpg");
    imwrite(s_imgName, medianBgSubImg);
  }
#endif


/* ----------------------------------------------------------------------- *
 * Binarization for points detection                                       *
 * ----------------------------------------------------------------------- */

  start = clock();

  cv::Mat level = cv::Mat::zeros(backCnt, backCnt, CV_64F);
  level = meanBg + 3.5*stdBg;
  
  Mat binaryImgPnt = binarizationZone(medianBgSubImg, vBackCnt, level);
  
  timeElapsed(infoFile, start, "Binarization");
  cv::waitKey(0);


/* ----------------------------------------------------------------------- *
 * Binarization for streaks detection                                      *
 * ----------------------------------------------------------------------- */

  start = clock();

  cv::Mat levelStk = cv::Mat::zeros(backCnt, backCnt, CV_64F);
  levelStk = meanBg + 1*stdBg;//2.8
  
  cv::Mat binaryImgStk = binarizationZone(medianBgSubImg, vBackCnt, levelStk);
  medianBgSubImg.release();

  timeElapsed(infoFile, start, "Binarization for streaks detection");
  cv::waitKey(0);
  

/* ----------------------------------------------------------------------- *
 * Distance transformation for streaks detection                           *
 * ----------------------------------------------------------------------- */

  start = clock();
    
  cv::Mat distStk = distTransform(binaryImgStk);
  binaryImgStk.release();

#if 0
  {
    char s_imgName[256];
    strcpy(s_imgName, input.at(4));
    strcat(s_imgName, input.at(1));
    strcat(s_imgName, "distStk");
    strcat(s_imgName, ".jpg");
    imwrite(s_imgName, distStk);
  }
#endif 


/* ----------------------------------------------------------------------- *
 * Convolution kernel for points detection                                 *
 * ----------------------------------------------------------------------- */

  start = clock();

  int szKernel = 3;
  Mat kernel = Mat::ones(szKernel, szKernel, CV_8U);
  double threshConv = 7;//6
  
  Mat convImgPnt = convolution(binaryImgPnt, kernel, threshConv);
  binaryImgPnt.release();
  timeElapsed(infoFile, start, "Convolution for points detection");
  cv::waitKey(0);
  

/* ----------------------------------------------------------------------- *
 * Morphology opening                                                      *
 * ----------------------------------------------------------------------- */

  start = clock();

  int radDisk = 6;  
  Mat openImg = morphologyOpen(convImgPnt, radDisk);

  timeElapsed(infoFile, start, "Morphology opening");
  
  cv::waitKey(0);
#if 0
  {
    char s_imgName[256];
    strcpy(s_imgName, input.at(4));
    strcat(s_imgName, input.at(1));
    strcat(s_imgName, "openImg");
    strcat(s_imgName, ".jpg");
    imwrite(s_imgName, openImg);
  }
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
  resize(distStk, resizeImg, dsize, f, f, INTER_LINEAR);

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

  cv::Mat sumStrRemImg = cv::Mat::zeros(histStretch.rows, histStretch.cols, CV_8U);

  for (size_t i = 0; i < angle.size(); ++i)
  {

/* ----------------------------------------------------------------------- *
 * Morphology opening with linear kernel for remove streaks                *
 * ----------------------------------------------------------------------- */
    
    int dimLineRem = 60;

    Mat morpOpLinRem = morphologyOpen(openImg, dimLineRem, angle.at(i).first);
    cv::waitKey(0);


/* ----------------------------------------------------------------------- *
 * Binary image with streaks                                               *
 * ----------------------------------------------------------------------- */

    sumStrRemImg = sumStrRemImg + morpOpLinRem;
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
  
  cv::Mat onlyPoints = openImg - sumStrRemImg;
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

  Mat kernelRm = Mat::ones(szKernel, szKernel, CV_8U);
  double threshConvRm = 8;
  
  Mat convImgRms = convolution(onlyPoints, kernelRm, threshConvRm);
  kernelRm.release();
  onlyPoints.release();
  cv::waitKey(0);

  timeElapsed(infoFile, start, "Convolution");
    
  
/* ----------------------------------------------------------------------- *
 * Connected components                                                    *
 * ----------------------------------------------------------------------- */

  start = clock();

  std::vector< cv::Vec<float, 3> > POINTS;
  std::vector< cv::Vec<float, 3> > STREAKS;
  std::vector<std::vector<cv::Point > > contoursP;
  std::vector<std::vector<cv::Point > > contoursS;

#if 0
    char s_imgNamePnt[256];
    strcpy(s_imgNamePnt, input.at(4));
    strcat(s_imgNamePnt, input.at(1));
    strcat(s_imgNamePnt, "Pnt.jpg");
    imwrite( s_imgNamePnt, convImgRms );    
#endif

  connectedComponents2(convImgRms, distStk, Img_input, POINTS, STREAKS, contoursP, contoursS);
#if 0
  {
    char s_imgName[256];
    strcpy(s_imgName, input.at(4));
    strcat(s_imgName, input.at(1));
    strcat(s_imgName, "convImgRms");
    strcat(s_imgName, ".jpg");
    imwrite(s_imgName, convImgRms);
  }
  {
    char s_imgName[256];
    strcpy(s_imgName, input.at(4));
    strcat(s_imgName, input.at(1));
    strcat(s_imgName, "distStk");
    strcat(s_imgName, ".jpg");
    imwrite(s_imgName, distStk);
  }
#endif
  convImgRms.release();  
  Img_input.release();

  timeElapsed(infoFile, start, "Connected components");


/* ----------------------------------------------------------------------- *
 * Write result                                                             *
 * ----------------------------------------------------------------------- */

#if SPD_STAMP_FILE_RESULT
  /* Open result file */
  char s_resFileName[256];
  strcpy (s_resFileName, input.at(4));
  strcat (s_resFileName, input.at(1));
  strcat (s_resFileName, ".txt" );
  std::ofstream resFile(s_resFileName);
# else
  std::ofstream resFile(stdout);
# endif

  writeResult(resFile, POINTS, STREAKS);

  
/* ----------------------------------------------------------------------- *
 * Plot result                                                             *
 * ----------------------------------------------------------------------- */

  plotResult(histStretch, POINTS, STREAKS, input);
    
  destroyAllWindows();
  
  return 0;
}
