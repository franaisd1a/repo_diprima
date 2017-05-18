/* ==========================================================================
* ALGO STREAK DETECTION
* ========================================================================== */

/* ==========================================================================
* MODULE FILE NAME: main_sigmaClipSimpleBig.cpp
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
#include <cstdio>
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
*        FUNCTION NAME: main_sigmaClipSimpleBig
* FUNCTION DESCRIPTION:
*        CREATION DATE: 20160727
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
int main_sigmaClipSimpleBig(const std::vector<char *>& input)
{
/* ----------------------------------------------------------------------- *
 * Initialization                                                          *
 * ----------------------------------------------------------------------- */

  /* Read file extension */
  const char* extjpg = "jpg";
  const char* extJPG = "JPG";
  const char* extfit = "fit";
  const char* extFIT = "FIT";
  
  /* Open log file */
# if SPD_STAMP_FILE_INFO
  char s_infoFileName[1024];
  ::strcpy (s_infoFileName, input.at(4));
  ::strcat (s_infoFileName, input.at(1));
  ::strcat (s_infoFileName, "_info.txt" );
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


/* ----------------------------------------------------------------------- *
 * Solve star field                                                        *
 * ----------------------------------------------------------------------- */

  clock_t start = clock();
  clock_t start0 = start;

  wcsPar par;
#if 1
  bool compPar = astrometry( input, par);
#else  
  std::future<bool> fut_astrometry = asyncAstrometry(input, par);
#endif

  timeElapsed(infoFile, start, "Astrometry");


/* ----------------------------------------------------------------------- *
 * Open and read file                                                      *
 * ----------------------------------------------------------------------- */
  
  /* Read raw image */
  Mat Img_input;

  if ( (0==strcmp(input.at(2), extJPG)) || (0==strcmp(input.at(2), extjpg)) ) {
    // Read file
    Img_input = imread(input.at(0), CV_LOAD_IMAGE_GRAYSCALE);

    // Check for invalid file
    if (!Img_input.data) {
      ::printf("Error: could not open or find the image.");
      return -1;
    }
  }
  else if ( (0==strcmp(input.at(2), extFIT)) || (0==strcmp(input.at(2), extfit)) ) {
    readFit(input.at(0), infoFile, Img_input);
  } else {
    ::printf("Error in reading process.");
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

  std::vector< cv::Vec<float, 3> > POINTS;
  std::vector< cv::Vec<float, 3> > STREAKS;

  size_t maxColdim = 4099;
  size_t maxRowdim = 4099;

  size_t regionNumR = static_cast<size_t>(::round(histStretch.rows / maxRowdim));
  size_t regionNumC = static_cast<size_t>(::round(histStretch.cols / maxColdim));

  /* Odd dimensions */
  if (0 == regionNumR % 2) {
    regionNumR = regionNumR + 1;
  }
  if (0 == regionNumC % 2) {
    regionNumC = regionNumC + 1;
  }

  size_t regionDimR = static_cast<size_t>(::round(histStretch.rows / regionNumR));
  size_t regionDimC = static_cast<size_t>(::round(histStretch.cols / regionNumC));

  std::vector<int> vRegRow;
  std::vector<int> vRegCol;

  for (size_t o = 0; o < regionNumR; ++o)
  {
    vRegRow.push_back(regionDimR*o);
  }
  vRegRow.push_back(histStretch.rows);

  for (size_t o = 0; o < regionNumC; ++o)
  {
    vRegCol.push_back(regionDimC*o);
  }
  vRegCol.push_back(histStretch.cols);
    
  for (size_t i = 0; i < regionNumR; ++i)
  {
    for (size_t j = 0; j < regionNumC; ++j)
    {
      const cv::Point ptTL = { vRegCol.at(j), vRegRow.at(i) };
      const cv::Point ptBR = { vRegCol.at(j + 1), vRegRow.at(i + 1) };

      cv::Rect region_of_interest = cv::Rect(ptTL, ptBR);
      cv::Mat histStretchPart = histStretch(region_of_interest);
      cv::Mat Img_inputPart = Img_input(region_of_interest);

      std::vector< cv::Vec<float, 3> > localPOINTS;
      std::vector< cv::Vec<float, 3> > localSTREAKS;

/******************************************************************************/
      sigmaClipProcessing(histStretchPart, Img_inputPart, infoFile
        , localPOINTS, localSTREAKS);
/******************************************************************************/

      for (size_t p = 0; p < localPOINTS.size(); ++p)
      {
        localPOINTS.at(p)[0] = localPOINTS.at(p)[0] + ptTL.x;
        localPOINTS.at(p)[1] = localPOINTS.at(p)[1] + ptTL.y;
      }
      for (size_t p = 0; p < localSTREAKS.size(); ++p)
      {
        localSTREAKS.at(p)[0] = ::round(localSTREAKS.at(p)[0] + ptTL.x);//::round(
        localSTREAKS.at(p)[1] = ::round(localSTREAKS.at(p)[1] + ptTL.y);//::round(
      }
      
      if (localPOINTS.size()>0) {
        POINTS.insert(POINTS.end(), localPOINTS.begin(), localPOINTS.end());
      }
      if (localSTREAKS.size()>0) {
        STREAKS.insert(STREAKS.end(), localSTREAKS.begin(), localSTREAKS.end());
      }
    }
  }


/* ----------------------------------------------------------------------- *
 * Coordinate conversion                                                   *
 * ----------------------------------------------------------------------- */

  std::vector< cv::Vec<double, 3> > radecS;
  std::vector< cv::Vec<double, 3> > radecP;

  if (0!=STREAKS.size() || 0!=POINTS.size())
  {
#if 0
    fut_astrometry.wait();
    bool compPar = fut_astrometry.get();    
#endif
    if (compPar) 
    {
      if (0 != STREAKS.size()) {
        coordConv(par, STREAKS, radecS);
      }
      if (0 != POINTS.size()) {
        coordConv(par, POINTS, radecP);
      }
    }
    else
    {
      for (size_t u = 0; u < STREAKS.size(); ++u) {
        radecS.push_back({ 0.0,0.0,0.0 });
      }
      for (size_t u = 0; u < POINTS.size(); ++u) {
        radecP.push_back({ 0.0,0.0,0.0 });
      }
    }
  }
  else
  {
    //fut_astrometry._Abandon();
  }


/* ----------------------------------------------------------------------- *
 * Write result                                                            *
 * ----------------------------------------------------------------------- */

#if SPD_STAMP_FILE_RESULT
  /* Open result file */
  char s_resFileName[256];
  ::strcpy (s_resFileName, input.at(4));
  ::strcat (s_resFileName, input.at(1));
  ::strcat (s_resFileName, ".txt" );
  std::ofstream resFile(s_resFileName);
# else
  std::ofstream resFile(stdout);
# endif

  writeResult(resFile, POINTS, STREAKS, radecP, radecS);

  
/* ----------------------------------------------------------------------- *
 * Plot result                                                             *
 * ----------------------------------------------------------------------- */

  plotResult(histStretch, POINTS, STREAKS, input);
    
  destroyAllWindows();
  infoFile.close();
  resFile.close();
  
  timeElapsed(infoFile, start0, "Total");

  return 0;
}
