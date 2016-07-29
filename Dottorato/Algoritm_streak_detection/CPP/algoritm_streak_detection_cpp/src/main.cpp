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
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv\highgui.h>
#include "function.h"

/* ==========================================================================
* MODULE PRIVATE MACROS
* ========================================================================== */
#define FIGURE 1U
#define FIGURE_1 0U
#define FILE 1U
#define CLEAR 0U
#define BACKGROUND_SUBTRACTION 1U
#define DIFFERENT_THRESHOLD 1U
#define FIT 1U
#define DILATE 1U

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
*        FUNCTION NAME: main
* FUNCTION DESCRIPTION:
*        CREATION DATE: 20160727
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
int main(int argc, char** argv)
{
  if (argc != 2)
  {cout << " Usage: display_image ImageToLoadAndDisplay" << endl;
    return -1;}

  // File name
  //char* name_file = "C:\\Users\\diprima\\Desktop\\scontoMOTO.PNG";
  char* name_file = argv[1];

  // Read the file
  Mat Img_input = imread(name_file, CV_LOAD_IMAGE_COLOR);

  // Check for invalid input
  if (!Img_input.data)                              
  {
    cout << "Could not open or find the image" << std::endl;
    return -1;
  }
  
  // Create a window for display.
  namedWindow("Display window", WINDOW_AUTOSIZE);
  imshow("Display window", Img_input);














  waitKey(0);
  return 0;
}
