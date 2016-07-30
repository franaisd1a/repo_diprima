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

/* ==========================================================================
* MODULE PRIVATE MACROS
* ========================================================================== */
/*#define FIGURE 1U
#define FIGURE_1 0U
#define FILE 1U
#define CLEAR 0U
#define BACKGROUND_SUBTRACTION 1U
#define DIFFERENT_THRESHOLD 1U
#define FIT 1U
#define DILATE 1U*/

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
  // Check for invalid input
  if (argc != 2)  {
    cout << "Error: insert input argument." << endl;
    return -1;
  }

  // File name
  //char* name_file = "C:\\Users\\diprima\\Desktop\\scontoMOTO.PNG";
  char* name_file = argv[1];

  // Read file
  Mat Img_input = imread(name_file, CV_LOAD_IMAGE_GRAYSCALE );
    
  // Check for invalid file
  if (!Img_input.data)  {
    cout << "Error: could not open or find the image." << std::endl;
    return -1;
  }
  
  int channels = Img_input.channels();
  int depth = Img_input.depth();
  int I_input_size[2] = { Img_input.cols, Img_input.rows  };
  double borders[2] = { 0.015, 0.985 };
  int imgBorders[2] = { static_cast<int>(ceil(borders[0] * I_input_size[0]))
                     , static_cast<int>(floor(borders[1] * I_input_size[1])) };
  
  if (FIGURE)
  {
    // Create a window for display.
    namedWindow("Display window", WINDOW_NORMAL);
    imshow("Display window", Img_input);
  }


  /* ======================================================================= *
   * Big Points detection                                                    *
   * ======================================================================= */

  /* ----------------------------------------------------------------------- *
   * Gaussian filter                                                         *
   * ----------------------------------------------------------------------- */

  int hsize[2] = {101, 101};
  double sigma = 30;
  Mat gaussImg = gaussianFilter(Img_input, hsize, sigma);


  /* ----------------------------------------------------------------------- *
   * Background subtraction                                                  *
   * ----------------------------------------------------------------------- */

  Mat backgroundSub = Img_input - gaussImg;


  /* ----------------------------------------------------------------------- *
   * Median filter                                                           *
   * ----------------------------------------------------------------------- */

  int kerlen = 11;
  Mat medianImg = medianFilter(backgroundSub, kerlen);


  /* ----------------------------------------------------------------------- *
   * Binarization                                                            *
   * ----------------------------------------------------------------------- */

  Mat binaryImg = binarization(medianImg);


  /* ----------------------------------------------------------------------- *
   * Convolution kernel                                                      *
   * ----------------------------------------------------------------------- */

  int szKernel = 3;
  Mat kernel = Mat::ones(szKernel, szKernel, CV_8U);
  double threshConv = szKernel*szKernel;
  Mat convImg = convolution(binaryImg, kernel, threshConv);
  

  /* ----------------------------------------------------------------------- *
   * Connected components                                                    *
   * ----------------------------------------------------------------------- */

  connectedComponents(convImg, imgBorders);


  /* ----------------------------------------------------------------------- *
   * Morphology opening                                                      *
   * ----------------------------------------------------------------------- */


  waitKey(0);
  return 0;
}
