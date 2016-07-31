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
#include "function.h"

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
cv::Mat gaussianFilter(cv::Mat& imgIn, int hsize[2], double sigma)
{
  cv::Mat imgOut;

  cv::Size h = { hsize[0], hsize[1] };

  GaussianBlur(imgIn, imgOut, h, sigma, sigma, cv::BORDER_DEFAULT);
  
  if (FIGURE_1)
  {
    // Create a window for display.
    namedWindow("Gaussain filter", cv::WINDOW_NORMAL);
    imshow("Gaussain filter", imgOut);
  }

  return imgOut;
}

/* ==========================================================================
*        FUNCTION NAME: medianFilter
* FUNCTION DESCRIPTION: Median filter
*        CREATION DATE: 20160727
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
cv::Mat medianFilter(cv::Mat& imgIn, int kerlen)
{
  cv::Mat imgOut;

  medianBlur(imgIn, imgOut, kerlen);

  if (FIGURE_1)
  {
    // Create a window for display.
    namedWindow("Median filter", cv::WINDOW_NORMAL);
    imshow("Median filter", imgOut);
  }

  return imgOut;
}

/* ==========================================================================
*        FUNCTION NAME: medianFilter
* FUNCTION DESCRIPTION: Median filter
*        CREATION DATE: 20160727
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
cv::Mat medianFilter(cv::Mat& imgIn, int littleKerlen, int bigKerlen)
{
  cv::Mat imgOut, imgBigKer;

  medianBlur(imgIn, imgOut, littleKerlen);
  medianBlur(imgIn, imgBigKer, bigKerlen);

  imgOut = imgOut - imgBigKer;

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
cv::Mat morphologyOpen(cv::Mat& imgIn, int dimLine, double teta_streak)
{
  cv::Mat imgOut;

  int iter = 1;
  cv::Point anchor = cv::Point(-1, -1);

  //InputArray kernel;
  cv::Mat horizontalStructure = getStructuringElement(cv::MORPH_RECT, cv::Size(dimLine, 1));

  morphologyEx(imgIn, imgOut, cv::MORPH_OPEN, horizontalStructure, anchor, iter
    , cv::BORDER_CONSTANT, cv::morphologyDefaultBorderValue());

  if (FIGURE_1)
  {
    // Create a window for display.
    namedWindow("Morphology opening with rectangular kernel", cv::WINDOW_NORMAL);
    imshow("Morphology opening with rectangular kernel", imgOut);
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
cv::Mat binarization(cv::Mat& imgIn)
{
  cv::Mat imgOut, binImg;
    
  double maxval = 255.0;
  double level = 0.0;
  
  level = threshold(imgIn, binImg, cv::THRESH_OTSU, maxval, cv::THRESH_BINARY);

  level = level * 1.5;
  
  threshold(imgIn, imgOut, level, maxval, cv::THRESH_BINARY);
  
  if (FIGURE_1)
  {
    /*/ Create a window for display.
    namedWindow("Binary image", WINDOW_NORMAL);
    imshow("Binary image", binImg);*/

    // Create a window for display.
    namedWindow("Binary image Otsu threshold", cv::WINDOW_NORMAL);
    imshow("Binary image Otsu threshold", imgOut);
  }

  return imgOut;
}

/* ==========================================================================
*        FUNCTION NAME: binarizationDiffTh
* FUNCTION DESCRIPTION: Image binarization
*        CREATION DATE: 20160727
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
cv::Mat binarizationDiffTh(cv::Mat& imgIn, int flag)
{
  cv::Mat imgOut, binImg;
  cv::Mat subBImgTL, subBImgTR, subBImgBL, subBImgBR;

  cv::Point imgSz = { imgIn.rows, imgIn.cols };

  /*int dims[] = { 5, 1 };
  cv::Mat level(2, dims, CV_64F);*/

  cv::Mat subImageTL(imgIn, cv::Rect(0, 0, imgIn.cols/2, imgIn.rows/2));
  cv::Mat subImageTR(imgIn, cv::Rect(0, imgIn.rows/2, imgIn.cols/2, imgIn.rows/2));
  cv::Mat subImageBL(imgIn, cv::Rect(0, imgIn.rows/2, imgIn.cols/2, imgIn.rows/2));
  cv::Mat subImageBR(imgIn, cv::Rect(imgIn.cols/2, imgIn.rows/2, imgIn.cols/2, imgIn.rows/2));

  
  double maxval = 1.0;
  double level1 = 0.0;
  double level2 = 0.0;
  double level3 = 0.0;
  double level4 = 0.0;
  double level5 = 0.0;

  level1 = threshold(subImageTL, subBImgTL, cv::THRESH_OTSU, maxval, cv::THRESH_BINARY);
  level2 = threshold(subImageTR, subBImgTR, cv::THRESH_OTSU, maxval, cv::THRESH_BINARY);
  level3 = threshold(subImageBL, subBImgBL, cv::THRESH_OTSU, maxval, cv::THRESH_BINARY);
  level4 = threshold(subImageBR, subBImgBR, cv::THRESH_OTSU, maxval, cv::THRESH_BINARY);
  level5 = threshold(binImg    ,    imgOut, cv::THRESH_OTSU, maxval, cv::THRESH_BINARY);

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

/* ==========================================================================
*        FUNCTION NAME: convolution
* FUNCTION DESCRIPTION: Image convolution
*        CREATION DATE: 20160727
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
cv::Mat convolution(cv::Mat& imgIn, cv::Mat& kernel, double thresh)
{
  cv::Mat imgOut, convImg;
  /*kernel_size = 3 + 2 * (ind % 5);
  kernel = Mat::ones(kernel_size, kernel_size, CV_32F) / (float)(kernel_size*kernel_size);*/

  int ddepth = -1;
  cv::Point anchor = cv::Point(-1, -1);
  double delta = 0;

  filter2D(imgIn, convImg, ddepth, kernel, anchor, delta, cv::BORDER_DEFAULT);

  double level = 0.0;
  double maxval = 255.0;

  level = cv::threshold(convImg, imgOut, thresh, maxval, cv::THRESH_BINARY);

  if (FIGURE_1)
  {
    // Create a window for display.
    namedWindow("Convolution image", cv::WINDOW_NORMAL);
    imshow("Convolution image", imgOut);

  }
  return imgOut;
}

/* ==========================================================================
*        FUNCTION NAME: connectedComponents
* FUNCTION DESCRIPTION: Found connected component
*        CREATION DATE: 20160727
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
std::vector< cv::Vec<int, 3> > connectedComponents
(
  cv::Mat& imgIn
  , cv::Vec<int, 4>& borders
)
{
  std::vector<std::vector<cv::Point> > contours;
  std::vector<cv::Vec4i> hierarchy;
  cv::Point offset = cv::Point(0, 0);
    
  findContours( imgIn, contours, hierarchy, CV_RETR_EXTERNAL
                , CV_CHAIN_APPROX_NONE , offset);

  std::vector< cv::Vec<int, 3> > POINTS;
  if (contours.size() > 0)
  {
    POINTS = connectedComponentsPoints
    (imgIn, contours, borders);
  }
  else
  {
    POINTS = { 0,0,0 };
  }

  if (FIGURE_1)
  {
    /// Draw contours
    cv::Mat drawing = cv::Mat::zeros(imgIn.size(), CV_8UC3);
    for (int i = 0; i < contours.size(); i++)
    {
      cv::Scalar color = cv::Scalar(0, 255, 0);
      drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, cv::Point());
    }

    /// Show in a window
    namedWindow("Contours", cv::WINDOW_NORMAL);
    imshow("Contours", drawing);
  }
  return POINTS;
}

/* ==========================================================================
*        FUNCTION NAME: connectedComponentsPoints
* FUNCTION DESCRIPTION: Found centroid of circular connected components
*        CREATION DATE: 20160727
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
std::vector< cv::Vec<int, 3> > connectedComponentsPoints
(
  cv::Mat& imgIn
  , std::vector<std::vector<cv::Point > >& contours
  , cv::Vec<int, 4>& borders
)
{
  int max_points_diameter = 0;
  int min_points_diameter = std::max(imgIn.cols, imgIn.rows);

  /* Initialize vector */  
  std::vector< int >        points(contours.size());
  std::vector< cv::Point >  centroid(contours.size());
  std::vector< int >        majorAxis(contours.size());
  std::vector< int >        minorAxis(contours.size());
  
  for (size_t i = 0; i < contours.size(); ++i)
  {
    cv::Point2f center;
    float radius;
    minEnclosingCircle(contours[i], center, radius);

    cv::Point centerP = { static_cast<int>(round(center.x)) 
                        , static_cast<int>(round(center.y)) };

    if(   (centerP.x>borders[0] && centerP.x<borders[2]) 
       && (centerP.y>borders[1] && centerP.y<borders[3]))
    {
      centroid.at(i) = centerP;

      cv::RotatedRect rotatedRect = fitEllipse(contours[i]);
      majorAxis.at(i) = rotatedRect.size.width;
      minorAxis.at(i) = rotatedRect.size.height;
      
      /* Identify circular connect components */
      if (majorAxis.at(i) / minorAxis.at(i) < 1.6)
      {
        points.at(i) = 1;
        if (majorAxis.at(i) > max_points_diameter)
        {
          max_points_diameter = majorAxis.at(i);
        }
        if (minorAxis.at(i) < min_points_diameter)
        {
          min_points_diameter = minorAxis.at(i);
        }
      } //if (majorAxis.at(i) / minorAxis.at(i) < 1.6)
    }
  } //for (size_t i = 0; i < contours.size(); ++i)

  int init = 0;
  int n_points = std::accumulate(points.begin(), points.end(), init);
  
  if (n_points)
  {
    for (size_t j = 0; j < contours.size(); ++j)
    {
      /* Delete little circular connect components */
      if (majorAxis.at(j)<ceil(max_points_diameter/2))
      {
        points.at(j) = 0;
      }
    }
  } //if (n_points)

  n_points = std::accumulate(points.begin(), points.end(), init);
  std::vector< cv::Vec<int, 3> >  outPOINTS(n_points);

  if (n_points)
  {
    int indx = 0;
    for (size_t k = 0; k < contours.size(); ++k)
    {
      if (1 == points.at(k))
      {
        outPOINTS.at(indx) = { centroid.at(k).x, centroid.at(k).y, 0};
        indx++;
      }
    }
  } //if (n_points)

  return outPOINTS;
}
