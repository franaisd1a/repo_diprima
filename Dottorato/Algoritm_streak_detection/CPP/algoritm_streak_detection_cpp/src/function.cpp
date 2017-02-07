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
#include "macros.h"

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
*        FUNCTION NAME: fileExt
* FUNCTION DESCRIPTION: Get file extension
*        CREATION DATE: 20160727
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
std::vector<char*> fileExt(const char* strN)
{
  std::vector<char*> vec;

  char nameFile[1024];
  strcpy ( nameFile, strN );
  char* pch;
  char *path[32][256];
  const char* slash = "\\";

  pch = strtok(nameFile,slash);
  size_t count = 0;
  while (pch != NULL)
  {
    //printf ("%s\n",pch);
    *path[count] = pch;
    pch = strtok (NULL, slash);
    count++;
  }
  char *name = *path[count-1];

  char s_pathFileName[256];
  strcpy (s_pathFileName, *path[0]);
  for (size_t i = 1; i < count - 1; ++i)
  {
    strcat(s_pathFileName, slash);
    strcat(s_pathFileName, *path[i]);
  }
  strcat(s_pathFileName, slash);
  
  char s_pathResFile[256];
  strcpy (s_pathResFile, s_pathFileName);
  strcat(s_pathResFile, "Result\\");
  
  pch = strtok(name,".");
  while (pch != NULL)
  {    
    //printf ("%s\n",pch);
    *path[count] = pch;
    pch = strtok (NULL, ".");
    count++;
  }
  char* fileName = *path[count-2];
  char* ext = *path[count-1];
  
  vec.push_back(fileName);
  vec.push_back(ext);
  vec.push_back(s_pathFileName);
  vec.push_back(s_pathResFile);

  return vec;
}

/* ==========================================================================
*        FUNCTION NAME: readFit
* FUNCTION DESCRIPTION: Read .fit file and copy in opencv Mat
*        CREATION DATE: 20160727
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
void readFit(const char* nameFile, std::ostream& stream, cv::Mat& img)
{
  // Check for invalid input
  fitsfile *fptr;
  char card[FLEN_CARD];
  int status = 0;
  int nkeys = 0;
  int ii = 0; /* MUST initialize status */
  //fits_open_file(&fptr, argv[1], READONLY, &status);
  fits_open_file(&fptr, nameFile, READONLY, &status);
  fits_get_hdrspace(fptr, &nkeys, NULL, &status);
  for (ii = 1; ii <= nkeys; ii++) {
    fits_read_record(fptr, ii, card, &status); /* read keyword */
#if SPD_STAMP_FIT_HEADER
      stamp(stream, card);
#endif
  }
  
  int naxis = 0;
  int status1 = 0;
  int imgDim = fits_get_img_dim(fptr, &naxis, &status1);
  
  if (2 != naxis)
  {
    stamp(stream, "Error! 3d image.");
  }

  int maxdim= 2;
  int bitpix= 0;
  int naxis2 = 0;
  long naxes[2] = { 0, 0 };
  int statusImg= 0;
  int imgParam = fits_get_img_param(fptr, maxdim, &bitpix, &naxis2, naxes, &statusImg);
  

  /* Read image */
  
  int statusRead = 0;
  void *nulval = NULL;
  long fpixel[2] = {1, 1};
  long nelements = naxes[0] * naxes[1];
  int anynul = 0;
  
  int datatype = TUSHORT;
  unsigned short* array = (unsigned short*)malloc(sizeof(unsigned short) * nelements);
  int readImg = fits_read_pix(fptr, datatype, fpixel, nelements, nulval, array, &anynul, &status);

  img = cv::Mat(naxes[1], naxes[0], CV_16U, array);

#if SPD_FIGURE_1
    // Create a window for display.
    namedWindow("Input .fit image", cv::WINDOW_NORMAL);
    imshow("Input .fit image", img);
#endif
#if SPD_STAMP_FIT_HEADER
  stamp(stream, "END .fit header\n");
#endif
  fits_close_file(fptr, &status);
  if (status) /* print any error messages */
  {
    fits_report_error(stderr, status);
  }
  return;
}

/* ==========================================================================
*        FUNCTION NAME: histogramStretching
* FUNCTION DESCRIPTION: Histogram Stretching
*        CREATION DATE: 20160727
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
cv::Mat histogramStretching(const cv::Mat& imgIn)
{
  //int depth = imgIn.depth();
  int type = imgIn.type();
  double outputByteDepth = 255.0;

  int color = 0;

  if (0 == type) {
    color = static_cast<int>(::pow(2, 8));
  } else if (2 == type) {
    color = static_cast<int>(::pow(2, 16));
  } else {
    printf("Error. Unsupported pixel type.\n");
  }
  cv::Mat hist = cv::Mat::zeros(1, color, CV_32F);
  cv::Mat LUT = cv::Mat::zeros(1, color, CV_32F);

  for (int row = 0; row < imgIn.rows; ++row)
  {
    const ushort* pLine = imgIn.ptr<ushort>(row);
    for (int col = 0; col < imgIn.cols; ++col) {
      ushort value = pLine[col];

      float* pLineH = hist.ptr<float>(0);
      pLineH[value] +=  1;
    }
  }

#if SPD_DEBUG
  //Print matrix value
  for (int row = 0; row < hist.rows; ++row)
  {
    const float* pLine = hist.ptr<float>(row);
    for (int col = 0; col < hist.cols; ++col) {
      float value = pLine[col];
      printf("%f ", value);      
    }
    printf("\n\n");
  }
#endif

  double maxHistValue = 0, minHistValue = 0;
  cv::Point minLocHistValue = 0, maxLocHistValue = 0;
  cv::minMaxLoc(imgIn, &minHistValue, &maxHistValue, &minLocHistValue, &maxLocHistValue, cv::noArray());

  double peakMax = 0, peakMin = 0;
  cv::Point peakMinLoc = 0, peakMaxLoc = 0;
  cv::minMaxLoc(hist, &peakMin, &peakMax, &peakMinLoc, &peakMaxLoc, cv::noArray());
  
  const double percentile[2] = { 0.432506, (1 - 0.97725) };
  double  lowThresh = peakMax * percentile[0];
  double highThresh = peakMax * percentile[1];

  int i = 0, k = 0;
  int minValue = 0, maxValue = 0;
  for (i = 0; i < peakMaxLoc.x; ++i)
  {
    k = peakMaxLoc.x - i;
    double val = static_cast<double>(hist.at<float>(0, k));
    if (val < lowThresh) {
      minValue = k;
      break;
    }
  }

  for (i = peakMaxLoc.x; i < hist.cols; ++i)
  {
    double val = static_cast<double>(hist.at<float>(0, i));
    if (val < highThresh) {
      maxValue = i;
      break;
    }
  }

  double scaleFactor = outputByteDepth/(maxValue-minValue);
  
  for (i = 0; i < hist.cols; ++i)
  {
    if (i < minValue)
    {
      LUT.at<float>(0, i) = 0;
    }
    else if (i > maxValue) {
      LUT.at<float>(0, i) = outputByteDepth;
    }
    else {
      LUT.at<float>(0, i) = (i - minValue)*scaleFactor;
    }
  }

  cv::Mat imgOut = cv::Mat::zeros(imgIn.rows, imgIn.cols, CV_8U);

  for (int row = 0; row < imgIn.rows; ++row)
  {
    const ushort* pLimgIn = imgIn.ptr<ushort>(row);
    const float* pLlut = LUT.ptr<float>(0);
    uchar* pLimgOut = imgOut.ptr<uchar>(row);

    for (int col = 0; col < imgIn.cols; ++col) {
      ushort valueIn = pLimgIn[col];
      float valueLUT = pLlut[valueIn];
      pLimgOut[col] =  static_cast<uchar>(valueLUT);
    }
  }
  
#if SPD_FIGURE_1
    namedWindow("8bits image", cv::WINDOW_NORMAL);
    imshow("8bits image", imgOut);
#endif
/* Differenza di 1 nel valore di alcuni pixel*/
#if SPD_DEBUG
  //Print matrix value
  for (int row = 0; row < imgOut.rows; ++row)
  {
    const uchar* pLine = imgOut.ptr<uchar>(row);
    for (int col = 0; col < imgOut.cols; ++col) {
      uchar value = pLine[col];
      printf("%u ", value);      
    }
    printf("\n\n");
  }
#endif

  return imgOut;
}

/* ==========================================================================
*        FUNCTION NAME: gaussianFilter
* FUNCTION DESCRIPTION: Gaussian lowpass filter
*        CREATION DATE: 20160727
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
cv::Mat gaussianFilter(const cv::Mat& imgIn, int hsize[2], double sigma)
{
  cv::Mat imgOut;
  cv::Size h = { hsize[0], hsize[1] };

  GaussianBlur(imgIn, imgOut, h, sigma, sigma, cv::BORDER_DEFAULT);
  
#if SPD_FIGURE_1
    // Create a window for display.
    namedWindow("Gaussain filter", cv::WINDOW_NORMAL);
    imshow("Gaussain filter", imgOut);
#endif

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
cv::Mat medianFilter(const cv::Mat& imgIn, int kerlen)
{
  cv::Mat imgOut;
  medianBlur(imgIn, imgOut, kerlen);

#if SPD_FIGURE_1
    namedWindow("Median filter", cv::WINDOW_NORMAL);
    imshow("Median filter", imgOut);
#endif

  return imgOut;
}

/* ==========================================================================
*        FUNCTION NAME: medianFilter
* FUNCTION DESCRIPTION: Subtraction of median filter
*        CREATION DATE: 20160727
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
cv::Mat medianFilter(const cv::Mat& imgIn, int littleKerlen, int bigKerlen)
{
  cv::Mat imgOut, imgBigKer;

  medianBlur(imgIn, imgOut, littleKerlen);
  medianBlur(imgIn, imgBigKer, bigKerlen);

  imgOut = imgOut - imgBigKer;

#if SPD_FIGURE_1
    namedWindow("Subtraction of median filter", cv::WINDOW_NORMAL);
    imshow("Subtraction of median filter", imgOut);
#endif

  return imgOut;
}

/* ==========================================================================
*        FUNCTION NAME: morphologyOpen
* FUNCTION DESCRIPTION: Morphology opening with linear structuring element
*        CREATION DATE: 20160727
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
cv::Mat morphologyOpen(const cv::Mat& imgIn, int dimLine, double teta)
{
  cv::Mat imgOut;

  cv::Mat structEl = linearKernel(dimLine, teta);

  int iter = 1;
  cv::Point anchor = cv::Point(-1, -1);
  morphologyEx(imgIn, imgOut, cv::MORPH_OPEN, structEl, anchor, iter
    , cv::BORDER_CONSTANT, cv::morphologyDefaultBorderValue());

#if SPD_FIGURE_1
    // Create a window for display.
    namedWindow("Morphology opening with rectangular kernel", cv::WINDOW_NORMAL);
    imshow("Morphology opening with rectangular kernel", imgOut);
#endif

  return imgOut;
}

/* ==========================================================================
*        FUNCTION NAME: morphologyOpen
* FUNCTION DESCRIPTION: Morphology opening with circular structuring element
*        CREATION DATE: 20160727
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
cv::Mat morphologyOpen(const cv::Mat& imgIn, int rad)
{
  cv::Mat imgOut;
  int iter = 1;
  cv::Point anchor = cv::Point(-1, -1);
  cv::Size size = cv::Size(rad, rad);

  //InputArray kernel;
  cv::Mat horizontalStructure = getStructuringElement(cv::MORPH_ELLIPSE, size, anchor);

  morphologyEx(imgIn, imgOut, cv::MORPH_OPEN, horizontalStructure, anchor, iter
    , cv::BORDER_CONSTANT, cv::morphologyDefaultBorderValue());

#if SPD_FIGURE_1
    // Create a window for display.
    namedWindow("Morphology opening with circular kernel", cv::WINDOW_NORMAL);
    imshow("Morphology opening with circular kernel", imgOut);
#endif

  return imgOut;
}

/* ==========================================================================
*        FUNCTION NAME: binarization
* FUNCTION DESCRIPTION: Image binarization using Otsu method
*        CREATION DATE: 20160727
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
cv::Mat binarization(const cv::Mat& imgIn)
{
  cv::Mat imgOut, binImg;
    
  double maxval = 255.0;
  double level = 0.0;
  
  level = threshold(imgIn, binImg, cv::THRESH_OTSU, maxval, cv::THRESH_BINARY);

  //level = level * 1.5;
  
  threshold(imgIn, imgOut, level, maxval, cv::THRESH_BINARY);
  
#if SPD_FIGURE_1
    /* Create a window for display.
    namedWindow("Binary image", WINDOW_NORMAL);
    imshow("Binary image", binImg);*/

    // Create a window for display.
    namedWindow("Binary image Otsu threshold", cv::WINDOW_NORMAL);
    imshow("Binary image Otsu threshold", imgOut);
#endif

  return imgOut;
}

/* ==========================================================================
*        FUNCTION NAME: binarization
* FUNCTION DESCRIPTION: Image binarization using user threshold
*        CREATION DATE: 20160727
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
cv::Mat binarization(const cv::Mat& imgIn, double level)
{
  cv::Mat imgOut;
  double maxval = 255.0;    
  double res = threshold(imgIn, imgOut, level, maxval, cv::THRESH_BINARY);

#if SPD_FIGURE_1
    namedWindow("Binary image user threshold", cv::WINDOW_NORMAL);
    imshow("Binary image user threshold", imgOut);
#endif

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
cv::Mat binarizationDiffTh(const cv::Mat& imgIn, int flag)
{
  cv::Mat imgOut, binImg;
  cv::Mat subBImgTL, subBImgTR, subBImgBL, subBImgBR;

  //cv::Point imgSz = { imgIn.rows, imgIn.cols };

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

#if SPD_FIGURE_1
    // Create a window for display.
    namedWindow("Binary image", cv::WINDOW_NORMAL);
    imshow("Binary image", imgOut);
#endif

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
cv::Mat convolution(const cv::Mat& imgIn, const cv::Mat& kernel, double thresh)
{
  cv::Mat imgOut, convImg;

  int ddepth = CV_32F;
  cv::Point anchor = cv::Point(-1, -1);
  double delta = 0;

  filter2D(imgIn, convImg, ddepth, kernel, anchor, delta, cv::BORDER_DEFAULT);

  double maxval = 255.0;

  cv::threshold(convImg, imgOut, thresh*maxval, maxval, cv::THRESH_BINARY);

  double alpha = 1, beta = 0;
  imgOut.convertTo(imgOut, CV_8U, alpha, beta);

#if SPD_FIGURE_1
    // Create a window for display.
    namedWindow("Convolution image", cv::WINDOW_NORMAL);
    imshow("Convolution image", imgOut);
#endif
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
void connectedComponents
(
  const cv::Mat& imgPoints
  , const cv::Mat& imgStreaks
  , const cv::Vec<int, 4>& borders
  , std::vector< cv::Vec<int, 3> >& POINTS
  , std::vector< cv::Vec<int, 3> >& STREAKS
)
{
  std::vector<std::vector<cv::Point> > contoursP;
  std::vector<std::vector<cv::Point> > contoursS;
  std::vector<cv::Vec4i> hierarchyP;
  std::vector<cv::Vec4i> hierarchyS;
  cv::Point offset = cv::Point(0, 0);
      
  findContours( imgPoints, contoursP, hierarchyP, CV_RETR_EXTERNAL
                , CV_CHAIN_APPROX_NONE , offset);
  
  std::vector< cv::Vec<int, 3> > outPOINTS;

  if (contoursP.size() > 0)
  {
    outPOINTS = connectedComponentsPoints
    (imgPoints, contoursP, borders);
  }

  findContours( imgStreaks, contoursS, hierarchyS, CV_RETR_EXTERNAL
                , CV_CHAIN_APPROX_NONE , offset);

  if (contoursS.size() > 0)
  {
    STREAKS = connectedComponentsStreaks
    (imgStreaks, contoursS, outPOINTS, contoursP, borders);
  }
      
  for (size_t j = 0; j < outPOINTS.size(); ++j)
  {
    if (-1 != outPOINTS.at(j)[2])
    {
      POINTS.push_back({ outPOINTS.at(j)[0], outPOINTS.at(j)[1], 0 });
    }
  }
  
#if SPD_FIGURE_1
    /// Draw contours
    cv::Mat drawing = cv::Mat::zeros(imgPoints.size(), CV_8UC3);
    for (int i = 0; i < contoursP.size(); i++)
    {
      cv::Scalar color = cv::Scalar(0, 255, 0);
      drawContours(drawing, contoursP, i, color, 2, 8, hierarchyP, 0, cv::Point());
    }
    for (int i = 0; i < contoursS.size(); i++)
    {
      cv::Scalar color = cv::Scalar(0, 255, 0);
      drawContours(drawing, contoursS, i, color, 2, 8, hierarchyS, 0, cv::Point());
    }
    /// Show in a window
    namedWindow("Contours", cv::WINDOW_NORMAL);
    imshow("Contours", drawing);
#endif
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
  const cv::Mat& imgIn
  , std::vector<std::vector<cv::Point > >& contours
  , const cv::Vec<int, 4>& borders
)
{
  float max_points_diameter = 0; 
  float min_points_diameter = std::max(imgIn.cols, imgIn.rows);
  float cumulativeMajorAxis = 0;
  float count = 0;

  /* Initialize vector */  
  std::vector< int >       points(contours.size());
  std::vector< cv::Point > centroid(contours.size());
  std::vector< float >     majorAxis(contours.size());
  std::vector< float >     minorAxis(contours.size());
  std::vector< int >       falsePoints(contours.size());
  
  for (size_t i = 0; i < contours.size(); ++i)
  {
    cv::Point2f center;
    float radius;
    minEnclosingCircle(contours[i], center, radius);

    cv::Point_<int> centerP ( static_cast<int>(round(center.x)) 
                            , static_cast<int>(round(center.y)) );

    if(   (centerP.x>borders[0] && centerP.x<borders[2]) 
       && (centerP.y>borders[1] && centerP.y<borders[3]))
    {
      centroid.at(i) = centerP;
      
      if(contours[i].size()>5)
      {
        cv::RotatedRect rotatedRect = fitEllipse(contours[i]);
        /*majorAxis.at(i) = static_cast<int>(rotatedRect.size.height);
        minorAxis.at(i) = static_cast<int>(rotatedRect.size.width);*/
        majorAxis.at(i) = rotatedRect.size.height;
        minorAxis.at(i) = rotatedRect.size.width;
        cumulativeMajorAxis += rotatedRect.size.height;
        count = count + 1;

        if (0 == minorAxis.at(i))
        { 
          /*contours.at(i).at(0).x = -1;
          contours.at(i).at(0).y = -1;*/
          falsePoints.at(i) = -1;
          continue;
        }
        
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
        else
        {
          /*contours.at(i).at(0).x = -1;
          contours.at(i).at(0).y = -1;*/
          falsePoints.at(i) = -1;
        }
      } //if(contours[i].size()>5)      
      else
      {
        /*contours.at(i).at(0).x = -1;
        contours.at(i).at(0).y = -1;*/
        falsePoints.at(i) = -1;
      }
    }
    else
    {
      /*contours.at(i).at(0).x = -1;
      contours.at(i).at(0).y = -1;*/
      falsePoints.at(i) = -1;
    }
  } //for (size_t i = 0; i < contours.size(); ++i)

  int init = 0;
  int n_points = std::accumulate(points.begin(), points.end(), init);
  
  if (n_points)
  {
    int threshValue=((max_points_diameter/4)+((cumulativeMajorAxis/count)/2))/2;
    for (size_t j = 0; j < contours.size(); ++j)
    {
      /* Delete little circular connect components */
      if (majorAxis.at(j)<threshValue) //ceil(max_points_diameter/2)
      {
        points.at(j) = 0;
        /*contours.at(j).at(0).x = -1;
        contours.at(j).at(0).y = -1;*/
        falsePoints.at(j) = -1;
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
      if ((1 == points.at(k)) && (-1 != falsePoints.at(k)))//(-1 != contours.at(k).at(0).x) )
      {
        outPOINTS.at(indx) = { centroid.at(k).x, centroid.at(k).y, 0};
        indx++;
      }
    }
  } //if (n_points)

  return outPOINTS;
}

/* ==========================================================================
*        FUNCTION NAME: connectedComponentsStreaks
* FUNCTION DESCRIPTION: Found centroid of linear connected components
*        CREATION DATE: 20160727
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
std::vector< cv::Vec<int, 3> > connectedComponentsStreaks
(
  const cv::Mat& imgIn
  , std::vector<std::vector<cv::Point > >& contoursS
  , std::vector< cv::Vec<int, 3> >& POINTS
  , std::vector<std::vector<cv::Point > >& contoursP
  , const cv::Vec<int, 4>& borders
)
{
  int min_streaks_minoraxis = std::max(imgIn.cols, imgIn.rows);
  int max_streaks_minoraxis=0;
  int max_streaks_majoraxis = 0;

  /* Initialize vector */
  std::vector< int >        streaks(contoursS.size());
  std::vector< cv::Point >  centroid(contoursS.size());
  std::vector< int >        majorAxis(contoursS.size());
  std::vector< int >        minorAxis(contoursS.size());
  
  for (size_t i = 0; i < contoursS.size(); ++i)
  {
    cv::Point2f center;
    float radius;
    minEnclosingCircle(contoursS[i], center, radius);

    cv::Point_<int> centerP ( static_cast<int>(round(center.x)) 
                            , static_cast<int>(round(center.y)) );

    if(   (centerP.x>borders[0] && centerP.x<borders[2]) 
       && (centerP.y>borders[1] && centerP.y<borders[3]))
    {
      centroid.at(i) = centerP;
      if(contoursS[i].size()>5)
      {      
        cv::RotatedRect rotatedRect = fitEllipse(contoursS[i]);
        majorAxis.at(i) = static_cast<int>(rotatedRect.size.height);
        minorAxis.at(i) = static_cast<int>(rotatedRect.size.width);
        
        if (0 == minorAxis.at(i))
        {
          contoursS.at(i).at(0).x = -1;
          contoursS.at(i).at(0).y = -1;
          continue;
        }
        
        /* Identify linear connect components */
        if (majorAxis.at(i) / minorAxis.at(i) > 4)//6
        {
          streaks.at(i) = 1;
          if (minorAxis.at(i) < min_streaks_minoraxis)
          {
            min_streaks_minoraxis = minorAxis.at(i);
          }
          if (minorAxis.at(i) > max_streaks_minoraxis)
          {
            max_streaks_minoraxis = minorAxis.at(i);
          }
          if (majorAxis.at(i) > max_streaks_majoraxis)
          {
            max_streaks_majoraxis = majorAxis.at(i);
          }          
        } //if (majorAxis.at(i) / minorAxis.at(i) > 6)
        else
        {
          contoursS.at(i).at(0).x = -1;
          contoursS.at(i).at(0).y = -1;
        }
      } //if(contoursS[i].size()>5)
      else
      {
        contoursS.at(i).at(0).x = -1;
        contoursS.at(i).at(0).y = -1;
      }
    }
    else
    {
      contoursS.at(i).at(0).x = -1;
      contoursS.at(i).at(0).y = -1;
    }
  } //for (size_t i = 0; i < contoursS.size(); ++i)

  int init = 0;
  int n_streaks = std::accumulate(streaks.begin(), streaks.end(), init);
  
  if (n_streaks)
  {
    if (n_streaks < 2)
    {
      min_streaks_minoraxis=2;
    }
    for (size_t j = 0; j < contoursS.size(); ++j)
    {
      /* Delete short linear connect components */
      if (minorAxis.at(j) < ceil(min_streaks_minoraxis))
      {
        streaks.at(j) = 0;
        contoursS.at(j).at(0).x = -1;
        contoursS.at(j).at(0).y = -1;
      }
    }
  } //if (n_streaks)
  
  n_streaks = std::accumulate(streaks.begin(), streaks.end(), init);
  
  if (n_streaks)
  {
    for (size_t j = 0; j < contoursS.size(); ++j)
    {
      /* Delete short linear connect components */
      if (majorAxis.at(j)<ceil(max_streaks_majoraxis*0.75))// /2
      {
        streaks.at(j) = 0;
        contoursS.at(j).at(0).x = -1;
        contoursS.at(j).at(0).y = -1;
      }
    }
  } //if (n_streaks)

  n_streaks = std::accumulate(streaks.begin(), streaks.end(), init);
  std::vector< cv::Vec<int, 3> >  outSTREAKS(n_streaks);

  int indx = 0;
  if (n_streaks)
  {    
    for (size_t k = 0; k < contoursS.size(); ++k)
    {
      if ((1 == streaks.at(k)) && (-1 != contoursS.at(k).at(0).x) )
      {
        outSTREAKS.at(indx) = { centroid.at(k).x, centroid.at(k).y, 0};
        indx++;
      }
    }
  } //if (n_streaks)

  /* Delete points on streak */

  size_t nP = POINTS.size();
  if (1 == nP) {
    if (-1 == POINTS.at(0)[0]) { nP = 0; }
  }

  if((0!=nP) && (0!=indx))
  {
    for (size_t j = 0; j < outSTREAKS.size(); ++j)
    {
      for (size_t i = 0; i < POINTS.size(); ++i)
      {
        bool measureDist = false;
        cv::Point2f centerS = {static_cast<float>(outSTREAKS.at(j)[0])
          , static_cast<float>(outSTREAKS.at(j)[1])};//centro striscia

        double insideP = cv::pointPolygonTest(contoursP, centerS, measureDist);
        
        if (insideP>0)
        {
          outSTREAKS.at(j)[2] = -1;
        }
        else
        {
          cv::Point2f centerP = { static_cast<float>(POINTS.at(i)[0])
            , static_cast<float>(POINTS.at(i)[1]) };//centro striscia
          double insideS = cv::pointPolygonTest(contoursS, centerP, measureDist);
          
          if (insideS>0)
          {
            POINTS.at(i)[2] = -1;
          }
        }
      }
    }
  }

  std::vector< cv::Vec<int, 3> >  STREAKS;
  int ind = 0;
  for (size_t j = 0; j < outSTREAKS.size(); ++j)
  {
    if (-1 != outSTREAKS.at(j)[2])
    {
      STREAKS.push_back({ outSTREAKS.at(j)[0], outSTREAKS.at(j)[1], 0 });
      ind++;
    }
  }

  return STREAKS;
}

/* ==========================================================================
*        FUNCTION NAME: hough
* FUNCTION DESCRIPTION: Hough transform
*        CREATION DATE: 20160911
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
std::vector<std::pair<float, int>> hough(const cv::Mat& imgIn)
{
  double rho = 1; //0.5;
  double theta = 2*CV_PI / 180; //CV_PI / (2*180); r:pi=g:180
  int threshold = 60;
  
  std::vector<cv::Vec2f> houghVal;
  HoughLines(imgIn, houghVal, rho, theta, threshold);
  
  // Loop for find lines with high thresoldh
  size_t maxNumAngles = 10;
  bool exitL = true;
  size_t countC = 0;

  while (exitL)
  {
    countC = countC + 1;
    if (houghVal.size() > maxNumAngles)
    {
      threshold = threshold * 1.5;
      HoughLines(imgIn, houghVal, rho, theta, threshold);
    }    
    else if ((houghVal.size()>=5) && (houghVal.size() <= 10))
    {
      exitL = false;
    }
    else
    {
      threshold = threshold / 1.05;
      HoughLines(imgIn, houghVal, rho, theta, threshold);
    }
    if (10 == countC) {
      exitL = false;
    }
  }

  
  // Select the inclination angles
  std::vector<float> angle;
  for (size_t i = 0; i < houghVal.size(); ++i)
  {
    angle.push_back(houghVal.at(i)[1]);
  }

  int count = 0;
  std::vector<std::pair<float, int>> countAngle;
  for (size_t i = 0; i < houghVal.size(); ++i)
  {
    int a = std::count(angle.begin(), angle.end(), angle.at(i));
    countAngle.push_back(std::make_pair(angle.at(i), a));
    count = count + a;
    if (houghVal.size() == count) break;
  }

#if SPD_FIGURE_1
    cv::Mat color_dst;
    cvtColor( imgIn, color_dst, CV_GRAY2BGR );
    double minLineLength = 20;
    double maxLineGap = 1;
    std::vector<cv::Vec4i> lines;
    HoughLinesP(imgIn, lines, rho, theta, threshold, minLineLength, maxLineGap);

    for (size_t i = 0; i < lines.size(); i++) {
      line(color_dst, cv::Point(lines[i][0], lines[i][1]),
        cv::Point(lines[i][2], lines[i][3]), cv::Scalar(0, 0, 255), 3, 8);
    }

    // Create a window for display.
    namedWindow("Hough transform", cv::WINDOW_NORMAL);
    imshow("Hough transform", color_dst);
    cv::waitKey(0);
#endif
  
  return countAngle;
}

/* ==========================================================================
*        FUNCTION NAME: timeElapsed
* FUNCTION DESCRIPTION: Compute elapsed time
*        CREATION DATE: 20160911
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
void timeElapsed(std::ostream& stream, clock_t start, const char* strName)
{
  clock_t stop = clock();
  double totalTime = (stop - start) / static_cast<double>(CLOCKS_PER_SEC);

  std::string str = " time: ";
  std::string s_t = strName + str + std::to_string(totalTime);
  stamp(stream, s_t.c_str());
}

/* ==========================================================================
*        FUNCTION NAME: linearKernel
* FUNCTION DESCRIPTION: Create linear structural element
*        CREATION DATE: 20160911
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
cv::Mat linearKernel(int dimLine, double teta)
{
#if 0
  int yDim = static_cast<int>(::ceil(dimLine * ::abs(::tan(teta))));
  if (0 == yDim) { yDim = 1; }
#else
  int yDim = static_cast<int>(::ceil(dimLine * ::abs(::sin((CV_PI/2)-teta))));
  int xDim = static_cast<int>(::ceil(dimLine * ::abs(::cos((CV_PI/2)-teta))));
#endif
  //cv::Mat kernel = cv::Mat::zeros(yDim, dimLine, CV_8U);
  cv::Mat kernel = cv::Mat::zeros(yDim, xDim, CV_8U);

  cv::Point pt1 = { 0, 0 };
  cv::Point pt2 = { 0, 0 };
  if (teta > 0) {
    pt1 = { 0, yDim };
    //pt2 = { dimLine, 0 };
    pt2 = { xDim, 0 };
  }
  else {
    pt1 = { 0, 0 };
    //pt2 = { dimLine, yDim };
    pt2 = { xDim, yDim };
  }

  const cv::Scalar color = cv::Scalar(255, 255, 255);
  int thickness = 1;
  int lineType = 8;
  int shift = 0;
  line(kernel, pt1, pt2, color, thickness, lineType, shift);

  return kernel;
}

/* ==========================================================================
*        FUNCTION NAME: stamp
* FUNCTION DESCRIPTION: Print on file and console the input string
*        CREATION DATE: 20160911
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
void stamp(std::ostream& stream, const char* strName)
{
#if SPD_STAMP
  stream << strName << std::endl;
#endif
}

/* ==========================================================================
*        FUNCTION NAME: writeResult
* FUNCTION DESCRIPTION: Print on file and console the result points and 
*                       streaks centroid
*        CREATION DATE: 20160911
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
void writeResult
(
  std::ostream& stream
  , std::vector< cv::Vec<int, 3> >& POINTS
  , std::vector< cv::Vec<int, 3> >& STREAKS
)
{
  std::string s_nP = "Detected points: " + std::to_string(POINTS.size());
  stamp(stream, s_nP.c_str());
  for (size_t i = 0; i < POINTS.size(); ++i)
  {
    std::string cP = "Centroid points: (" + std::to_string(POINTS.at(i)[0]) + "," + std::to_string(POINTS.at(i)[1]) + ")";
    stamp(stream, cP.c_str());
  }

  std::string s_nS = "Detected streaks: " + std::to_string(STREAKS.size());
  stamp(stream, s_nS.c_str());
  for (size_t i = 0; i < STREAKS.size(); ++i)
  {
    std::string cS = "Centroid streaks: (" + std::to_string(STREAKS.at(i)[0]) + "," + std::to_string(STREAKS.at(i)[1]) + ")";
    stamp(stream, cS.c_str());
  }
}
