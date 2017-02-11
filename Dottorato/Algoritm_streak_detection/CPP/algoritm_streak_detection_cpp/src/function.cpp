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

#if _WIN32
  const char* slash = "\\";
#else
  const char* slash = "//";
#endif

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
  cv::Point minLocHistValue = {0, 0};
  cv::Point maxLocHistValue = {0, 0};
  cv::minMaxLoc(imgIn, &minHistValue, &maxHistValue, &minLocHistValue, &maxLocHistValue, cv::noArray());

  double peakMax = 0, peakMin = 0;
  cv::Point peakMinLoc = {0, 0};
  cv::Point peakMaxLoc = {0, 0};
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
  const cv::Point imgSz = {imgPoints.rows, imgPoints.cols};
  float max_img_sz = std::max(imgPoints.cols, imgPoints.rows);

  std::vector<std::vector<cv::Point> > contoursP;
  std::vector<std::vector<cv::Point> > contoursS;
  std::vector<cv::Vec4i> hierarchyP;
  std::vector<cv::Vec4i> hierarchyS;
  cv::Point offset = cv::Point(0, 0);
  
  /* Find points contours */
  findContours( imgPoints, contoursP, hierarchyP, CV_RETR_EXTERNAL
                , CV_CHAIN_APPROX_NONE , offset);
  
  std::vector< cv::Vec<int, 3> > outPOINTS;
  std::vector<std::vector<cv::Point > > contoursResP;

  if (contoursP.size() > 0)
  {
    outPOINTS = connectedComponentsPoints
    (max_img_sz, contoursP, borders, contoursResP);
    contoursP.clear();
  }

  /* Find streaks contours */
  findContours( imgStreaks, contoursS, hierarchyS, CV_RETR_EXTERNAL
                , CV_CHAIN_APPROX_NONE , offset);

  std::vector< cv::Vec<int, 3> > outSTREAKS;
  std::vector<std::vector<cv::Point > > contoursResS;

  if (contoursS.size() > 0)
  {
    outSTREAKS = connectedComponentsStreaks
    (max_img_sz, contoursS, borders, contoursResS);
    contoursS.clear();
  }
  
  /* Delete overlapping objects */
  deleteOverlapping
  (imgSz, outPOINTS, outSTREAKS, contoursResP, contoursResS, POINTS, STREAKS);
  
#if SPD_FIGURE_1
    /// Draw contours
    cv::Mat drawing = cv::Mat::zeros(imgPoints.size(), CV_8UC3);    
    int cIdx = -1;
    cv::Scalar color = cv::Scalar(0, 255, 0);
    cv::Scalar colorS = cv::Scalar(0, 0, 255);
    int lineType = 8;
    cv::InputArray hierarchy = cv::noArray();
    int maxLevel = 0;
    drawContours(drawing, contoursResP, cIdx, color, 1, 8, hierarchy, 0, offset);
    drawContours(drawing, contoursResS, cIdx, colorS, 1, 8, hierarchy, 0, offset);
    
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
  const float max_img_sz
  , const std::vector<std::vector<cv::Point > >& contours
  , const cv::Vec<int, 4>& borders
  , std::vector<std::vector<cv::Point > >& outContoursRes
)
{
  float max_points_diameter = 0; 
  float min_points_diameter = max_img_sz;
  float cumulativeMajorAxis = 0;
  float count = 0;

  /* Initialize vector */  
  std::vector< cv::Point > centroidV;
  std::vector< float >     majorAxisV;
  std::vector< float >     minorAxisV;
  std::vector<std::vector<cv::Point > > contoursResFV;

  cv::Point2f center = {0.0f, 0.0f};
  float radius = 0.0f;
  float majorAxis = 0.0f;
  float minorAxis = 0.0f;

  for (size_t i = 0; i < contours.size(); ++i)
  {
    center = {0.0f, 0.0f};
    majorAxis = 0.0f;
    minorAxis = 0.0f;

    minEnclosingCircle(contours[i], center, radius);

    cv::Point_<int> centerP ( static_cast<int>(round(center.x)) 
                            , static_cast<int>(round(center.y)) );

    if(   (centerP.x>borders[0] && centerP.x<borders[2]) 
       && (centerP.y>borders[1] && centerP.y<borders[3]))
    {
      if(contours[i].size()>5)
      {
        cv::RotatedRect rotatedRect = fitEllipse(contours[i]);
        majorAxis = rotatedRect.size.height;
        minorAxis = rotatedRect.size.width;

        if (0 == minorAxis) {           
          continue;
        }
        
        /* Identify circular connect components */
        if (majorAxis / minorAxis < 1.6)
        {
          if (majorAxis > max_points_diameter)
          {
            max_points_diameter = majorAxis;
          }
          if (minorAxis < min_points_diameter)
          {
            min_points_diameter = minorAxis;
          }

          centroidV.push_back(centerP);
          majorAxisV.push_back(majorAxis);
          minorAxisV.push_back(minorAxis);
          contoursResFV.push_back(contours[i]);

          cumulativeMajorAxis += rotatedRect.size.height;
          count = count + 1;

        } //if (majorAxis / minorAxis < 1.6)
        else {
          continue;
        }
      } //if(contours[i].size()>5)      
      else {
        continue;
      }
    } //if inside borders
    else {
      continue;
    }
  } //for (size_t i = 0; i < contours.size(); ++i)


  /**/
  std::vector< char > points(contoursResFV.size());
  int init = 0;
  int n_points = 0;

  if (contoursResFV.size())
  {
    int threshValue=((max_points_diameter/4)+((cumulativeMajorAxis/count)/2))/2;
    for (size_t j = 0; j < contoursResFV.size(); ++j)
    {
      /* Delete little circular connect components */
      if (majorAxisV.at(j)<threshValue) //ceil(max_points_diameter/2)
      {
        points.at(j) = 0;
      }
      else
      {
        points.at(j) = 1;
      }
    }
  }

  n_points = std::accumulate(points.begin(), points.end(), init);
  
  std::vector< cv::Vec<int, 3> >  POINTS(n_points);
      
  if (n_points)
  {
    int indx = 0;
    for (size_t k = 0; k < contoursResFV.size(); ++k)
    {
      if (1 == points.at(k))
      {
        POINTS.at(indx) = { centroidV.at(k).x, centroidV.at(k).y, 0};
        outContoursRes.push_back(contoursResFV[k]);
        indx++;
      }
    }
  } //if (n_points)

  return POINTS;
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
  const float max_img_sz
  , const std::vector<std::vector<cv::Point > >& contours
  , const cv::Vec<int, 4>& borders
  , std::vector<std::vector<cv::Point > >& outContoursRes
)
{
  int min_streaks_minoraxis = max_img_sz;
  int max_streaks_minoraxis=0;
  int max_streaks_majoraxis = 0;

  /* Initialize vector */
  std::vector< cv::Point > centroidV;
  std::vector< float >     majorAxisV;
  std::vector< float >     minorAxisV;
  std::vector<std::vector<cv::Point > > contResStreakFV;
    
  cv::Point2f center = {0.0f, 0.0f};
  float radius = 0.0f;
  float majorAxis = 0.0f;
  float minorAxis = 0.0f;

  for (size_t i = 0; i < contours.size(); ++i)
  {
    center = {0.0f, 0.0f};
    majorAxis = 0.0f;
    minorAxis = 0.0f;
    
    minEnclosingCircle(contours[i], center, radius);

    cv::Point_<int> centerP ( static_cast<int>(round(center.x)) 
                            , static_cast<int>(round(center.y)) );

    if(   (centerP.x>borders[0] && centerP.x<borders[2]) 
       && (centerP.y>borders[1] && centerP.y<borders[3]))
    {
      if(contours[i].size()>5)
      {
        cv::RotatedRect rotatedRect = fitEllipse(contours[i]);
        majorAxis = rotatedRect.size.height;
        minorAxis = rotatedRect.size.width;
        
        if (0 == minorAxis)
        {
          continue;
        }
        
        /* Identify linear connect components */
        if (majorAxis / minorAxis > 4)//6
        {
          if (minorAxis < min_streaks_minoraxis)
          {
            min_streaks_minoraxis = minorAxis;
          }
          if (minorAxis > max_streaks_minoraxis)
          {
            max_streaks_minoraxis = minorAxis;
          }
          if (majorAxis > max_streaks_majoraxis)
          {
            max_streaks_majoraxis = majorAxis;
          }

          centroidV.push_back(centerP);
          majorAxisV.push_back(majorAxis);
          minorAxisV.push_back(minorAxis);
          contResStreakFV.push_back(contours[i]);

        } //if (majorAxis / minorAxis > 6)
        else {
          continue;
        }
      } //if(contoursS[i].size()>5)
      else {
        continue;
      }
    } //if inside borders
    else {
      continue;
    }
  } //for (size_t i = 0; i < contoursS.size(); ++i)

  std::vector< char > streaks(contResStreakFV.size());
  int init = 0;
  int n_streaks = 0;
  
  if (contResStreakFV.size())
  {
    if (contResStreakFV.size() < 2)
    {
      min_streaks_minoraxis=2;
    }
    for (size_t j = 0; j < contResStreakFV.size(); ++j)
    {
      /* Delete short linear connect components */
      if (minorAxisV.at(j) < ceil(min_streaks_minoraxis))
      {
        streaks.at(j) = 0;
      }
      else
      {
        streaks.at(j) = 1;
      }
    }
  }
  
  n_streaks = std::accumulate(streaks.begin(), streaks.end(), init);
  
  if (n_streaks)
  {
    for (size_t j = 0; j < contResStreakFV.size(); ++j)
    {
      /* Delete short linear connect components */
      if (majorAxisV.at(j)<ceil(max_streaks_majoraxis*0.75))// /2
      {
        streaks.at(j) = 0;
      }
    }
  }

  n_streaks = std::accumulate(streaks.begin(), streaks.end(), init);

  std::vector< cv::Vec<int, 3> > STREAKS(n_streaks);

  int indx = 0;
  if (n_streaks)
  {    
    for (size_t k = 0; k < contResStreakFV.size(); ++k)
    {
      if (1 == streaks.at(k))
      {
        STREAKS.at(indx) = { centroidV.at(k).x, centroidV.at(k).y, 0};
        outContoursRes.push_back(contResStreakFV[k]);
        indx++;
      }
    }
  }

  return STREAKS;
}

/* ==========================================================================
*        FUNCTION NAME: deleteOverlapping
* FUNCTION DESCRIPTION: Delete overlapping objects
*        CREATION DATE: 20160727
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
void deleteOverlapping
(
  const cv::Point imgSz
  , std::vector< cv::Vec<int, 3> >& inPOINTS
  , std::vector< cv::Vec<int, 3> >& inSTREAKS
  , const std::vector<std::vector<cv::Point > >& contoursP
  , const std::vector<std::vector<cv::Point > >& contoursS
  , std::vector< cv::Vec<int, 3> >& outPOINTS
  , std::vector< cv::Vec<int, 3> >& outSTREAKS
)
{
  /* Delete streaks on points */
  cv::Mat imgP = cv::Mat::zeros(imgSz.x, imgSz.y, CV_8U);
  int cIdx = -1;
  const cv::Scalar color = 255;
  int lineType = 8;
  cv::InputArray hierarchy = cv::noArray();
  int maxLevel = 0;
  cv::Point offset = {0,0};
  drawContours(imgP, contoursP, cIdx, color, CV_FILLED, lineType, hierarchy, maxLevel, offset);
    
  int n_streaks = 0;

  for (size_t i = 0; i < inSTREAKS.size(); ++i)
  {
    const uchar* pRimgP = imgP.ptr<uchar>(inSTREAKS.at(i)[1]);
    uchar value = pRimgP[inSTREAKS.at(i)[0]];

    if (255 != value)
    {
      inSTREAKS.at(i)[2] = 1;
    }
    n_streaks = n_streaks + inSTREAKS.at(i)[2];
  }
  imgP.release();

  std::vector<std::vector<cv::Point > > contoursResS;

  if (n_streaks)
  {    
    for (size_t k = 0; k < inSTREAKS.size(); ++k)
    {
      if (1 == inSTREAKS.at(k)[2])
      {
        outSTREAKS.push_back({ inSTREAKS.at(k)[0], inSTREAKS.at(k)[1], 0});
        contoursResS.push_back(contoursS[k]);
      }
    }
  }

  /* Delete points on streak */
  cv::Mat imgS = cv::Mat::zeros(imgSz.x, imgSz.y, CV_8U);
  drawContours(imgS, contoursResS, cIdx, color, CV_FILLED, lineType, hierarchy, maxLevel, offset);
  
  int n_points = 0;

  for (size_t i = 0; i < inPOINTS.size(); ++i)
  {
    const uchar* pRimgS = imgS.ptr<uchar>(inPOINTS.at(i)[1]);
    uchar value = pRimgS[inPOINTS.at(i)[0]];

    if (255 != value)
    {
      inPOINTS.at(i)[2] = 1;
    }
    n_points = n_points + inPOINTS.at(i)[2];
  }

  imgS.release();

  std::vector<std::vector<cv::Point > > contoursResP;

  if (n_points)
  {    
    for (size_t k = 0; k < inPOINTS.size(); ++k)
    {
      if (1 == inPOINTS.at(k)[2])
      {
        outPOINTS.push_back({ inPOINTS.at(k)[0], inPOINTS.at(k)[1], 0});
        contoursResP.push_back(contoursP[k]);
      }
    }
  }    
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

  if (0 == yDim) {
    yDim = 1;
  }
  if (0 == xDim) {
    xDim = 1;
  }

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
