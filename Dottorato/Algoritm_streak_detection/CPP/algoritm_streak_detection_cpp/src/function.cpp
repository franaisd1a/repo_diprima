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
#include "function_os.h"
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
  
  char *nameL[32][256];
  pch = strtok(name,".");
  count = 0;
  while (pch != NULL)
  {    
    //printf ("%s\n",pch);
    *nameL[count] = pch;
    pch = strtok (NULL, ".");
    count++;
  }
  char* ext = *nameL[count-1];
  //char* fileName = *path[count-2];
  char s_fileName[256];
  strcpy (s_fileName, *nameL[0]);
  for (size_t i = 1; i < count - 1; ++i)
  {
    strcat(s_fileName, ".");
    strcat(s_fileName, *nameL[i]);
  }

  vec.push_back(s_fileName);
  vec.push_back(ext);
  vec.push_back(s_pathFileName);
  vec.push_back(s_pathResFile);

  return vec;
}

std::string fileCygwin(const std::string strN)
{
  char cwStr[1024];

  const char slashS = 92;
  const char slashD = 47;
  const char duePunti = 58;

  size_t count = 0;
  for (size_t i = 0; i < strN.size(); ++i)
  {
    if (0 == i)
    {
      cwStr[count] = tolower(strN[i]);
      count++;
    }
    else if (slashS == strN[i])
    {
      cwStr[count] = slashD;
      count++;
    }
    else if (duePunti == strN[i])
    {
      
    }
    else
    {
      cwStr[count] = strN[i];
      count++;
    }
  }

  cwStr[count] = 0;

  char cwStrF[1024];
  strcpy ( cwStrF, "/cygdrive/" );
  strcat ( cwStrF, cwStr );

  return cwStrF;
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
  
  void *nulval = NULL;
  long fpixel[2] = {1, 1};
  long nelements = naxes[0] * naxes[1];
  int anynul = 0;
  
  int datatype = TUSHORT;
  unsigned short* array = (unsigned short*)malloc(sizeof(unsigned short) * nelements);
  int readImg = fits_read_pix(fptr, datatype, fpixel, nelements, nulval, array, &anynul, &status);

#if 0
  FILE * outBinFile;
  outBinFile = fopen ("img.bin", "wb");
  size_t nCnt = fwrite ( array, sizeof(unsigned short), nelements, outBinFile );
  fclose (outBinFile);
#endif

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
*        FUNCTION NAME: histogram
* FUNCTION DESCRIPTION: Histogram
*        CREATION DATE: 20160727
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
cv::Mat histogram(const cv::Mat& imgIn, double& outputByteDepth, int& minValue, int& maxValue)
{
  //int depth = imgIn.depth();
  int type = imgIn.type();
  outputByteDepth = 255.0;

  int color = 0;

  if (0 == type) {
    color = static_cast<int>(::pow(2, 8));
  } else if (2 == type) {
    color = static_cast<int>(::pow(2, 16));
  } else {
    printf("Error. Unsupported pixel type.\n");
  }
  cv::Mat hist = cv::Mat::zeros(1, color, CV_32F);

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
  minValue = 0;
  maxValue = 0;

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

  return hist;
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
  cv::Mat hist = cv::Mat::zeros(1, color, CV_32S);
  cv::Mat LUT = cv::Mat::zeros(1, color, CV_32F);

  int* pLineH = hist.ptr<int>(0);
  for (int row = 0; row < imgIn.rows; ++row)
  {
    const ushort* pLine = imgIn.ptr<ushort>(row);
    for (int col = 0; col < imgIn.cols; ++col) {
      ushort value = pLine[col];
      
      pLineH[value] +=  1;
    }
  }

#if SPD_DEBUG
  //Print matrix value
  const int* pLine = hist.ptr<int>(0);
  for (int col = 0; col < hist.cols; ++col) {
    int value = pLine[col];
    printf("BIN=%d value=%d\n", col, value);
  }
  printf("\n\n");
#endif

#if 0
  double maxHistValue = 0, minHistValue = 0;
  cv::Point minLocHistValue = {0, 0};
  cv::Point maxLocHistValue = {0, 0};
  cv::minMaxLoc(imgIn, &minHistValue, &maxHistValue, &minLocHistValue, &maxLocHistValue, cv::noArray());
#endif

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
    double val = static_cast<double>(hist.at<int>(0, k));
    if (val < lowThresh) {
      minValue = k;
      break;
    }
  }

  for (i = peakMaxLoc.x; i < hist.cols; ++i)
  {
    double val = static_cast<double>(hist.at<int>(0, i));
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
    cv::waitKey(0);
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
*        FUNCTION NAME: subtraction
* FUNCTION DESCRIPTION: Image subtraction
*        CREATION DATE: 20160727
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
cv::Mat subtraction(const cv::Mat& imgA, const cv::Mat& imgB)
{
  cv::Mat imgOut = cv::Mat::zeros(imgA.rows, imgA.cols, CV_8U);

  for (int row = 0; row < imgA.rows; ++row)
  {
    const uchar* pLimgA = imgA.ptr<uchar>(row);
    const uchar* pLimgB = imgB.ptr<uchar>(row);
    uchar* pLimgOut = imgOut.ptr<uchar>(row);
    
    for (int col = 0; col < imgA.cols; ++col) {      
      pLimgOut[col] =  pLimgA[col]>pLimgB[col] ? pLimgA[col]-pLimgB[col] : 0;
    }
    
  }
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
  cv::waitKey(0);
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
  cv::waitKey(0);
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

cv::Mat morphologyOpen2(const cv::Mat& imgIn, int dimLine, double teta)
{
  cv::Mat imgOut;

#if 0
  int bord = 2*dimLine+1;
  const cv::Scalar value = cv::Scalar(0);
  copyMakeBorder(imgIn, imgOut, bord, bord, bord, bord, cv::BORDER_CONSTANT, value);
  namedWindow("Mor00", cv::WINDOW_NORMAL);
  imshow("Mor00", imgOut);
  cv::Mat structEl = cv::Mat::ones(5, 7, CV_8U);
#endif

  cv::Mat structEl = linearKernel(dimLine, teta);
  
  int iter = 1;
  cv::Point anchor = cv::Point(-1, -1);
  morphologyEx(imgIn, imgOut, cv::MORPH_OPEN, structEl, anchor, iter
    , cv::BORDER_CONSTANT, cv::Scalar(0.0));

#if 0
  cv::Point ptTL = { bord, bord };
  cv::Point ptBR = { imgOut.cols - bord, imgOut.rows - bord };
  cv::Rect region_of_interest = cv::Rect(ptTL, ptBR);
  cv::Mat imgPart = imgOut(region_of_interest);
  namedWindow("Mor", cv::WINDOW_NORMAL);
    imshow("Mor", imgPart);
#endif

#if SPD_FIGURE_1
    // Create a window for display.
    namedWindow("Morphology opening", cv::WINDOW_NORMAL);
    imshow("Morphology opening", imgOut);
    
    cv::waitKey(0);
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
  cv::waitKey(0);
#endif

  return imgOut;
}

/* ==========================================================================
*        FUNCTION NAME: backgroundEstimation
* FUNCTION DESCRIPTION: Background Estimation
*        CREATION DATE: 20160727
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
cv::Mat backgroundEstimation(const cv::Mat& imgInOr, const cv::Point backCnt, cv::Mat& meanBg, cv::Mat& stdBg)
{
  cv::Mat imgIn = imgInOr.clone();
  
  size_t backSzR = imgIn.rows;
  if (0 != backCnt.y)
  {
    backSzR = static_cast<size_t>(::round(imgIn.rows / backCnt.y));
  }
  size_t backSzC = imgIn.cols;
  if (0 != backCnt.x)
  {
    backSzC = static_cast<size_t>(::round(imgIn.cols / backCnt.x));
  }
  
  std::vector<int> vBackSrow;
  std::vector<int> vBackScol;

  for (int o = 0; o < backCnt.y; ++o)
  {
    vBackSrow.push_back(backSzR*o);
  }
  vBackSrow.push_back(imgIn.rows);

  for (int o = 0; o < backCnt.x; ++o)
  {
    vBackScol.push_back(backSzC*o);
  }
  vBackScol.push_back(imgIn.cols);
    
  cv::Mat outImg = cv::Mat::zeros(imgIn.rows, imgIn.cols, imgIn.type());
  
  for (int i = 0; i < backCnt.y; ++i)
  {
    for (int j = 0; j < backCnt.x; ++j)
    {
      const cv::Point ptTL = {vBackScol.at(j), vBackSrow.at(i)};
      const cv::Point ptBR = {vBackScol.at(j+1), vBackSrow.at(i+1)};

      cv::Rect region_of_interest = cv::Rect(ptTL, ptBR);
      cv::Mat imgPart = imgIn(region_of_interest);
      cv::Mat imgPartTh = cv::Mat::zeros(imgPart.rows, imgPart.cols, imgPart.type());

      float oldStd=0;
      float diffPercStd = 1;

#if 1 //NO
      meanBg.at<double>(i,j) = *(cv::mean(imgPart, cv::noArray()).val);

      while (diffPercStd>0.2f)
      {
        cv::Scalar meanBGmod = 0;
        cv::Scalar stdBGs = 0;
        meanStdDev(imgPart, meanBGmod, stdBGs, cv::noArray());
        stdBg.at<double>(i,j) = *(stdBGs.val);

        double threshH = meanBg.at<double>(i,j)+2.5*stdBg.at<double>(i,j);//3

        double maxval = 1.0;
        double asdf = cv::threshold(imgPart, imgPartTh, threshH, maxval, cv::THRESH_TOZERO_INV);



        diffPercStd = ::abs((stdBg.at<double>(i,j)-oldStd)/stdBg.at<double>(i,j));
        oldStd=stdBg.at<double>(i,j);        
      }
#else
      cv::Scalar meanBGmod1 = 0;
      cv::Scalar stdBGs1 = 0;
      meanStdDev(imgPart, meanBGmod1, stdBGs1, cv::noArray());
      stdBg.at<double>(i, j) = *(stdBGs1.val);
      meanBg.at<double>(i,j) = *(meanBGmod1.val);

      while (diffPercStd>0.2f)
      {
        double threshH = meanBg.at<double>(i,j)+2.5*stdBg.at<double>(i,j);//3

        double maxval = 1.0;
        double asdf = cv::threshold(imgPart, imgPartTh, threshH, maxval, cv::THRESH_TOZERO_INV);

        cv::Scalar meanBGmod = 0;
        cv::Scalar stdBGs = 0;
        meanStdDev(imgPartTh, meanBGmod, stdBGs, cv::noArray());
        stdBg.at<double>(i,j) = *(stdBGs.val);


        diffPercStd = ::abs((stdBg.at<double>(i,j)-oldStd)/stdBg.at<double>(i,j));
        oldStd=stdBg.at<double>(i,j);        
      }
#endif
      imgPartTh.copyTo(outImg(region_of_interest));
    }
  }

#if SPD_FIGURE_1
    // Create a window for display.
    namedWindow("Background estimationl", cv::WINDOW_NORMAL);
    imshow("Background estimationl", outImg);
    cv::waitKey(0);
#endif

  return outImg;
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
*        FUNCTION NAME: binarizationZone
* FUNCTION DESCRIPTION: Image binarization using user threshold
*        CREATION DATE: 20160727
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
cv::Mat binarizationZone(const cv::Mat& imgIn, const cv::Point zoneCnt, const cv::Mat& level)
{
  size_t zoneSzR = static_cast<size_t>(::round(imgIn.rows / zoneCnt.y));
  size_t zoneSzC = static_cast<size_t>(::round(imgIn.cols / zoneCnt.x));
  
  std::vector<int> vBackSrow;
  std::vector<int> vBackScol;

  for (int o = 0; o < zoneCnt.y; ++o)
  {
    vBackSrow.push_back(zoneSzR*o);
  }
  vBackSrow.push_back(imgIn.rows);

  for (int o = 0; o < zoneCnt.x; ++o)
  {
    vBackScol.push_back(zoneSzC*o);
  }
  vBackScol.push_back(imgIn.cols);
    
  cv::Mat outImg = cv::Mat::zeros(imgIn.rows, imgIn.cols, imgIn.type());
  
  for (int i = 0; i < zoneCnt.y; ++i)
  {
    for (int j = 0; j < zoneCnt.x; ++j)
    {
      const cv::Point ptTL = { vBackScol.at(j), vBackSrow.at(i) };
      const cv::Point ptBR = { vBackScol.at(j + 1), vBackSrow.at(i + 1) };

      cv::Rect region_of_interest = cv::Rect(ptTL, ptBR);
      cv::Mat imgPart = imgIn(region_of_interest);
      cv::Mat imgPartTh = cv::Mat::zeros(imgPart.rows, imgPart.cols, imgPart.type());

      double maxval = 255.0;
      double asdf = cv::threshold(imgPart, imgPartTh, level.at<double>(i,j), maxval, cv::THRESH_BINARY);
      imgPart.copyTo(outImg(region_of_interest));
    }
  }
  
#if SPD_FIGURE_1
    namedWindow("Binary image user thresholdZones", cv::WINDOW_NORMAL);
    imshow("Binary image user thresholdZones", outImg);
    cv::waitKey(0);
#endif

  return outImg;
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
  cv::waitKey(0);
#endif
  return imgOut;
}

/* ==========================================================================
*        FUNCTION NAME: distTransform
* FUNCTION DESCRIPTION: Distance transformation
*        CREATION DATE: 20160727
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
cv::Mat distTransform(const cv::Mat& imgIn)
{
  // Perform the distance transform algorithm
  cv::Mat dist;
  distanceTransform(imgIn, dist, CV_DIST_L2, 3);

  // Normalize the distance image for range = {0.0, 1.0}
  // so we can visualize and threshold it
  normalize(dist, dist, 0, 1., cv::NORM_MINMAX);

  // Threshold to obtain the peaks
  // This will be the markers for the foreground objects
  threshold(dist, dist, .03, 1., CV_THRESH_BINARY);

  // Dilate a bit the dist image
  cv::Mat kernel = cv::Mat::ones(3, 3, CV_8UC1);
  dilate(dist, dist, kernel, cv::Point(-1, -1), 2);

#if SPD_FIGURE_1
  namedWindow("dist", cv::WINDOW_NORMAL);
  imshow("dist", dist);
  cv::waitKey(0);
#endif

  cv::Mat dist_8u;
  dist.convertTo(dist_8u, CV_8U, 255.0);

#if 0 //SPD_FIGURE_1
  namedWindow("dist_8u", cv::WINDOW_NORMAL);
  imshow("dist_8u", dist_8u);
  cv::waitKey(0);
#endif

  double threshold = 5.0;
  cv::Mat distStk = convolution(dist_8u, kernel, threshold);

  return distStk;
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
  , const cv::Mat& imgInput
  , const cv::Vec<int, 4>& borders
  , std::vector< cv::Vec<float, 3> >& POINTS
  , std::vector< cv::Vec<float, 3> >& STREAKS
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
  
  std::vector< cv::Vec<int, 3> > firstPOINTS;
  std::vector<std::vector<cv::Point > > firstContoursP;

  if (contoursP.size() > 0)
  {
    firstPOINTS = connectedComponentsPoints
    (max_img_sz, contoursP, borders, firstContoursP);
    contoursP.clear();
  }

  /* Find streaks contours */
  findContours( imgStreaks, contoursS, hierarchyS, CV_RETR_EXTERNAL
                , CV_CHAIN_APPROX_NONE , offset);

  std::vector< cv::Vec<int, 3> > firstSTREAKS;
  std::vector<std::vector<cv::Point > > firstContoursS;

  if (contoursS.size() > 0)
  {
    firstSTREAKS = connectedComponentsStreaks
    (max_img_sz, contoursS, borders, firstContoursS);
    contoursS.clear();
  }
  
  /* Delete overlapping objects */
  std::vector<std::vector<cv::Point > > outContoursP;
  std::vector<std::vector<cv::Point > > outContoursS;
  std::vector< cv::Vec<int, 3> > delOverlapPOINTS;
  std::vector< cv::Vec<int, 3> > delOverlapSTREAKS;

  deleteOverlapping
  (imgSz, firstPOINTS, firstSTREAKS, firstContoursP, firstContoursS
    , delOverlapPOINTS, delOverlapPOINTS, outContoursP, outContoursS);
  firstPOINTS.clear();
  firstSTREAKS.clear();
  firstContoursP.clear();
  firstContoursS.clear();

  preciseCentroid(imgInput, outContoursP, POINTS);
  preciseCentroid(imgInput, outContoursS, STREAKS);

#if SPD_FIGURE_1
    /// Draw contours
    cv::Mat drawing = cv::Mat::zeros(imgPoints.size(), CV_8UC3);    
    int cIdx = -1;
    cv::Scalar color = cv::Scalar(0, 255, 0);
    cv::Scalar colorS = cv::Scalar(0, 0, 255);
    int lineType = 8;
    cv::InputArray hierarchy = cv::noArray();
    int maxLevel = 0;
    drawContours(drawing, outContoursP, cIdx, color, 1, 8, hierarchy, 0, offset);
    drawContours(drawing, outContoursS, cIdx, colorS, 1, 8, hierarchy, 0, offset);
    
    // Show in a window
    namedWindow("Contours", cv::WINDOW_NORMAL);
    imshow("Contours", drawing);
#endif

  outContoursP.clear();
  outContoursS.clear();
}

void connectedComponents2
(
  const cv::Mat& imgPoints
  , const cv::Mat& imgStreaks
  , const cv::Mat& imgInput
  , std::vector< cv::Vec<float, 3> >& POINTS
  , std::vector< cv::Vec<float, 3> >& STREAKS
  , std::vector<std::vector<cv::Point > >& outContoursP
  , std::vector<std::vector<cv::Point > >& outContoursS
)
{
  const cv::Point imgSz = {imgPoints.rows, imgPoints.cols};
  float max_img_sz = std::max(imgPoints.cols, imgPoints.rows);

  cv::Point_<int> I_input_size = { imgInput.cols, imgInput.rows };
  double bordersThick = 0.015;
  cv::Point_<double> perc = { bordersThick, 1 - bordersThick };
  cv::Vec<int, 4> borders = { static_cast<int>( ceil(perc.x * I_input_size.x))
                            , static_cast<int>( ceil(perc.x * I_input_size.y))
                            , static_cast<int>(floor(perc.y * I_input_size.x))
                            , static_cast<int>(floor(perc.y * I_input_size.y)) };

  std::vector<std::vector<cv::Point> > contoursP;
  std::vector<std::vector<cv::Point> > contoursS;
  std::vector<cv::Vec4i> hierarchyP;
  std::vector<cv::Vec4i> hierarchyS;
  cv::Point offset = cv::Point(0, 0);
  
  /* Find points contours */
  findContours( imgPoints, contoursP, hierarchyP, CV_RETR_EXTERNAL
                , CV_CHAIN_APPROX_NONE , offset);

  std::vector< cv::Vec<int, 3> > firstPOINTS;
  std::vector<std::vector<cv::Point > > firstContoursP;

  if (contoursP.size() > 0)
  {
    firstPOINTS = connectedComponentsPoints2
    (max_img_sz, contoursP, borders, firstContoursP);
    contoursP.clear();
  }

  /* Find streaks contours */
  findContours( imgStreaks, contoursS, hierarchyS, CV_RETR_EXTERNAL
                , CV_CHAIN_APPROX_NONE , offset);

  std::vector< cv::Vec<int, 3> > firstSTREAKS;
  std::vector<std::vector<cv::Point > > firstContoursS;

  if (contoursS.size() > 0)
  {
    firstSTREAKS = connectedComponentsStreaks2
    (contoursS, borders, firstContoursS);
    contoursS.clear();
  }
  
  /* Delete overlapping objects */
  
  deleteOverlapping2
  (imgSz, firstPOINTS, firstSTREAKS, firstContoursP, firstContoursS
    , outContoursP, outContoursS);
  firstPOINTS.clear();
  firstSTREAKS.clear();
  firstContoursP.clear();
  firstContoursS.clear();

  preciseCentroid(imgInput, outContoursP, POINTS);
  preciseCentroid(imgInput, outContoursS, STREAKS);

  if (STREAKS.size() > 1)
  {
    dashedLine(imgStreaks, outContoursS, STREAKS);
  }
#if SPD_FIGURE_1
  /// Draw contours
  cv::Mat drawing = cv::Mat::zeros(imgPoints.size(), CV_8UC3);
  int cIdx = -1;
  cv::Scalar color = cv::Scalar(0, 255, 0);
  cv::Scalar colorS = cv::Scalar(0, 0, 255);
  int lineType = 8;
  cv::InputArray hierarchy = cv::noArray();
  int maxLevel = 0;
  drawContours(drawing, outContoursP, cIdx, color, 1, 8, hierarchy, 0, offset);
  drawContours(drawing, outContoursS, cIdx, colorS, 1, 8, hierarchy, 0, offset);

  // Show in a window
  namedWindow("Contours", cv::WINDOW_NORMAL);
  imshow("Contours", drawing);
  cv::waitKey(0);
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

std::vector< cv::Vec<int, 3> > connectedComponentsPoints2
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

    cv::Point centerP ( static_cast<int>(round(center.x)) 
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
      if(contours[i].size()>20)//5
      {
        cv::RotatedRect rotatedRect = fitEllipse(contours[i]);
        majorAxis = rotatedRect.size.height;
        minorAxis = rotatedRect.size.width;
        
        if (minorAxis<1.9 || majorAxis<5)
        {
          continue;
        }
        
        /* Identify linear connect components */
        if (majorAxis / minorAxis > 6)//4
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

std::vector< cv::Vec<int, 3> > connectedComponentsStreaks2
(
  const std::vector<std::vector<cv::Point > >& contours
  , const cv::Vec<int, 4>& borders
  , std::vector<std::vector<cv::Point > >& outContoursRes
)
{
  std::vector< cv::Vec<int, 3> > STREAKS;

  for (size_t i = 0; i < contours.size(); ++i)
  {
    cv::Point2f center = {0.0f, 0.0f};
    float radius = 0.0f;
    minEnclosingCircle(contours[i], center, radius);

    cv::Point centerP ( static_cast<int>(round(center.x)) 
                      , static_cast<int>(round(center.y)) );
        
    if(   (centerP.x>borders[0] && centerP.x<borders[2]) 
       && (centerP.y>borders[1] && centerP.y<borders[3]))
    {
      if(contours[i].size()>20)//5
      {
        cv::RotatedRect rotRec = fitEllipse(contours[i]);
        float majorAxis = rotRec.size.height;
        float minorAxis = rotRec.size.width;
        
        if (minorAxis<1.9 || majorAxis<5)
        {
          continue;
        }
        
        /* Identify linear connect components */
        if (majorAxis / minorAxis > 3.78)//4
        {
          outContoursRes.push_back(contours[i]);
          STREAKS.push_back(
          { static_cast<int>(rotRec.center.x)
            , static_cast<int>(rotRec.center.y), 0 });
        }
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
  , const std::vector<std::vector<cv::Point > >& incontoursP
  , const std::vector<std::vector<cv::Point > >& incontoursS
  , std::vector< cv::Vec<int, 3> >& outPOINTS
  , std::vector< cv::Vec<int, 3> >& outSTREAKS
  , std::vector<std::vector<cv::Point > >& outContoursP
  , std::vector<std::vector<cv::Point > >& outContoursS
)
{
  /* Delete streaks on points */
  cv::Mat imgP = cv::Mat::zeros(imgSz.x, imgSz.y, CV_8U);
  const cv::Scalar color = 255;
  int lineType = 8;
  cv::InputArray hierarchy = cv::noArray();
  int maxLevel = 0;
  cv::Point offset = {0,0};
  for (size_t l = 0; l < incontoursP.size(); ++l) {
    drawContours(imgP, incontoursP, l, color, CV_FILLED, lineType, hierarchy, maxLevel, offset);
  }
  
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
#if SPD_DEBUG
  // Show in a window
  namedWindow("ContoursP", cv::WINDOW_NORMAL);
  imshow("ContoursP", imgP);
  cv::waitKey(0);
#endif
  imgP.release();
    
  if (n_streaks)
  {    
    for (size_t k = 0; k < inSTREAKS.size(); ++k)
    {
      if (1 == inSTREAKS.at(k)[2])
      {
        outSTREAKS.push_back({ inSTREAKS.at(k)[0], inSTREAKS.at(k)[1], 0});
        outContoursS.push_back(incontoursS[k]);
      }
    }
  }

  /* Delete points on streak */
  cv::Mat imgS = cv::Mat::zeros(imgSz.x, imgSz.y, CV_8U);
  for (size_t l = 0; l < outContoursS.size(); ++l) {
    drawContours(imgS, outContoursS, l, color, CV_FILLED, lineType, hierarchy, maxLevel, offset);
  }
    
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
#if SPD_DEBUG
  // Show in a window
  namedWindow("ContoursS", cv::WINDOW_NORMAL);
  imshow("ContoursS", imgS);
  cv::waitKey(0);
#endif
  imgS.release();
    
  if (n_points)
  {    
    for (size_t k = 0; k < inPOINTS.size(); ++k)
    {
      if (1 == inPOINTS.at(k)[2])
      {
        outPOINTS.push_back({ inPOINTS.at(k)[0], inPOINTS.at(k)[1], 0});
        outContoursP.push_back(incontoursP[k]);
      }
    }
  }    
}

void deleteOverlapping2
(
  const cv::Point imgSz
  , std::vector< cv::Vec<int, 3> >& inPOINTS
  , std::vector< cv::Vec<int, 3> >& inSTREAKS
  , const std::vector<std::vector<cv::Point > >& incontoursP
  , const std::vector<std::vector<cv::Point > >& incontoursS
  , std::vector<std::vector<cv::Point > >& outContoursP
  , std::vector<std::vector<cv::Point > >& outContoursS
)
{
  /* Delete streaks on points */
  cv::Mat imgP = cv::Mat::zeros(imgSz.x, imgSz.y, CV_8U);
  const cv::Scalar color = 255;
  int lineType = 8;
  cv::InputArray hierarchy = cv::noArray();
  int maxLevel = 0;
  cv::Point offset = {0,0};
  for (size_t l = 0; l < incontoursP.size(); ++l) {
    drawContours(imgP, incontoursP, l, color, CV_FILLED, lineType, hierarchy, maxLevel, offset);
  }
  
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
#if SPD_DEBUG
  // Show in a window
  namedWindow("ContoursP", cv::WINDOW_NORMAL);
  imshow("ContoursP", imgP);
  cv::waitKey(0);
#endif
  imgP.release();
    
  if (n_streaks)
  {    
    for (size_t k = 0; k < inSTREAKS.size(); ++k)
    {
      if (1 == inSTREAKS.at(k)[2])
      {
        outContoursS.push_back(incontoursS[k]);
      }
    }
  }

  /* Delete points on streak */
  cv::Mat imgS = cv::Mat::zeros(imgSz.x, imgSz.y, CV_8U);
  for (size_t l = 0; l < outContoursS.size(); ++l) {
    drawContours(imgS, outContoursS, l, color, CV_FILLED, lineType, hierarchy, maxLevel, offset);
  }
    
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
#if SPD_DEBUG
  // Show in a window
  namedWindow("ContoursS", cv::WINDOW_NORMAL);
  imshow("ContoursS", imgS);
  cv::waitKey(0);
#endif
  imgS.release();
    
  if (n_points)
  {    
    for (size_t k = 0; k < inPOINTS.size(); ++k)
    {
      if (1 == inPOINTS.at(k)[2])
      {
        outContoursP.push_back(incontoursP[k]);
      }
    }
  }    
}

/* ==========================================================================
*        FUNCTION NAME: preciseCentroid
* FUNCTION DESCRIPTION: Compute precise centroid position
*        CREATION DATE: 20160727
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
void preciseCentroid
(
  const cv::Mat& img
  , const std::vector<std::vector<cv::Point > >& contours
  , std::vector< cv::Vec<float, 3> >& center
)
{
  std::vector<cv::Point> pixelIdList;
  int epsB = 1;
  for (size_t i = 0; i < contours.size(); ++i)
  {
    cv::Rect br = boundingRect(contours.at(i));
    int minX = br.x;
    int minY = br.y;
    
    for (int y = 0; y < br.height + 2*epsB; ++y)
    {
      int row = static_cast<int>(minY +y - epsB);
      for (int x=0; x< br.width + 2*epsB; ++x)
      {        
        int col = static_cast<int>(minX + x - epsB);
        cv::Point pIn = { col, row };
        bool insideP = rayCasting(contours.at(i), pIn);
        if (insideP)
        {
          pixelIdList.push_back(pIn);
        }
      } //for col       
    } //for row
    /* Mancano alcuni punti del contorno */
   
    /* Compute object's barycentre */
    cv::Point2f p = { 0.0f, 0.0f };
    barycentre(img, pixelIdList, p);

    center.push_back({p.x, p.y, 0.0f});

    pixelIdList.clear();
  }
}

#if 0
void dashedLine
(
  const cv::Mat& img
  , const std::vector<std::vector<cv::Point > >& cIn
  , std::vector<std::vector<cv::Point > >& cOut
)
{
  float distAlongThresh = 55;
  float alongThresh = 0;
  int cntAlongThresh = 0;
  float distCrossThresh = 5;

  /* Mappa delle linee con uguale angolo di inclinazione */
  std::unordered_map<int, std::vector<int>> inclAngle;

  for (int i = 0; i < cIn.size(); ++i)
  {
    cv::RotatedRect rotRec = fitEllipse(cIn[i]);
    float angle = rotRec.angle;
    float minorAxis = rotRec.size.width;
    alongThresh += minorAxis;
    cntAlongThresh++;
    //prendere solo 4 cifre decimali di angle tipo floor(angle*1000)
    int key = static_cast<int>(floor(angle));
    if (0 == inclAngle.count(key))
    {
      inclAngle.insert({ key, {i} });
    }
    else
    {
      inclAngle.at(key).push_back(i);
    }
  }
  
  alongThresh = alongThresh / cntAlongThresh;
  distAlongThresh = 55;

  /* Sort streak by x */
  std::vector<std::vector<std::pair<float,size_t>>> orderLine;
  size_t indx = 0;
  for (auto& x : inclAngle)
  {
    if (x.second.size() > 1)
    {
      orderLine.push_back({});
      indx++;
      //  looping for each inclination
      for (int i = 0; i < x.second.size(); ++i)
      {
        cv::Point2f head = {};
        cv::Point2f tail = {};
        float m = .0f;
        float q = .0f;
        headTail(cIn[x.second.at(i)], x.first, head, tail, m, q);
        headTail(cIn[x.second.at(i)], head, tail, m, q);
        orderLine.at(indx - 1).push_back({head.x,x.second.at(i)});
      }
    }
  }

  for (int i=0; i<orderLine.size();++i)
  {
    std::sort(orderLine.at(i).begin(), orderLine.at(i).end());
  }


  /* Studio delle linee con stessa inclinazione */
  std::map<size_t, std::vector<size_t>> mAlignLine;
  std::set<size_t> usedLine;

  for (auto& x : orderLine) 
  {
    //  looping for each streak
    for (int i = 0; i < x.size(); ++i)
    {
      cv::Point2f head = {};
      cv::Point2f tail = {};
      float m = .0f;
      float q = .0f;
      cv::RotatedRect rotRec = fitEllipse(cIn[x.at(i).second]);
      float angle = rotRec.angle;
      headTail(cIn[x.at(i).second], angle, head, tail, m, q);
      headTail(cIn[x.at(i).second], head, tail, m, q);

      /* Calcolo dei 2 punti per cui passa la retta parallela alla 
       * striscia che identificano la regione di piano ammissibile 
       */
      float distX = ::sin(angle * CV_PI / 360)*distAlongThresh;
      float distY = ::cos(angle * CV_PI / 360)*distAlongThresh;

      cv::Point2f center = { 0.0f, 0.0f };
      float radius = 0.0f;
      minEnclosingCircle(cIn[x.at(i).second], center, radius);
      
      cv::Point2f pntA = {};
      cv::Point2f pntB = {};
      cv::Point2f pntC = {};
      cv::Point2f pntD = {};

      if (angle > 90) {
        pntA = { std::min(center.x - distX, center.x + distX), std::min(center.y - distY, center.y + distY) };
        pntB = { std::max(center.x - distX, center.x + distX), std::max(center.y - distY, center.y + distY) };
        pntC = { center.x - abs(distX), center.y - abs(distY) };
        pntD = { center.x + abs(distX), center.y + abs(distY) };
      }
      else {
        pntA = { std::max(center.x - distX, center.x + distX), std::min(center.y - distY, center.y + distY) };
        pntB = { std::min(center.x - distX, center.x + distX), std::max(center.y - distY, center.y + distY) };
        pntC = { center.x + abs(distX), center.y - abs(distY) };
        pntD = { center.x - abs(distX), center.y + abs(distY) };
      }

      float qA = pntA.y - m*pntA.x;
      float qB = pntB.y - m*pntB.x;

      //  looping from i + 1 to ignore same pair again
      for (int j = i + 1; j < x.size(); j++)
      {
        cv::Point2f headNext = {};
        cv::Point2f tailNext = {};
        float m2 = .0f;
        float q2 = .0f;
        headTail(cIn[x.at(j).second], angle, headNext, tailNext, m2, q2);
        headTail(cIn[x.at(j).second], headNext, tailNext, m2, q2);

        if ((q2 > std::min(qA, qB)) && (q2 < std::max(qA, qB)))
        {
          if (0 == mAlignLine.count(x.at(i).second)) {
            if ((0 == usedLine.count(x.at(i).second)) && (0 == usedLine.count(x.at(j).second))) {
              mAlignLine.insert({ x.at(i).second, {x.at(j).second} });
              usedLine.insert(x.at(i).second);
              usedLine.insert(x.at(j).second);
            }
          }
          else {
            if (0 == usedLine.count(x.at(j).second)) {
              mAlignLine.at(x.at(i).second).push_back(x.at(j).second);
              usedLine.insert(x.at(j).second);
            }
          }

        }

      }
    }
  }


#if 0
  for (auto& x : inclAngle) 
  {
    if (x.second.size() > 1)
    {
      //  looping for each inclination
      for (int i = 0; i < x.second.size(); ++i)
      {
        cv::Point2f head = {};
        cv::Point2f tail = {};
        float m = .0f;
        float q = .0f;

        headTail(cIn[x.second.at(i)], x.first, head, tail, m, q);

        cv::Point2f center = { 0.0f, 0.0f };
        float radius = 0.0f;
        minEnclosingCircle(cIn[x.second.at(i)], center, radius);

        float distX = ::sin(x.first * CV_PI / 360)*distCrossThresh;
        float distY = ::cos(x.first * CV_PI / 360)*distCrossThresh;

        /* Punti retta perpendicolare a retta distanti d*/
        float xA = center.x - distX;
        float yA = center.y + distY;

        float xB = center.x + distX;
        float yB = center.y - distY;

        //  looping from i + 1 to ignore same pair again
        for (int j = i + 1; j < x.second.size(); j++)
        {
          cv::Point2f headNext = {};
          cv::Point2f tailNext = {};
          float m2 = .0f;
          float q2 = .0f;
          headTail(cIn[x.second.at(j)], x.first, headNext, tailNext, m2, q2);

          /*controllo se allineata */
          float c1 = m*(headNext.x - xA) + yA;//>
          float c2 = m*(headNext.x - xB) + yB;//<
          float c3 = m*(tailNext.x - xA) + yA;//>
          float c4 = m*(tailNext.x - xB) + yB;//<

          if (!((c1 > 0) && (c3 > 0) && (c2 < 0) && (c4 < 0)))
          {
            continue;
          }          
          
          /* Controllo se troppo distante */
          if (sqrt(pow(tail.x-headNext.x,2)+pow(tail.y-headNext.y,2))>distAlongThresh)
          {
            continue;
          }

          printf("Arrivasti");
        }
      }
    }    
  }
#endif

}

#else
void dashedLine
(
  const cv::Mat& img
  , std::vector<std::vector<cv::Point > >& cIn
  , std::vector< cv::Vec<float, 3> >& inSTREAKS
)
{
  float distCrossThresh = 55;
  float crossThresh = 0;  
  float distAlongThresh = 5;
  float alongThresh = 0;
    
  /* Sort streak by x */
  std::vector<std::pair<float,size_t>> orderLine;
  
  for (size_t x=0;x<cIn.size();++x)
  {
    cv::RotatedRect rotRec = fitEllipse(cIn[x]);
    float angle = rotRec.angle;    
    float majorAxis = rotRec.size.height;
    float minorAxis = rotRec.size.width;
    crossThresh += minorAxis;
    alongThresh += majorAxis;
#if 1
    cv::Point2f center = { 0.0f, 0.0f };
    float radius = 0.0f;
    minEnclosingCircle(cIn.at(x), center, radius);
    orderLine.push_back({ center.x,x });
#else
    cv::Point2f head = {};
    cv::Point2f tail = {};
    float m = .0f;
    float q = .0f;
    headTail(cIn.at(x), head, tail, m, q);
    orderLine.push_back({ head.x,x });
#endif    
  }
  distCrossThresh = (crossThresh/ cIn.size())*2;
  distAlongThresh = (alongThresh/ cIn.size())*0.9;

  std::sort(orderLine.begin(), orderLine.end());
  

  /* Studio delle linee con stessa inclinazione */
  std::map<size_t, std::vector<size_t>> mAlignLine;
  std::set<size_t> usedLine;

  for (int i = 0; i < orderLine.size(); ++i)
  {
    cv::Point2f head = {};
    cv::Point2f tail = {};
    float m = .0f;
    float q = .0f;
    cv::RotatedRect rotRec = fitEllipse(cIn[orderLine.at(i).second]);
    float angle = rotRec.angle;
    headTail(cIn[orderLine.at(i).second], head, tail, m, q);

    /* Calcolo dei 2 punti per cui passa la retta parallela alla
     * striscia che identificano la regione di piano ammissibile
     */
    float distX = ::sin(angle * CV_PI / 360)*distCrossThresh;
    float distY = ::cos(angle * CV_PI / 360)*distCrossThresh;

    cv::Point2f center = { 0.0f, 0.0f };
    float radius = 0.0f;
    minEnclosingCircle(cIn[orderLine.at(i).second], center, radius);

    cv::Point2f pntA = {};
    cv::Point2f pntB = {};

    if (angle > 90) {
      pntA = { center.x - abs(distX), center.y - abs(distY) };
      pntB = { center.x + abs(distX), center.y + abs(distY) };
    }
    else {
      pntA = { center.x + abs(distX), center.y - abs(distY) };
      pntB = { center.x - abs(distX), center.y + abs(distY) };
    }

    float qA = pntA.y - m*pntA.x;
    float qB = pntB.y - m*pntB.x;

    //  looping from i + 1 to ignore same pair again
    for (int j = i + 1; j < orderLine.size(); j++)
    {
      cv::Point2f headNext = {};
      cv::Point2f tailNext = {};
      float m2 = .0f;
      float q2 = .0f;
      headTail(cIn[orderLine.at(j).second], headNext, tailNext, m2, q2);

      if ((q2 > std::min(qA, qB)) && (q2 < std::max(qA, qB)))
      {
        if (0 == mAlignLine.count(orderLine.at(i).second)) {
          if ((0 == usedLine.count(orderLine.at(i).second)) 
            && (0 == usedLine.count(orderLine.at(j).second))) {
            mAlignLine.insert({ orderLine.at(i).second, {orderLine.at(j).second} });
            usedLine.insert(orderLine.at(i).second);
            usedLine.insert(orderLine.at(j).second);
          }
        }
        else {
          if (0 == usedLine.count(orderLine.at(j).second)) {
            mAlignLine.at(orderLine.at(i).second).push_back(orderLine.at(j).second);
            usedLine.insert(orderLine.at(j).second);
          }
        }
      }
    }
  }

  /* Studio distanze tra strisciate */
  std::vector<std::set<size_t>> finalLine;
  size_t ind = 0;
  for (auto& x : mAlignLine)
  {
    finalLine.push_back({});
    
    std::vector<size_t> indexS;
    indexS.push_back(x.first);
    for (size_t u = 0; u < x.second.size(); ++u)
    {
      indexS.push_back(x.second.at(u));
    }
       
    for (size_t i = 0; i < indexS.size()-1; ++i)
    {
      cv::Point2f head = {};
      cv::Point2f tail = {};
      float m = .0f;
      float q = .0f;
      cv::RotatedRect rotRec = fitEllipse(cIn[orderLine.at(i).second]);
      float angle = rotRec.angle;
      headTail(cIn[indexS.at(i)], head, tail, m, q);

      size_t streakB = x.second.at(i);
      cv::Point2f headNext = {};
      cv::Point2f tailNext = {};
      float m2 = .0f;
      float q2 = .0f;
      headTail(cIn[indexS.at(i+1)], headNext, tailNext, m2, q2);

      float dist = sqrt(pow(tail.x-headNext.x, 2) + pow(tail.y-headNext.y, 2));
      if (dist < distAlongThresh)
      {
        finalLine.at(ind).insert(indexS.at(i));
        finalLine.at(ind).insert(indexS.at(i+1));
      }
    }
    ind++;
  }

  /* Calcolo nuovi centroidi */
  std::vector< cv::Vec<float, 3> > streak;
  std::vector<std::vector<cv::Point > > cOut;
  
  for (size_t i = 0; i < finalLine.size(); ++i)
  {
    float sumX = 0;
    float sumY = 0;
    size_t num = finalLine.at(i).size();
    std::vector<cv::Point> sumP;
    //for (size_t j = 0;j<num;++j)
    for(auto& x : finalLine.at(i))
    {
      sumX += inSTREAKS.at(x)[0];
      sumY += inSTREAKS.at(x)[1];

      sumP.insert( sumP.end(), cIn.at(x).begin(), cIn.at(x).end() );
    }
    streak.push_back({sumX/num,sumY/num,0.0f});
    cOut.push_back(sumP);
  }

  /* Copia strisciate corrette */
  for (size_t i = 0; i < inSTREAKS.size(); ++i)
  {
    for (size_t j = 0; j < finalLine.size(); ++j)
    {
      if (0 == finalLine.at(j).count(i))
      {
        streak.push_back(inSTREAKS.at(i));
        cOut.push_back(cIn.at(i));
      }
    }
  }

  inSTREAKS = streak;
  cIn = cOut;
}
#endif

void headTail
(
  const std::vector<cv::Point >& pnt
  , cv::Point2f& head
  , cv::Point2f& tail
  , float& m, float& q
)
{
  cv::RotatedRect rotRec = fitEllipse(pnt);
  float angle = rotRec.angle;

  //Point of the roted rectangle RoI
  //Point of the roted rectangle big image
  cv::Point2f pRotRec[4];
  rotRec.points(pRotRec);
  cv::Point2f pRotRecRoI[4];
  pRotRecRoI[0] = pRotRec[0];
  pRotRecRoI[1] = pRotRec[1];
  pRotRecRoI[2] = pRotRec[2];
  pRotRecRoI[3] = pRotRec[3];

  //Central point of the roted rectangle RoI
  cv::Point2f pMinV[4];
  pMinV[0] = { std::min(pRotRecRoI[0].x , pRotRecRoI[1].x) , std::min(pRotRecRoI[0].y , pRotRecRoI[1].y) };
  pMinV[1] = { std::min(pRotRecRoI[1].x , pRotRecRoI[2].x) , std::min(pRotRecRoI[1].y , pRotRecRoI[2].y) };
  pMinV[2] = { std::min(pRotRecRoI[2].x , pRotRecRoI[3].x) , std::min(pRotRecRoI[2].y , pRotRecRoI[3].y) };
  pMinV[3] = { std::min(pRotRecRoI[3].x , pRotRecRoI[0].x) , std::min(pRotRecRoI[3].y , pRotRecRoI[0].y) };

  cv::Point2f pCenterP[4];
  pCenterP[0] = { pMinV[0].x + std::abs(pRotRecRoI[0].x - pRotRecRoI[1].x) / 2 , pMinV[0].y + std::abs(pRotRecRoI[0].y - pRotRecRoI[1].y) / 2 };
  pCenterP[1] = { pMinV[1].x + std::abs(pRotRecRoI[1].x - pRotRecRoI[2].x) / 2 , pMinV[1].y + std::abs(pRotRecRoI[1].y - pRotRecRoI[2].y) / 2 };
  pCenterP[2] = { pMinV[2].x + std::abs(pRotRecRoI[2].x - pRotRecRoI[3].x) / 2 , pMinV[2].y + std::abs(pRotRecRoI[2].y - pRotRecRoI[3].y) / 2 };
  pCenterP[3] = { pMinV[3].x + std::abs(pRotRecRoI[3].x - pRotRecRoI[0].x) / 2 , pMinV[3].y + std::abs(pRotRecRoI[3].y - pRotRecRoI[0].y) / 2 };


  if (angle > 90) {
    head = { std::max(pCenterP[1].x,pCenterP[3].x),std::min(pCenterP[1].y,pCenterP[3].y) };
    tail = { std::min(pCenterP[1].x,pCenterP[3].x),std::max(pCenterP[1].y,pCenterP[3].y) };
  }
  else {
    head = { std::min(pCenterP[1].x,pCenterP[3].x),std::min(pCenterP[1].y,pCenterP[3].y) };
    tail = { std::max(pCenterP[1].x,pCenterP[3].x),std::max(pCenterP[1].y,pCenterP[3].y) };
  }

  /* Coefficienti retta */
  m = (head.y - tail.y) / (head.x - tail.x);
  q = -tail.x*m + tail.y;
}

/* ==========================================================================
*        FUNCTION NAME: barycentre
* FUNCTION DESCRIPTION: Compute baricentre position
*        CREATION DATE: 20160911
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
void barycentre
(
  const cv::Mat& img
  , const std::vector<cv::Point> pixelIdList
  , cv::Point2f& p
)
{
  int col = 0;
  int row = 0;
  const ushort* pLine = nullptr;
  ushort valueI = 0;

  float sumValue = 0;
  float sumProdValPosX = 0;
  float sumProdValPosY = 0;

  for (size_t i = 0; i < pixelIdList.size(); ++i)
  {
    col = pixelIdList[i].x;
    row = pixelIdList[i].y;

    pLine = img.ptr<ushort>(row);
    valueI = pLine[col];
    
    sumValue = sumValue + valueI;
    sumProdValPosX = sumProdValPosX + valueI*col;
    sumProdValPosY = sumProdValPosY + valueI*row;

  }

  p = {sumProdValPosX/sumValue, sumProdValPosY/sumValue};
}

/* ==========================================================================
*        FUNCTION NAME: rayCasting
* FUNCTION DESCRIPTION: Ray Casting algorithm
*        CREATION DATE: 20160911
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
bool rayCasting
(
  const std::vector<cv::Point > & poly
  , const cv::Point& p
)
{
  bool inside = false;
  cv::Point2f pnt = { static_cast<float>(p.x), static_cast<float>(p.y) };
  
  const std::numeric_limits<float> FLOAT;
  float eps = 0.001;// FLOAT.epsilon();
  float huge = FLOAT.max();

  for (size_t e = 0; e < poly.size(); ++e)
  {    
    /* Selezione punti */
    cv::Point2f Aa = {static_cast<float>(poly.at(e).x), static_cast<float>(poly.at(e).y)};
    size_t q = (e+1) % poly.size();
    cv::Point2f Bb = {static_cast<float>(poly.at(q).x), static_cast<float>(poly.at(q).y)};

    /* Ordina punti con A.y<B.y */
    cv::Point2f A = Aa;
    cv::Point2f B = Bb;
    if (Aa.y > Bb.y)
    {
      A = Bb;
      B = Aa;
    }
    
    /* Verifica che il punto non \E8 alla stessa altezza dei vertici */
    if ((pnt.y == B.y) || (pnt.y == A.y))
    {
      pnt.y += eps;
    }

    /* Verifica se il punto \E8 sopra (1) o sotto al segmento (2) */
    /* Verifica se il punto \E8 alla destra del segmento (3) */
    if ((pnt.y > B.y) || (pnt.y < A.y) || (pnt.x > std::max(A.x, B.x)))
    {
      continue;
    }

    /* Il punto \E8 all'estrema sinistra del segmento quindi lo interseca */
    if ( pnt.x < std::min(A.x, B.x) )
    {
      inside = !inside;
      continue;
    }

    /* Verifica se il punto \E8 alla sinistra o destra del segmento */
    float m_edge;
    /*if (0.0f == B.x - A.x)
    {m_edge = huge;}
    else
    {m_edge = (B.y-A.y) / (B.x-A.x);}*/
    m_edge = B.x - A.x > FLOAT.epsilon() ? (B.y-A.y) / (B.x-A.x) : huge;
    
    float m_pnt;
    /*if (0.0f == pnt.x - A.x)
    {m_pnt = huge;}
    else
    {m_pnt = (pnt.y-A.y) / (pnt.x-A.x);}*/

    m_pnt = pnt.x - A.x > FLOAT.epsilon() ? (pnt.y-A.y) / (pnt.x-A.x) : huge;
    
    if (m_pnt >= m_edge)
    {
      inside = !inside;
      continue;
    }
  }
  return inside;
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
  int goodVal = 5;
  
  std::vector<cv::Vec2f> houghVal;
  std::unordered_map<size_t, int> houghThresh;
  HoughLines(imgIn, houghVal, rho, theta, threshold);
  houghThresh.insert({houghVal.size(), threshold});
  
  // Loop for find lines with high thresoldh
  size_t maxNumAngles = 10;
  bool exitL = true;
  bool foundGoodThresh = false;
  size_t countC = 0;

  while (exitL)
  {
    countC = countC + 1;
    if (houghVal.size() > maxNumAngles)
    {
      threshold = threshold * 1.5;
      HoughLines(imgIn, houghVal, rho, theta, threshold);
      if (0 == houghThresh.count(houghVal.size())) {
          houghThresh.insert({houghVal.size(), threshold});
      }
    }    
    else if ((houghVal.size()>=goodVal) && (houghVal.size() <= 2*goodVal))
    {
      exitL = false;
      foundGoodThresh = true;
    }
    else
    {
      threshold = threshold / 1.05;
      HoughLines(imgIn, houghVal, rho, theta, threshold);
      if (0 == houghThresh.count(houghVal.size())) {
          houghThresh.insert({houghVal.size(), threshold});
      }
    }
    if (10 == countC) {
      exitL = false;
    }
  }
    
  if (!foundGoodThresh)
  {
    if (houghThresh.size() > 1)
    {
      /* Sort mapped keys */
      std::vector<size_t> keys;
      keys.reserve(houghThresh.size());

      for (auto& it : houghThresh) {
        keys.push_back(it.first);
      }
      std::sort(keys.begin(), keys.end());

      /* Found the value near to goodVal */
      std::vector<int> keys2;
      keys2.reserve(houghThresh.size());
      for (size_t i = 0; i < keys.size(); ++i)
      {
        keys2.push_back(keys.at(i) - goodVal);
      }
      int currV = 0;
      int nearV = keys2.at(0);
      for (size_t i = 0; i < keys2.size(); ++i) 
      {        
        currV = keys2.at(i) * keys2.at(i);
        if (currV <= (nearV * nearV)) {
          nearV = keys2.at(i);
        }
      }
      HoughLines(imgIn, houghVal, rho, theta, houghThresh[nearV+goodVal]);
    }        
  }

  // Select the inclination angles
  std::vector<float> angle;
  for (size_t i = 0; i < houghVal.size(); ++i)
  {
    angle.push_back(houghVal.at(i)[1]);
  }
  angle.push_back(CV_PI / 2); //Force research at 0\B0
  
  size_t count = 0;
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
  int yDim = static_cast<int>(::ceil(dimLine * ::fabs(::sin((CV_PI/2)-teta))));
  int xDim = static_cast<int>(::ceil(dimLine * ::fabs(::cos((CV_PI/2)-teta))));

  /* Minimum dimensions */
  if(yDim<3){
    yDim = 3;
  }
  if(xDim<3){
    xDim = 3;
  }

  /* Odd dimensions */
  if (0 == yDim % 2) {
    yDim = yDim + 1;
  }
  if (0 == xDim % 2) {
    xDim = xDim + 1;
  }
    
  cv::Mat kernel = cv::Mat::zeros(yDim, xDim, CV_8U);

  cv::Point pt1 = { 0, 0 };
  cv::Point pt2 = { 0, 0 };
  if (teta > 0) {
    pt1 = { 0, yDim };
    pt2 = { xDim, 0 };
  }
  else {
    pt1 = { 0, 0 };
    pt2 = { xDim, yDim };
  }
  
  //printf("xDim=%d  yDim=%d ", xDim, yDim);
  
  const cv::Scalar color = cv::Scalar(255, 255, 255);
  int thickness = 1;
  int lineType = 8;
  int shift = 0;
  line(kernel, pt1, pt2, color, thickness, lineType, shift);

#if SPD_FIGURE_1
    // Create a window for display.
    namedWindow("Kernel", cv::WINDOW_NORMAL);
    imshow("Kernel", kernel);    
    cv::waitKey(0);
#endif

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
  , const std::vector< cv::Vec<float, 3> >& POINTS
  , const std::vector< cv::Vec<float, 3> >& STREAKS
)
{
  std::string s_nS = "Detected streaks: " + std::to_string(STREAKS.size());
  stamp(stream, s_nS.c_str());
  for (size_t i = 0; i < STREAKS.size(); ++i)
  {
    std::string cS = "Centroid streaks: (" + std::to_string(STREAKS.at(i)[0]) + "," + std::to_string(STREAKS.at(i)[1]) + ")";
    stamp(stream, cS.c_str());
  }

  std::string s_nP = "Detected points: " + std::to_string(POINTS.size());
  stamp(stream, s_nP.c_str());
  for (size_t i = 0; i < POINTS.size(); ++i)
  {
    std::string cP = "Centroid points: (" + std::to_string(POINTS.at(i)[0]) + "," + std::to_string(POINTS.at(i)[1]) + ")";
    stamp(stream, cP.c_str());
  }
}

void writeResult
(
  std::ostream& stream
  , const std::vector< cv::Vec<float, 3> >& POINTS
  , const std::vector< cv::Vec<float, 3> >& STREAKS
  , const std::vector< cv::Vec<double, 3> >& radecP
  , const std::vector< cv::Vec<double, 3> >& radecS
)
{
  std::string s_nS = "Detected streaks: " + std::to_string(STREAKS.size());
  stamp(stream, s_nS.c_str());
  for (size_t i = 0; i < STREAKS.size(); ++i)
  {
    /*std::string cS = "Centroid streaks: Row Col (" + std::to_string(STREAKS.at(i)[0]) + "," + std::to_string(STREAKS.at(i)[1]) + ")" 
      + " Ra Dec (" + std::to_string(radecS.at(i)[0]) + "," + std::to_string(radecS.at(i)[1]) + ")";*/

    char buffer [512];
    int n = sprintf(buffer, "Centroid streaks: Row Col (%f,%f) Ra Dec (%18.15f,%18.15f)"
      , STREAKS.at(i)[0], STREAKS.at(i)[1], radecS.at(i)[0], radecS.at(i)[1]);

    stamp(stream, buffer);
  }

  std::string s_nP = "Detected points: " + std::to_string(POINTS.size());
  stamp(stream, s_nP.c_str());
  for (size_t i = 0; i < POINTS.size(); ++i)
  {
    /*std::string cP = "Centroid points: Row Col  (" + std::to_string(POINTS.at(i)[0]) + "," + std::to_string(POINTS.at(i)[1]) + ")"
      + " Ra Dec (" + std::to_string(radecP.at(i)[0]) + "," + std::to_string(radecP.at(i)[1]) + ")";*/

    char buffer [512];
    int n = sprintf(buffer, "Centroid points:  Row Col (%f,%f) Ra Dec (%18.15f,%18.15f)"
      , POINTS.at(i)[0], POINTS.at(i)[1], radecP.at(i)[0], radecP.at(i)[1]);

    stamp(stream, buffer);
  }
}

/* ==========================================================================
*        FUNCTION NAME: plotResult
* FUNCTION DESCRIPTION: Plot result points and streaks centroid
*        CREATION DATE: 20160911
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
void plotResult
(
  const cv::Mat& imgIn
  , const std::vector< cv::Vec<float, 3> >& POINTS
  , const std::vector< cv::Vec<float, 3> >& STREAKS
  , const std::vector<char *>& input
)
{
  cv::Mat color_Img_input;
  cvtColor(imgIn, color_Img_input, CV_GRAY2BGR);

  int radius = 9;
  int radiusS = 11;
  cv::Scalar colorP = { 0,255,0 };
  cv::Scalar colorS = { 0,0,255 };
  int thickness = 2;
  int thicknessS = 3;
  int lineType = 8;
  int shift = 0;

  for (size_t i = 0; i < POINTS.size(); ++i) {
    cv::Point center = { static_cast<int>(POINTS.at(i)[0]), static_cast<int>(POINTS.at(i)[1]) };
    circle(color_Img_input, center, radius, colorP, thickness, lineType, shift);
  }
  for (size_t i = 0; i < STREAKS.size(); ++i) {
    cv::Point center = { static_cast<int>(STREAKS.at(i)[0]), static_cast<int>(STREAKS.at(i)[1]) };
    circle(color_Img_input, center, radiusS, colorS, thicknessS, lineType, shift);
  }

#if SPD_FIGURE
  namedWindow("Algo simple", cv::WINDOW_NORMAL);
  imshow("Algo simple", color_Img_input);
#endif

#if SPD_SAVE_FIGURE
  char s_imgName[256];
  strcpy(s_imgName, input.at(4));
  strcat(s_imgName, input.at(1));
  strcat(s_imgName, ".jpg");
  imwrite(s_imgName, color_Img_input);
#endif

}

/* ==========================================================================
*        FUNCTION NAME: parseWCS
* FUNCTION DESCRIPTION: Parse Wordl Coordinate System
*        CREATION DATE: 20160911
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
void parseWCS(const char* file, wcsPar& par)
{ 
  std::ifstream myfile;
  std::string line;

  myfile.open(file, std::ifstream::in);
    
  if (myfile.is_open()) {
    while (getline(myfile, line)) {
      //std::cout << line << '\n';
    }
    myfile.close();
  }
  else std::cout << "Unable to open file";
  int len = 80;

  char buf[81];
  char name[81];
  char strN[81];
  char strValue[81];

  for (size_t p = 0; p < line.size(); p += len) 
  {
    ::memset(buf,0,sizeof(buf));
    snprintf(buf, 81, "%s", line.c_str() + p);
    
    ::memset(name,0,sizeof(name));
    ::memset(strN,0,sizeof(strN));
    ::memset(strValue,0,sizeof(strValue));
    
    sscanf(buf, "%s %s %s %*s", name, strN, strValue);

    if (0 == strcmp(name, "CRVAL1")){
      par.CRVAL1 = atof(strValue);
    }
    else if (0 == strcmp(name, "CRVAL2")){
      par.CRVAL2 = atof(strValue);
    }
    else if (0 == strcmp(name, "CRPIX1")){
      par.CRPIX1 = atof(strValue);
    }
    else if (0 == strcmp(name, "CRPIX2")){
      par.CRPIX2 = atof(strValue);
    }
    else if (0 == strcmp(name, "CD1_1")){
      par.CD1_1 = atof(strValue);
    }
    else if (0 == strcmp(name, "CD1_2")){      
      par.CD1_2 = atof(strValue);
    }
    else if (0 == strcmp(name, "CD2_1")){
      par.CD2_1 = atof(strValue);
    }
    else if (0 == strcmp(name, "CD2_2")){
      par.CD2_2 = atof(strValue);
    }
    else if (0 == strcmp(name, "A_0_2")){
      par.A_0_2 = atof(strValue);
    }
    else if (0 == strcmp(name, "A_1_1")){
      par.A_1_1 = atof(strValue);
    }
    else if (0 == strcmp(name, "A_2_0")){
      par.A_2_0 = atof(strValue);
    }
    else if (0 == strcmp(name, "B_0_2")){
      par.B_0_2 = atof(strValue);
    }
    else if (0 == strcmp(name, "B_1_1")){
      par.B_1_1 = atof(strValue);
    }
    else if (0 == strcmp(name, "B_2_0")){
      par.B_2_0 = atof(strValue);
    }
    else
    {      
    }
  }
  ::memset(name, 0, sizeof(name));
  ::memset(strN, 0, sizeof(strN));
  ::memset(strValue, 0, sizeof(strValue));
}

/* ==========================================================================
*        FUNCTION NAME: coordConv
* FUNCTION DESCRIPTION: Coordinate Converter
*        CREATION DATE: 20160911
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
void coordConv
(
  const wcsPar& par
  , const std::vector< cv::Vec<float, 3> >& pixel
  , std::vector< cv::Vec<double, 3> >& radec
)
{
  for (size_t i = 0; i<pixel.size(); ++i)
  {
#if 0
    double u = static_cast<double>(pixel.at(i)[0]) - par.CRPIX1 +1;
    double v = static_cast<double>(pixel.at(i)[1]) - par.CRPIX2 +1;

    double f = 0;// par.A_0_2 * v*v + par.A_1_1 * u*v + par.A_2_0 * u*u;
    double g = 0;// par.B_0_2 * v*v + par.B_1_1 * u*v + par.B_2_0 * u*u;

    cv::Mat GM = cv::Mat::zeros(2, 2, CV_64F);
    GM.at<double>(0,0) = par.CD1_1;
    GM.at<double>(0,1) = par.CD1_2;
    GM.at<double>(1,0) = par.CD2_1;
    GM.at<double>(1,1) = par.CD2_2;
    //std::cout << "M = "<< std::endl << " "  << GM << std::endl << std::endl;

    cv::Mat  inV = cv::Mat::zeros(2, 1, CV_64F);
    inV.at<double>(0,0) = u + f;
    inV.at<double>(1,0) = v + g;
    
    cv::Mat outV = cv::Mat::zeros(2, 1, CV_64F);
    outV = GM * inV;

    radec.push_back({ static_cast<float>(par.CRVAL1 + outV.at<double>(0, 0))
    , static_cast<float>(par.CRVAL2 + outV.at<double>(1, 0)), 0 });
#else

    double u = static_cast<double>(pixel.at(i)[0]) - par.CRPIX1 -1;
    double v = static_cast<double>(pixel.at(i)[1]) - par.CRPIX2 +1;

    double f = par.A_0_2 * v*v + par.A_1_1 * u*v + par.A_2_0 * u*u;
    double g = par.B_0_2 * v*v + par.B_1_1 * u*v + par.B_2_0 * u*u;

    u = u + f;
    v = v + g;

    cv::Mat GM = cv::Mat::zeros(2, 2, CV_64F);
    GM.at<double>(0,0) = par.CD1_1;
    GM.at<double>(0,1) = par.CD1_2;
    GM.at<double>(1,0) = par.CD2_1;
    GM.at<double>(1,1) = par.CD2_2;

    cv::Mat  inV = cv::Mat::zeros(2, 1, CV_64F);
    inV.at<double>(0,0) = u;
    inV.at<double>(1,0) = v;
    
    cv::Mat outV = cv::Mat::zeros(2, 1, CV_64F);
    outV = GM * inV;

    double xDeg = par.CD1_1 * u + par.CD1_2 * v;
    double yDeg = par.CD2_1 * u + par.CD2_2 * v;

    double x = -((CV_PI / 180) * xDeg);
    double y = (CV_PI / 180) * yDeg;

    double cosdec = cos((CV_PI / 180) * par.CRVAL2);
    double rx = cosdec * cos((CV_PI / 180) * par.CRVAL1);
    double ry = cosdec * sin((CV_PI / 180) * par.CRVAL1);
    double rz = sin((CV_PI / 180) * par.CRVAL2);

    double ix = ry;
    double iy = -rx;
    double norm = hypot(ix, iy);
    ix = ix / norm;
    iy = iy / norm;

    double jx = iy * rz;
    double jy = -ix * rz;
    double jz = ix * ry - iy * rx;

    double no_rm=sqrt(jx*jx+jy*jy+jz*jz);

    double xyz[3] = {0,0,0};

    xyz[0] = ix*x + jx*y + rx;
    xyz[1] = iy*x + jy*y + ry;
    xyz[2] =        jz*y + rz; // iz = 0

    double no_rm2 = sqrt(xyz[0]*xyz[0]+xyz[1]*xyz[1]+xyz[2]*xyz[2]);

    double raF = atan2(xyz[1], xyz[0]);
    if (raF < 0)
    {
      raF = raF + 2.0 * CV_PI;
    }
    
    double decF = asin(xyz[2]);

    double ra = (180 / CV_PI) * (raF);///15
    double dec = (180 / CV_PI) * decF;

    radec.push_back({ ra, dec, 0.0 });

#endif
  }
}

/* ==========================================================================
*        FUNCTION NAME: coordConvA
* FUNCTION DESCRIPTION: Coordinate Converter
*        CREATION DATE: 20160911
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
void coordConvA
(
  const wcsPar& par
  , const std::vector< cv::Vec<float, 3> >& pixel
  , std::vector< cv::Vec<double, 3> >& radec
)
{
  for (size_t i = 0; i<pixel.size(); ++i)
  {
    double u = static_cast<double>(pixel.at(i)[0]) - par.CRPIX1 -1;
    double v = static_cast<double>(pixel.at(i)[1]) - par.CRPIX2 -1;

    double f = 0;// par.A_0_2 * v*v + par.A_1_1 * u*v + par.A_2_0 * u*u;
    double g = 0;// par.B_0_2 * v*v + par.B_1_1 * u*v + par.B_2_0 * u*u;

    cv::Mat GM = cv::Mat::zeros(2, 2, CV_64F);
    GM.at<double>(0,0) = par.CD1_1;
    GM.at<double>(0,1) = par.CD1_2;
    GM.at<double>(1,0) = par.CD2_1;
    GM.at<double>(1,1) = par.CD2_2;
    //std::cout << "M = "<< std::endl << " "  << GM << std::endl << std::endl;

    cv::Mat  inV = cv::Mat::zeros(2, 1, CV_64F);
    inV.at<double>(0,0) = u + f;
    inV.at<double>(1,0) = v + g;
    
    cv::Mat outV = cv::Mat::zeros(2, 1, CV_64F);
    outV = GM * inV;

    radec.push_back({ static_cast<float>(par.CRVAL1 + outV.at<double>(0, 0))
    , static_cast<float>(par.CRVAL2 + outV.at<double>(1, 0)), 0 });

  }
}

/* ==========================================================================
*        FUNCTION NAME: coordConvB
* FUNCTION DESCRIPTION: Coordinate Converter
*        CREATION DATE: 20160911
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
void coordConvB
(
  const wcsPar& par
  , const std::vector< cv::Vec<float, 3> >& pixel
  , std::vector< cv::Vec<double, 3> >& radec
)
{
  for (size_t i = 0; i<pixel.size(); ++i)
  {
    double u = static_cast<double>(pixel.at(i)[0]) - par.CRPIX1 -1;
    double v = static_cast<double>(pixel.at(i)[1]) - par.CRPIX2 -1;

    double f = par.A_0_2 * v*v + par.A_1_1 * u*v + par.A_2_0 * u*u;
    double g = par.B_0_2 * v*v + par.B_1_1 * u*v + par.B_2_0 * u*u;

    u = u + f;
    v = v + g;

    cv::Mat GM = cv::Mat::zeros(2, 2, CV_64F);
    GM.at<double>(0,0) = par.CD1_1;
    GM.at<double>(0,1) = par.CD1_2;
    GM.at<double>(1,0) = par.CD2_1;
    GM.at<double>(1,1) = par.CD2_2;

    cv::Mat  inV = cv::Mat::zeros(2, 1, CV_64F);
    inV.at<double>(0,0) = u;
    inV.at<double>(1,0) = v;
    
    cv::Mat outV = cv::Mat::zeros(2, 1, CV_64F);
    outV = GM * inV;

    radec.push_back({ static_cast<float>(par.CRVAL1 + outV.at<double>(0, 0))
    , static_cast<float>(par.CRVAL2 + outV.at<double>(1, 0)), 0 });
  }
}

/* ==========================================================================
*        FUNCTION NAME: coordConv
* FUNCTION DESCRIPTION: Coordinate Converter
*        CREATION DATE: 20160911
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
void coordConv2
(
  const wcsPar& par
  , const std::vector< cv::Vec<float, 3> >& pixel
  , std::vector< cv::Vec<double, 3> >& radec
)
{
  for (size_t i = 0; i<pixel.size(); ++i)
  {
    /* RA */
    double ra = .0;

    {
      double u = static_cast<double>(pixel.at(i)[0]) - par.CRPIX1 - 1;
      double v = static_cast<double>(pixel.at(i)[1]) - par.CRPIX2 - 1;

      double f = par.A_0_2 * v*v + par.A_1_1 * u*v + par.A_2_0 * u*u;
      double g = par.B_0_2 * v*v + par.B_1_1 * u*v + par.B_2_0 * u*u;

      u = u + f;
      v = v + g;

      cv::Mat GM = cv::Mat::zeros(2, 2, CV_64F);
      GM.at<double>(0, 0) = par.CD1_1;
      GM.at<double>(0, 1) = par.CD1_2;
      GM.at<double>(1, 0) = par.CD2_1;
      GM.at<double>(1, 1) = par.CD2_2;

      cv::Mat  inV = cv::Mat::zeros(2, 1, CV_64F);
      inV.at<double>(0, 0) = u;
      inV.at<double>(1, 0) = v;

      cv::Mat outV = cv::Mat::zeros(2, 1, CV_64F);
      outV = GM * inV;

      double xDeg = par.CD1_1 * u + par.CD1_2 * v;
      double yDeg = par.CD2_1 * u + par.CD2_2 * v;

      double x = -((CV_PI / 180) * xDeg);
      double y = (CV_PI / 180) * yDeg;

      double cosdec = cos((CV_PI / 180) * par.CRVAL2);
      double rx = cosdec * cos((CV_PI / 180) * par.CRVAL1);
      double ry = cosdec * sin((CV_PI / 180) * par.CRVAL1);
      double rz = sin((CV_PI / 180) * par.CRVAL2);

      double ix = ry;
      double iy = -rx;
      double norm = hypot(ix, iy);
      ix = ix / norm;
      iy = iy / norm;

      double jx = iy * rz;
      double jy = -ix * rz;
      double jz = ix * ry - iy * rx;

      double no_rm = sqrt(jx*jx + jy*jy + jz*jz);

      double xyz[3] = { 0,0,0 };

      xyz[0] = ix*x + jx*y + rx;
      xyz[1] = iy*x + jy*y + ry;
      xyz[2] = jz*y + rz; // iz = 0

      double no_rm2 = sqrt(xyz[0] * xyz[0] + xyz[1] * xyz[1] + xyz[2] * xyz[2]);

      double raF = atan2(xyz[1], xyz[0]);
      if (raF < 0)
      {
        raF = raF + 2.0 * CV_PI;
      }

      double decF = asin(xyz[2]);

      ra = (180 / CV_PI) * (raF);
    }

    /* Dec */
    double dec = .0;
    {
      double u = static_cast<double>(pixel.at(i)[0]) - par.CRPIX1 + 1;
      double v = static_cast<double>(pixel.at(i)[1]) - par.CRPIX2 + 1;

      double f = par.A_0_2 * v*v + par.A_1_1 * u*v + par.A_2_0 * u*u;
      double g = par.B_0_2 * v*v + par.B_1_1 * u*v + par.B_2_0 * u*u;

      u = u + f;
      v = v + g;

      cv::Mat GM = cv::Mat::zeros(2, 2, CV_64F);
      GM.at<double>(0, 0) = par.CD1_1;
      GM.at<double>(0, 1) = par.CD1_2;
      GM.at<double>(1, 0) = par.CD2_1;
      GM.at<double>(1, 1) = par.CD2_2;

      cv::Mat  inV = cv::Mat::zeros(2, 1, CV_64F);
      inV.at<double>(0, 0) = u;
      inV.at<double>(1, 0) = v;

      cv::Mat outV = cv::Mat::zeros(2, 1, CV_64F);
      outV = GM * inV;

      double xDeg = par.CD1_1 * u + par.CD1_2 * v;
      double yDeg = par.CD2_1 * u + par.CD2_2 * v;

      double x = -((CV_PI / 180) * xDeg);
      double y = (CV_PI / 180) * yDeg;

      double cosdec = cos((CV_PI / 180) * par.CRVAL2);
      double rx = cosdec * cos((CV_PI / 180) * par.CRVAL1);
      double ry = cosdec * sin((CV_PI / 180) * par.CRVAL1);
      double rz = sin((CV_PI / 180) * par.CRVAL2);

      double ix = ry;
      double iy = -rx;
      double norm = hypot(ix, iy);
      ix = ix / norm;
      iy = iy / norm;

      double jx = iy * rz;
      double jy = -ix * rz;
      double jz = ix * ry - iy * rx;

      double no_rm = sqrt(jx*jx + jy*jy + jz*jz);

      double xyz[3] = { 0,0,0 };

      xyz[0] = ix*x + jx*y + rx;
      xyz[1] = iy*x + jy*y + ry;
      xyz[2] = jz*y + rz; // iz = 0

      double no_rm2 = sqrt(xyz[0] * xyz[0] + xyz[1] * xyz[1] + xyz[2] * xyz[2]);

      double raF = atan2(xyz[1], xyz[0]);
      if (raF < 0)
      {
        raF = raF + 2.0 * CV_PI;
      }

      double decF = asin(xyz[2]);

      dec = (180 / CV_PI) * decF;      
    }
    radec.push_back({ ra, dec, 0.0 });
  }
}

/* ==========================================================================
*        FUNCTION NAME: sigmaClipProcessing
* FUNCTION DESCRIPTION: 
*        CREATION DATE: 20160911
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
void sigmaClipProcessing
(
  const cv::Mat& histStretch
  , const cv::Mat& Img_input
  , std::ostream& infoFile
  , std::vector< cv::Vec<float, 3> >& POINTS
  , std::vector< cv::Vec<float, 3> >& STREAKS
  , std::vector<std::vector<cv::Point > >& contoursP
  , std::vector<std::vector<cv::Point > >& contoursS
)
{
  
/* ======================================================================= *
 * Points detection                                                        *
 * ======================================================================= */
  
/* ----------------------------------------------------------------------- *
 * Median filter                                                           *
 * ----------------------------------------------------------------------- */

  clock_t start = clock();
  
  int kerlenSz = 3;
  cv::Mat medianImg = medianFilter(histStretch, kerlenSz);

  timeElapsed(infoFile, start, "Median filter");


/* ----------------------------------------------------------------------- *
 * Background estimation                                                   *
 * ----------------------------------------------------------------------- */

  start = clock();

  size_t maxColdim = 512;
  size_t maxRowdim = 512;

  int regionNumR = static_cast<int>(::round(histStretch.rows / maxRowdim));
  if (0 == regionNumR) { regionNumR = 1; }
  int regionNumC = static_cast<int>(::round(histStretch.cols / maxColdim));
  if (0 == regionNumC) { regionNumC = 1; }

  cv::Point backCnt = {regionNumC, regionNumR};
  cv::Mat meanBg = cv::Mat::zeros(backCnt.y, backCnt.x, CV_64F);
  cv::Mat  stdBg = cv::Mat::zeros(backCnt.y, backCnt.x, CV_64F);

  cv::Mat backgroungImg = 
    backgroundEstimation(medianImg, backCnt, meanBg, stdBg);

  timeElapsed(infoFile, start, "Background estimation");


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
  
  cv::Mat medianBgSubImg = medianFilter(bgSubtracImg, kerlenSz);
  bgSubtracImg.release();

  timeElapsed(infoFile, start, "Median filter");
  
  
/* ----------------------------------------------------------------------- *
 * Binarization for points detection                                       *
 * ----------------------------------------------------------------------- */

  start = clock();

  cv::Mat level = cv::Mat::zeros(backCnt.y, backCnt.x, CV_64F);
  level = meanBg + 3.5*stdBg;
  
  cv::Mat binaryImgPnt = binarizationZone(medianBgSubImg, backCnt, level);
  
  timeElapsed(infoFile, start, "Binarization for points detection");
  

/* ----------------------------------------------------------------------- *
 * Binarization for streaks detection                                      *
 * ----------------------------------------------------------------------- */

  start = clock();

  cv::Mat levelStk = cv::Mat::zeros(backCnt.y, backCnt.x, CV_64F);
  levelStk = meanBg + 1*stdBg;//2.8
  
  cv::Mat binaryImgStk = binarizationZone(medianBgSubImg, backCnt, levelStk);
  medianBgSubImg.release();

  timeElapsed(infoFile, start, "Binarization for streaks detection");
    

/* ----------------------------------------------------------------------- *
 * Distance transformation for streaks detection                           *
 * ----------------------------------------------------------------------- */

  start = clock();
    
  cv::Mat distStk = distTransform(binaryImgStk);
  binaryImgStk.release();

  timeElapsed(infoFile, start, "Distance transformation");

/* ----------------------------------------------------------------------- *
 * Convolution kernel for points detection                                 *
 * ----------------------------------------------------------------------- */

  start = clock();

  int szKernel = 3;
  cv::Mat kernel = cv::Mat::ones(szKernel, szKernel, CV_8U);
  double threshConv = 7;//6
  
  cv::Mat convImgPnt = convolution(binaryImgPnt, kernel, threshConv);
  binaryImgPnt.release();
  timeElapsed(infoFile, start, "Convolution for points detection");
    

/* ----------------------------------------------------------------------- *
 * Morphology opening                                                      *
 * ----------------------------------------------------------------------- */

  start = clock();

  int radDisk = 6;  
  cv::Mat openImg = morphologyOpen(convImgPnt, radDisk);

  timeElapsed(infoFile, start, "Morphology opening");
  
  
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

  cv::Mat resizeImg;
  double f = 0.5;
  cv::Size dsize = { 0, 0 };
  resize(distStk, resizeImg, dsize, f, f, cv::INTER_LINEAR);

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

    cv::Mat morpOpLinRem = morphologyOpen(openImg, dimLineRem, angle.at(i).first);
    

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
  
  start = clock();

  cv::Mat onlyPoints = openImg - sumStrRemImg;
  sumStrRemImg.release();
  openImg.release();

#if SPD_FIGURE_1
  namedWindow("onlyPoints", cv::WINDOW_NORMAL);
  imshow("onlyPoints", onlyPoints);
  cv::waitKey(0);
#endif
  
  timeElapsed(infoFile, start, "Subtract image");

/* ----------------------------------------------------------------------- *
 * Convolution kernel remove streaks                                       *
 * ----------------------------------------------------------------------- */

  start = clock();

  cv::Mat kernelRm = cv::Mat::ones(szKernel, szKernel, CV_8U);
  double threshConvRm = 8;
  
  cv::Mat convImgRms = convolution(onlyPoints, kernelRm, threshConvRm);
  kernelRm.release();
  onlyPoints.release();  

  timeElapsed(infoFile, start, "Convolution");
    
  
/* ----------------------------------------------------------------------- *
 * Connected components                                                    *
 * ----------------------------------------------------------------------- */

  start = clock();

  connectedComponents2(convImgRms, distStk, Img_input
    , POINTS, STREAKS, contoursP, contoursS);

  distStk.release();
  convImgRms.release();

  timeElapsed(infoFile, start, "Connected components");


/* ----------------------------------------------------------------------- *
 * Light curve study                                                       *
 * ----------------------------------------------------------------------- */

  //lightCurve(Img_input, STREAKS, contoursS);//Img_input//histStretch

}

#if 0
std::future<bool> asyncAstrometry(std::string& pStr, wcsPar& par)
{
  return std::async(std::launch::async, [&]() 
  {
    bool res = false;

    //lancia script per risolvere campo stellare
    
    //verifica esistenza file .wcs
    bool existWCS = spd_os::fileExists(pStr.c_str());
    
    if (existWCS)
    {      
      parseWCS(pStr.c_str(), par);
      res = true;
    }
    
    return res;
  });
}

std::future<bool> asyncAstrometry(const std::vector<char *>& input, wcsPar& par)
{
  return std::async(std::launch::async, [&]() 
  {
    bool res = false;
        
    //Launch script for solve astrometry
#if 0
#ifdef _WIN32
    std::string astroScript = "launcherWIN.bat ";
#else
    std::string astroScript = "./astrometricReduction.sh ";
#endif
#endif
    std::string astroScript = "./astrometricReduction.sh ";

    std::string file = fileCygwin(input.at(0));
    std::string pathRes = fileCygwin(input.at(4));

    std::string command = astroScript + file + " " + pathRes;
    printf("\n%s\n", command.c_str());
    int asd = system(command.c_str());

    //Parse .wcs file
    std::string wcsF = std::string(input.at(4)) + input.at(1) + ".wcs";    

    bool existWCS = spd_os::fileExists(wcsF.c_str());
    
    if (existWCS) {      
      parseWCS(wcsF.c_str(), par);
      res = true;
    }
    
    return res;
  });
}
#else

bool astrometry(const std::vector<char *>& input, wcsPar& par)
{
  bool res = false;
  std::string astroScript = "./astrometricReduction.sh ";

  std::string command = astroScript + input.at(0) + " " + input.at(4);
  printf("\n%s\n", command.c_str());
  int asd = system(command.c_str());

  //Parse .wcs file
  std::string wcsF = std::string(input.at(4)) + input.at(1) + ".wcs";    

  bool existWCS = spd_os::fileExists(wcsF.c_str());
    
  if (existWCS) {      
    parseWCS(wcsF.c_str(), par);
    res = true;
  }
    
  return res;  
}

#endif

void lightCurve
(
  const cv::Mat& img
  , const std::vector< cv::Vec<float, 3> >& STREAKS
  , const std::vector<std::vector<cv::Point > >& contours
)
{
  std::vector< std::vector< std::vector< cv::Vec<ushort, 1> > > > buf(contours.size());
  std::vector< std::vector< std::vector< cv::Point> > > points(contours.size());

  for (size_t i = 0; i < contours.size(); ++i)
  {
    cv::RotatedRect rotRec = fitEllipse(contours[i]);

    //Bounding box
    cv::Rect rotBoundRec = rotRec.boundingRect();
    cv::Point tlBBi = rotBoundRec.tl();
    cv::Point2f tlBBf = { static_cast<float>(tlBBi.x), static_cast<float>(tlBBi.y) };

    //RoI extraction
    cv::Mat roi = img(rotBoundRec);

    //Image rotation
    cv::Mat rotRoi = imageRotation(img, contours[i]);//roi

#if SPD_FIGURE_1
    namedWindow("ROI light curve", cv::WINDOW_NORMAL);
    imshow("ROI light curve", roi);
    cv::waitKey(0);

    namedWindow("ROI roted", cv::WINDOW_NORMAL);
    imshow("ROI roted", rotRoi);
    cv::waitKey(0);
#endif

#if SPD_SAVE_FIGURE
    {
      char s_imgName[256] = "ROI";
      strcat(s_imgName, ".jpg");
      imwrite(s_imgName, roi);
    } {
      char s_imgName[256] = "ROIrot";
      strcat(s_imgName, ".jpg");
      imwrite(s_imgName, rotRoi);
    }
#endif

    //Point of the roted rectangle big image
    cv::Point2f pRotRec[4];
    rotRec.points(pRotRec);

    //Point of the roted rectangle RoI
    cv::Point2f pRotRecRoI[4];
    pRotRecRoI[0] = pRotRec[0] - tlBBf;
    pRotRecRoI[1] = pRotRec[1] - tlBBf;
    pRotRecRoI[2] = pRotRec[2] - tlBBf;
    pRotRecRoI[3] = pRotRec[3] - tlBBf;

    //Central point of the roted rectangle RoI
    cv::Point2f pMinV[4];
    pMinV[0] = {std::min(pRotRecRoI[0].x , pRotRecRoI[1].x) , std::min(pRotRecRoI[0].y , pRotRecRoI[1].y)};
    pMinV[1] = {std::min(pRotRecRoI[1].x , pRotRecRoI[2].x) , std::min(pRotRecRoI[1].y , pRotRecRoI[2].y)};
    pMinV[2] = {std::min(pRotRecRoI[2].x , pRotRecRoI[3].x) , std::min(pRotRecRoI[2].y , pRotRecRoI[3].y)};
    pMinV[3] = {std::min(pRotRecRoI[3].x , pRotRecRoI[0].x) , std::min(pRotRecRoI[3].y , pRotRecRoI[0].y)};

    cv::Point2f pCenterP[4];
    pCenterP[0] = {pMinV[0].x + std::abs(pRotRecRoI[0].x - pRotRecRoI[1].x)/2 , pMinV[0].y + std::abs(pRotRecRoI[0].y - pRotRecRoI[1].y)/2};
    pCenterP[1] = {pMinV[1].x + std::abs(pRotRecRoI[1].x - pRotRecRoI[2].x)/2 , pMinV[1].y + std::abs(pRotRecRoI[1].y - pRotRecRoI[2].y)/2};
    pCenterP[2] = {pMinV[2].x + std::abs(pRotRecRoI[2].x - pRotRecRoI[3].x)/2 , pMinV[2].y + std::abs(pRotRecRoI[2].y - pRotRecRoI[3].y)/2};
    pCenterP[3] = {pMinV[3].x + std::abs(pRotRecRoI[3].x - pRotRecRoI[0].x)/2 , pMinV[3].y + std::abs(pRotRecRoI[3].y - pRotRecRoI[0].y)/2};

    
    //Compute count cross streak
    char s_lCfileName[1024];
    ::strcpy(s_lCfileName, "lightCurve_Cross_");
    ::strcat(s_lCfileName, std::to_string(i).c_str());
    ::strcat(s_lCfileName, ".txt");
    std::ofstream lcFile(s_lCfileName);

    std::vector< cv::Vec<ushort, 1> > bufC;
    std::vector< cv::Point> pointsC;
    linePoints(roi, pCenterP[0], pCenterP[2], tlBBi, bufC, pointsC, lcFile);
    
    //Compute count along streak
    char s_lCafileName[1024];
    ::strcpy(s_lCafileName, "lightCurve_Along_");
    ::strcat(s_lCafileName, std::to_string(i).c_str());
    ::strcat(s_lCafileName, ".txt");
    std::ofstream lcaFile(s_lCafileName);

    std::vector< cv::Vec<ushort, 1> > bufL;
    std::vector< cv::Point> pointsL;
    linePoints(roi, pCenterP[3], pCenterP[1], tlBBi, bufL, pointsL, lcaFile);

    buf[i].push_back(bufC);
    buf[i].push_back(bufL);

    points[i].push_back(pointsC);
    points[i].push_back(pointsL);        
  }
}

void linePoints
(
  const cv::Mat& img
  , const cv::Point2f & p1
  , const cv::Point2f & p2
  , const cv::Point tl
  , std::vector< cv::Vec<ushort,1> >& buf
  , std::vector< cv::Point>& points
  , std::ostream& stream
)
{
  int connectivity = 8;
  bool leftToRight = true;

  cv::LineIterator it{ img, p1, p2, connectivity, leftToRight };

  for (int x = 0; x < it.count; ++x)
  {
    points.push_back({ it.pos().x + tl.x , it.pos().y + tl.y });

    const ushort* pLine = img.ptr<ushort>(it.pos().y);
    buf.push_back({ pLine[it.pos().x] });

    it++;
    stream << buf[x] << " " << points[x].x << " " << points[x].y << std::endl;    
  }
}

cv::Mat imageRotation(const cv::Mat img, const std::vector<cv::Point >& contours)
{
  cv::Mat imgOut;
    
  //Centroid and angle  
  cv::RotatedRect rotRec = cv::fitEllipse(contours);

  //Bounding box
  cv::Rect rotBoundRec = rotRec.boundingRect();
  cv::Point tlBBi = rotBoundRec.tl();
  cv::Point2f tlBBf = { static_cast<float>(tlBBi.x), static_cast<float>(tlBBi.y) };

  //RoI extraction
  cv::Mat imgIn = img(rotBoundRec);
  

  cv::Point2f centroid = rotRec.center;
  float angle = -1.0f * (90.0f - rotRec.angle);
  
  cv::Point2f pCenterP[4];
  pCenterP[0] = {                     centroid.x - tlBBf.x ,                              0 - tlBBf.y};
  pCenterP[1] = {                              0 - tlBBf.x ,                     centroid.y - tlBBf.y};
  pCenterP[2] = {                     centroid.x - tlBBf.x , static_cast<float>(imgIn.rows) - tlBBf.y};
  pCenterP[3] = { static_cast<float>(imgIn.cols) - tlBBf.x ,                     centroid.y - tlBBf.y};


  // Get the rotation matrix
  cv::Mat rot_mat( 2, 3, CV_32FC1 );
    
  rot_mat = getRotationMatrix2D( centroid, angle, 1 );

  // Rotate the warped image
  warpAffine( imgIn, imgOut, rot_mat, imgIn.size(), cv::INTER_LINEAR );

  //Light curve

  //Compute count along streak
  char s_lCafileName[1024];
  ::strcpy(s_lCafileName, "lightCurveRot_Along_");
  ::strcat(s_lCafileName, std::to_string(1).c_str());
  ::strcat(s_lCafileName, ".txt");
  std::ofstream lcaFile(s_lCafileName);

  std::vector< cv::Vec<ushort, 1> > bufL;
  std::vector< cv::Point> pointsL;

  linePoints(imgOut, pCenterP[3], pCenterP[1], tlBBi, bufL, pointsL, lcaFile);

  return imgOut;
}

/* ==========================================================================
*        FUNCTION NAME: preCompression
* FUNCTION DESCRIPTION: Original image with only streaks and points
*        CREATION DATE: 20160911
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
void preCompression(
  const cv::Mat imgIn
  , std::vector<std::vector<cv::Point > >& contoursS
  , std::vector<std::vector<cv::Point > >& contoursP
  , const std::vector<char *>& input
)
{
  cv::Mat imgOut = cv::Mat::zeros(imgIn.rows, imgIn.cols, imgIn.type());

  /* Select RoIs with Streaks */
  for (size_t i=0; i<contoursS.size(); ++i)
  {
    cv::Rect bbox = boundingRect(contoursS.at(i));
    cv::Mat imgPart = imgIn(bbox);
    imgPart.copyTo(imgOut(bbox));
  }


  /* Select RoIs with Points */
  for (size_t i=0; i<contoursP.size(); ++i)
  {
    cv::Rect bbox = boundingRect(contoursP.at(i));
    cv::Mat imgPart = imgIn(bbox);
    imgPart.copyTo(imgOut(bbox));
  }

  #if SPD_FIGURE
  namedWindow("Image for compression purpose", cv::WINDOW_NORMAL);
  imshow("Image for compression purpose", imgOut);
#endif

#if SPD_SAVE_FIGURE
  char s_imgName[256];
  ::strcpy(s_imgName, input.at(4));
  ::strcat(s_imgName, input.at(1));
  ::strcat(s_imgName, "_preCompr");
  ::strcat(s_imgName, ".jpg");
  cv::imwrite(s_imgName, imgOut);
#endif

}
