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
#include <time.h>
#include <string.h>
#include <stdlib.h>
#include <fitsio.h>

#include <opencv2/opencv.hpp>
#include <opencv/highgui.h>

/* ==========================================================================
* MODULE PRIVATE MACROS
* ========================================================================== */
#define NBIT 8U

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
*        FUNCTION NAME: main_fits
* FUNCTION DESCRIPTION:
*        CREATION DATE: 20160727
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
int main_fits(char* name_file)
{
//int main(int argc, char** argv){
  
  // Check for invalid input

  fitsfile *fptr;
  char card[FLEN_CARD];
  int status = 0, nkeys, ii; /* MUST initialize status */
  //fits_open_file(&fptr, argv[1], READONLY, &status);
  fits_open_file(&fptr, name_file, READONLY, &status);
  fits_get_hdrspace(fptr, &nkeys, NULL, &status);
  for (ii = 1; ii <= nkeys; ii++) {
    fits_read_record(fptr, ii, card, &status); /* read keyword */
    printf("%s\n", card);
  }

#if 0
  int bitpix = 0;
  int status0 = 0;
  int imgType = fits_get_img_type(fptr, &bitpix, &status0);
  //        int fits_get_img_type(fitsfile *fptr, int *bitpix, int *status)

  

  int maxdim= 2;
  long naxesA[2] = { 0, 0 };
  long naxesB= 0;
  int status2= 0;
  int imgSz = fits_get_img_size(fptr, maxdim, naxesA, &status2);
  //      int fits_get_img_size(fitsfile *fptr, int maxdim, long *naxes, int *status)
#endif

  int naxis = 0;
  int status1 = 0;
  int imgDim = fits_get_img_dim(fptr, &naxis, &status1);
  
  if (2 != naxis)
  {
    printf("Error! 3d image.\n");
  }

  int maxdim= 2;
  int bitpix= 0;
  int naxis2 = 0;
  long naxes[2] = { 0, 0 };
  int statusImg= 0;
  int imgParam = fits_get_img_param(fptr, maxdim, &bitpix, &naxis2, naxes, &statusImg);
  

  /* Read image */
  
  int statusRead = 0;
  void *nulval = NULL;//0;//
  long fpixel[2] = {1, 1};
  long nelements = naxes[0] * naxes[1];
  int anynul = 0;

#if 0
  int datatype;
  void* array = nullptr;
  if (2 == bitpix/NBIT)
  {
    datatype = TUSHORT;
    array = (unsigned short*)malloc(sizeof(unsigned short) * nelements);
  }
  else if (4 == bitpix / NBIT)
  {
    datatype = TULONG;
    array = (unsigned long*)malloc(sizeof(unsigned long) * nelements);
  }
  else {
    printf("Error! Unsupported pixel type.\n");
  }
#endif

  int datatype = TUSHORT;
  unsigned short* array = (unsigned short*)malloc(sizeof(unsigned short) * nelements);
  int readImg = fits_read_pix(fptr, datatype, fpixel, nelements, nulval, array, &anynul, &status);

#if 0
  /*TBYTE     unsigned char
    TSBYTE    signed char
    TSHORT    signed short
                                      TUSHORT   unsigned short
    TINT      signed int
                                      TUINT     unsigned int
    TLONG     signed long
    TLONGLONG signed 8-byte integer
                                      TULONG    unsigned long
                                      TFLOAT    float
                                      TDOUBLE   double
                                      */


  printf(" unsigned char = %d\n", sizeof(unsigned char));
  printf("   signed char = %d\n", sizeof(signed char));
  printf("  signed short = %d\n", sizeof(signed short));
  printf("unsigned short = %d\n", sizeof(unsigned short));
  printf("    signed int = %d\n", sizeof(signed int));
  printf("  unsigned int = %d\n", sizeof(unsigned int));
  printf("   signed long = %d\n", sizeof(signed long));
  printf("        signed = %d\n", sizeof(signed));
  printf(" unsigned long = %d\n", sizeof(unsigned long));
  printf("         float = %d\n", sizeof(float));
  printf("        double = %d\n", sizeof(double));
#endif
  
#if 0
  for (size_t i = 0; i < nelements; ++i)
  {    
#if 0    
    if (2 == bitpix / NBIT) {
      unsigned short* ptrArray = (unsigned short*)array;
      printf("%d ", *ptrArray);
      ptrArray++;
    }
    else if (4 == bitpix / NBIT) {
      unsigned long* ptrArray = (unsigned long*)array;
      printf("%d ", *ptrArray);
      ptrArray++;
    }
#endif
   
    printf("%d ", *array);
    array++;
  }
#endif

  cv::Mat img = cv::Mat(naxes[1], naxes[0], CV_16U, array);
  // Create a window for display.
  namedWindow("Gaussain filter", cv::WINDOW_NORMAL);
  imshow("Gaussain filter", img);

  printf("END\n\n"); /* terminate listing with END */
  fits_close_file(fptr, &status);
  if (status) /* print any error messages */
    fits_report_error(stderr, status);

  cv::Point p{0,0};

  for (size_t i = 0; i < naxes[1]; ++i)
  {
    for (size_t j = 0; j < naxes[0]; ++j)
    {
      p.x= j;
      p.y= i;
      unsigned short val = img.at<unsigned short>(p);
      printf("%d ", val);
    }
  }

  cv::waitKey(0);
  
  return 1;  
}
 