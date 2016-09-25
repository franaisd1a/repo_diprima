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
#include <cmath>
#include <iostream>
#include <fstream>
#include <memory>

#include "function.h"

//#include <CCfits>
#include <CCfits/CCfits>
#include <CCfits/FITS.h>
#include <CCfits/FITSBase.h>

#include "fitsio.h"

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
using namespace CCfits;

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
  cout << "file name: " << name_file << std::endl;

#if 0
  fitsfile *fptr;
          char card[FLEN_CARD];
  int status = 0,  nkeys, ii;  /* MUST initialize status */

  fits_open_file(&fptr, name_file, READONLY, &status);
  fits_get_hdrspace(fptr, &nkeys, NULL, &status);

  for (ii = 1; ii <= nkeys; ii++)  {
	fits_read_record(fptr, ii, card, &status); /* read keyword */
	printf("%s\n", card);
  }
  printf("END\n\n");  /* terminate listing with END */
  fits_close_file(fptr, &status);

  if (status)          /* print any error messages */
  fits_report_error(stderr, status);
  //return(status);

#endif




  // Reading the data file
  //std::auto_ptr<FITS> pInfile(new FITS(name_file, Read));
  //std::auto_ptr<FITS> pInfile(new FITS(name_file,Read,true));

#if 0
  std::auto_ptr<FITS> pInfile(new FITS(name_file,Read,true));

  PHDU& image = pInfile->pHDU();
  std::valarray<unsigned long> contents;

  // read all user-specifed, coordinate, and checksum keys in the image
  image.readAllKeys();
  image.read(contents);
  // this doesn’t print the data, just header info.
  std::cout << image << std::endl;
  long ax1(image.axis(0));
  long ax2(image.axis(1));
  for (long j = 0; j < ax2; j+=10)
  {
  std::ostream_iterator<short> c(std::cout,"\t");
  std::copy(&contents[j*ax1],&contents[(j+1)*ax1-1],c);
  std::cout << std::endl;
  }
  std::cout << std::endl;
#endif



  /*/ Read file
  Mat Img_input = imread(name_file, CV_LOAD_IMAGE_GRAYSCALE );
    
  // Check for invalid file
  if (!Img_input.data)  {
    cout << "Error: could not open or find the image." << std::endl;
    return -1;
  }
*/


  if (FIGURE)
  {
    /*/ Create a window for display.
    namedWindow("Display window", WINDOW_NORMAL);
    imshow("Display window", color_Img_input);
    */
  }

  waitKey(0);
  return 0;
}
