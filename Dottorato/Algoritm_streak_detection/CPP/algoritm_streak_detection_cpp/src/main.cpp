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
#include "main_2.h"
#include "main_simple.h"
#include "main_GPU.h"
#include "main_fits.h"

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

  // Algo simple
#if 0
  int algoSimple = main_simple(name_file);
#endif

  // Algo 2
#if 0  
  int algo2 = main_2(name_file);
#endif

  // Algo GPU
#if 0
  int algo3 = main_GPU(name_file);
#endif

  // Test fits
#if 1
  int testFits = main_fits(name_file);
#endif

  return 0;
}
