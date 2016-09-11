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
  //int algoSimple = main_simple(name_file);

  // Algo 2
  int algo2 = main_2(name_file);

  return 0;
}
