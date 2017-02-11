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
#include <stdio.h>
#include <string.h>
#include <iostream>
#include "file_selection.h"

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
  if (argc < 3)  {
    std::cout << "Error: insert input argument." << std::endl;
    return -1;
  }

  char* folder = "-D";
  char* file = "-F";
  bool folderMod = false;
 
  if (0 == ::strcmp(folder, argv[1])) {
    folderMod = true;
  }
  else if (0 == ::strcmp(file, argv[1])) {

  }
  else {
    std::cout << "Error. Select the modality:" << std::endl; 
    std::cout << "-F for single file or -D for folder." << std::endl;
    return -1;
  }

  std::cout << "Start streaks points detection algorithm" << std::endl;

  bool res = file_selection(argv[2], folderMod);

  std::cout << "End streaks points detection algorithm" << std::endl;

  return 1;
}
