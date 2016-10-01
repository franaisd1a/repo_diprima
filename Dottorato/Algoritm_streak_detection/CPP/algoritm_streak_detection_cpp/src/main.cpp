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
    std::cout << "Error: insert input argument." << std::endl;
    return -1;
  }

  // File name
  //char* name_file = "C:\\Users\\diprima\\Desktop\\scontoMOTO.PNG";
  char* name_file = argv[1];

  clock_t start, stop;
	double totalTime;

  std::cout << "Start streaks points detection algorithms." << std::endl;

/* ------------------------------- AlgoSimple ------------------------------- */

  start = clock();
	
  // Algo simple
  int algoSimple = 0;
#if 1
  algoSimple = main_simple(name_file);
#endif

  stop = clock();
	totalTime = (stop - start) / static_cast<double>(CLOCKS_PER_SEC);
  
  std::cout << "algoSimple time: " << totalTime << std::endl;

/* --------------------------------- Algo2 ---------------------------------- */

  start = clock();

  // Algo 2
  int algo2 = 0;
#if 0  
  algo2 = main_2(name_file);
#endif

  stop = clock();
	totalTime = (stop - start) / static_cast<double>(CLOCKS_PER_SEC);

  std::cout << "algo2 time: " << totalTime << std::endl;

/* -------------------------------- AlgoGPU --------------------------------- */
  
  start = clock();
  
  // Algo GPU

  int algoGPU = 0;
#if 0
  algoGPU = main_GPU(name_file);
#endif

  stop = clock();
	totalTime = (stop - start) / static_cast<double>(CLOCKS_PER_SEC);

  std::cout << "algoGPU time: " << totalTime << std::endl;

  // Test fits
#if 0
  int testFits = main_fits(name_file);
#endif

  cv::waitKey(0);

  return algoSimple+algo2+algoGPU;

}
