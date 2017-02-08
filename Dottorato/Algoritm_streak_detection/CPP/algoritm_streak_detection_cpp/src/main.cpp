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

  bool res = file_selection(argv[2], folderMod);

  std::cout << "End " << std::endl;

#if 0
  clock_t start, stop;
  double totalTime, totalTimeCUDAkernel;

  std::cout << "Start streaks points detection algorithms" << std::endl;

  int repeatCycle = 1;

for (int u=0; u<repeatCycle;++u)
{
/* ------------------------------- AlgoSimple ------------------------------- */
#if 1
  start = clock();
  
  // Algo simple

  int algoSimple = main_simple(name_file);


  stop = clock();
  totalTime = (stop - start) / static_cast<double>(CLOCKS_PER_SEC);
  
  //std::cout << "algoSimple time: " << totalTime << std::endl;
  std::cout << "CPU time: " << totalTime << " sec" << std::endl;
#endif
/* --------------------------------- Algo2 ---------------------------------- */
#if 0  
  start = clock();

  // Algo 2

  int algo2 = main_2(name_file);

  stop = clock();
  totalTime = (stop - start) / static_cast<double>(CLOCKS_PER_SEC);

  std::cout << "algo2 time: " << totalTime << std::endl;
#endif
/* ----------------------------- AlgoCUDAkernel ----------------------------- */
#if 0  
  start = clock();

  // AlgoCUDAkernel

  int AlgoCUDAkernel = main_GPU_cuda(name_file);


  stop = clock();
  totalTimeCUDAkernel = (stop - start) / static_cast<double>(CLOCKS_PER_SEC);

  //std::cout << "AlgoCUDAkernel time: " << totalTimeCUDAkernel << std::endl;
  std::cout << "GPU time: " << totalTimeCUDAkernel << " sec" << std::endl;
#endif
/* -------------------------------- AlgoGPU --------------------------------- */
#if 0
  start = clock();
  
  // Algo GPU


  int algoGPU = main_GPU(name_file);


  stop = clock();
  totalTime = (stop - start) / static_cast<double>(CLOCKS_PER_SEC);

  std::cout << "AlgoGPU time: " << totalTime << std::endl;
#endif

/* ------------------------------- TestFITS --------------------------------- */
  // Test fits
#if 0
  int testFits = main_fits(name_file);
#endif
}


/* -------------------------------------------------------------------------- */

if (repeatCycle>1)
{
  //std::cout << "algoSimple: " << totalTime << " AlgoCUDAkernel: "<< totalTimeCUDAkernel << std::endl;
  std::cout << "End " << std::endl;
}
  cv::waitKey(0);
#endif

  return 1;
}
