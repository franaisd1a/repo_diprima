/* ==========================================================================
* ALGO STREAK DETECTION
* ========================================================================== */

/* ==========================================================================
* MODULE FILE NAME: algo_selection_GPU.cpp
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
#include <iostream>

#include "algo_selection_GPU.h"
#include "macros.h"

#include "main_sigmaClipSimpleBig_GPU.cuh"

//#include "main_GPU_cuda.cuh"
//#include "main_GPU.h"


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
using namespace std;

/* ==========================================================================
*        FUNCTION NAME: algo_selection
* FUNCTION DESCRIPTION:
*        CREATION DATE: 20160727
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
bool algo_selection(const std::vector<char *>& input)
{
  bool outputRes = true;
  
  if (5 != input.size())
  {
    printf("Error in input parameters.");
  }

  std::cout << "Start GPU processing: " << input.at(1) << "." << input.at(2) << std::endl;

#if SPD_DEBUG
  std::cout <<      "nameFile " << input.at(0) << std::endl;
  std::cout <<     "onlyNameF " << input.at(1) << std::endl;
  std::cout <<       "fileExt " << input.at(2) << std::endl;
  std::cout <<      "namePath " << input.at(3) << std::endl;
  std::cout << "nameResFolder " << input.at(4) << std::endl;
#endif

  clock_t start, stop;
  double totalTime, totalTimeCUDAkernel;

  int repeatCycle = 10;

  for (int u = 0; u < repeatCycle; ++u)
  {
/* --------------------- AlgoSigmaClippingSimpleBig_GPU --------------------- */
#if 1
    start = clock();

    // Algo simple

    int sigmaClip = main_sigmaClipSimpleBig_GPU(input);


    stop = clock();
    totalTime = (stop - start) / static_cast<double>(CLOCKS_PER_SEC);

    std::cout << "GPU time: " << totalTime << " sec" << std::endl;
#endif
/* ----------------------------- AlgoCUDAkernel ----------------------------- */
#if 0  
    start = clock();

    // AlgoCUDAkernel

    int AlgoCUDAkernel = main_GPU_cuda(name_file);


    stop = clock();
    totalTimeCUDAkernel = (stop - start) / static_cast<double>(CLOCKS_PER_SEC);

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
  }

/* -------------------------------------------------------------------------- */

  if (repeatCycle > 1)
  {
    //std::cout << "algoSimple: " << totalTime << " AlgoCUDAkernel: "<< totalTimeCUDAkernel << std::endl;
    std::cout << "End GPU algo." << std::endl;
  }

  return outputRes;
}
