/* ==========================================================================
* ALGO STREAK DETECTION
* ========================================================================== */

/* ==========================================================================
* MODULE FILE NAME: algo_selection.cpp
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
#include "algo_selection.h"
#include "macros.h"
#include <time.h>

#include "main_simple.h"
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
bool algo_selection(char* nameFile)
{
  bool outputRes = true;
  
  clock_t start, stop;
  double totalTime, totalTimeCUDAkernel;

  std::cout << "Start streaks points detection algorithms" << std::endl;

  int repeatCycle = 1;

  for (int u = 0; u < repeatCycle; ++u)
  {
/* ------------------------------- AlgoSimple ------------------------------- */
#if 1
    start = clock();

    // Algo simple

    int algoSimple = main_simple(nameFile);


    stop = clock();
    totalTime = (stop - start) / static_cast<double>(CLOCKS_PER_SEC);

    //std::cout << "algoSimple time: " << totalTime << std::endl;
    std::cout << "CPU time: " << totalTime << " sec" << std::endl;
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
  }

/* -------------------------------------------------------------------------- */

  if (repeatCycle > 1)
  {
    //std::cout << "algoSimple: " << totalTime << " AlgoCUDAkernel: "<< totalTimeCUDAkernel << std::endl;
    std::cout << "End algo." << std::endl;
  }

  return outputRes;
}
