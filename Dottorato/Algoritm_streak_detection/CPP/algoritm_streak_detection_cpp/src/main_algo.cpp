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
#include "main_algo.h"


#include <windows.h>

#include <time.h>

//#include "main_GPU_cuda.cuh"
#include "main_simple.h"
//#include "main_GPU.h"
#include "macros.h"
//#include "main_2.h"
#include "main_fits.h"

#ifdef WIN32
#include "function_os_win.h"
#else
//gestione file system di linux
#endif

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
*        FUNCTION NAME: main_simple
* FUNCTION DESCRIPTION:
*        CREATION DATE: 20160727
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
bool main_algo(char* input, bool folderMod)
{
  bool outputRes = true;

/* ----------------------------------------------------------------------- *
 * Open and read file                                                      *
 * ----------------------------------------------------------------------- */

  /*
   * folderMod = true folder path
   * folderMod = false  file path
   */
  
  const char* extjpg = "jpg";
  const char* extJPG = "JPG";
  const char* extfit = "fit";
  const char* extFIT = "FIT";
  char nameFile[1024];
  //::memset(nameFile,0,sizeof(nameFile));

  void* hdir = ::malloc(512);
  bool exitLoop = false;

  if (folderMod) {
    bool expOpen = spd_os::directoryOpen(hdir,input);
  }

  while (!exitLoop)
  {
    if (folderMod)
    {
      char file[1024];
      ::memset(file,0,sizeof(file));
      ::memset(nameFile,0,sizeof(nameFile));

      exitLoop = spd_os::scan(hdir, file);

      strcpy(nameFile, input);
      
      char lastC = nameFile[strlen(nameFile) - 1];
      char slash = 92;

      if (lastC!=slash) {
        nameFile[strlen(nameFile)] = slash;
      }

      strcat(nameFile, file);
    }
    else
    {
      ::memset(nameFile,0,sizeof(nameFile));

      std::vector<char*> vec = fileExt(input);
      char* ext = vec.at(1);

      if ((0 == strcmp(ext, extJPG)) || (0 == strcmp(ext, extjpg))
        || (0 == strcmp(ext, extFIT)) || (0 == strcmp(ext, extfit)))
      {
        strncpy(nameFile, input, strlen(input));
      }            
      exitLoop = true;
    }    

    if ((strlen(nameFile) > strlen(input)+1) || !folderMod)
    {
      //chiamata agli algoritmi
      std::cout << "File " << nameFile << std::endl;
    }

    if (!exitLoop) { continue; }
  }

  if (folderMod)
  {
    bool expClose = spd_os::directoryClose(hdir);
  }

  std::cout << "End." << std::endl;




  /*****************************************************************************/
#if 0


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
#if 0
    int testFits = main_fits(name_file);
#endif
  }

/* -------------------------------------------------------------------------- */

  if (repeatCycle > 1)
  {
    //std::cout << "algoSimple: " << totalTime << " AlgoCUDAkernel: "<< totalTimeCUDAkernel << std::endl;
    std::cout << "End " << std::endl;
  }
#endif
  return outputRes;
}
