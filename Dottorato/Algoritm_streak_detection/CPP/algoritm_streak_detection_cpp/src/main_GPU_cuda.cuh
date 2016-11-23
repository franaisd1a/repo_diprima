/* ==========================================================================
* PK SPACEKIT LIBRARY
* (C) 2016 PLANETEK Italia SRL
* ========================================================================== */

/* ==========================================================================
*   INCLUDE FILE NAME: main_GPU_cuda.h
* INCLUDE DESCRIPTION: Algo simple for streaks and points detection on GPU
*       CREATION DATE: 20160727
*             AUTHORS: Francesco Diprima
*        DESIGN ISSUE: None.
*
*             HISTORY: See table below.
*
* 27-Jul-2016 | Francesco Diprima | 0.0 |
* Initial creation of this file.
*
* ========================================================================== */

#ifndef MAIN_GPU_CUDA_H
#define MAIN_GPU_CUDA_H

/* ==========================================================================
* INCLUDE
* ========================================================================== */
#include "function_GPU.h"

/* ==========================================================================
* MACROS
* ========================================================================== */

/* ==========================================================================
* CLASS DECLARATION
* ========================================================================== */

/* ==========================================================================
* FUNCTION DECLARATION
* ========================================================================== */
 
/**
* main_GPU_cuda Algo for streaks and points detection on GPU with CUDA kernel
* @param imgIn Input image
* @return 
*/
int main_GPU_cuda(char* name_file);

#endif /* MAIN_GPU_CUDA_H */

