/* ==========================================================================
* PK SPACEKIT LIBRARY
* (C) 2016 PLANETEK Italia SRL
* ========================================================================== */

/* ==========================================================================
*   INCLUDE FILE NAME: main_GPU.h
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

#ifndef MAIN_GPU_H
#define MAIN_GPU_H

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
* main_GPU Algo simple for streaks and points detection on GPU
* @param imgIn Input image
* @return 
*/
int main_median(char* name_file);

#endif /* MAIN_GPU_H */

