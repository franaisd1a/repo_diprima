/* ==========================================================================
* PK SPACEKIT LIBRARY
* (C) 2016 PLANETEK Italia SRL
* ========================================================================== */

/* ==========================================================================
*   INCLUDE FILE NAME: main_sigmaClipSimpleBig_GPU.cuh
* INCLUDE DESCRIPTION: Algo simple for streaks and points detection
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

#ifndef MAIN_SIGMA_CLIP_SIMPLE_BIG_GPU_CUH
#define MAIN_SIGMA_CLIP_SIMPLE_BIG_GPU_CUH

/* ==========================================================================
* INCLUDE
* ========================================================================== */

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
* main_sigmaClipSimpleBig_GPU Algo simple for streaks and points detection
* @param imgIn Input image
* @return 
*/
int main_sigmaClipSimpleBig_GPU(const std::vector<char *>& input);

#endif /* MAIN_SIGMA_CLIP_SIMPLE_BIG_GPU_CUH */
