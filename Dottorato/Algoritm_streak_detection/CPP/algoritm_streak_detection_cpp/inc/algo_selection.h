/* ==========================================================================
* PK SPACEKIT LIBRARY
* (C) 2016 PLANETEK Italia SRL
* ========================================================================== */

/* ==========================================================================
*   INCLUDE FILE NAME: algo_selection.h
* INCLUDE DESCRIPTION: Selection of the algorithm
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

#ifndef ALGO_SELECTION_H
#define ALGO_SELECTION_H

/* ==========================================================================
* INCLUDE
* ========================================================================== */
#include <vector>
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
* algo_selection Selection of the algorithm
* @param imgIn Input image
* @return 
*/
bool algo_selection(const std::vector<char *>& input);

#endif /* ALGO_SELECTION_H */