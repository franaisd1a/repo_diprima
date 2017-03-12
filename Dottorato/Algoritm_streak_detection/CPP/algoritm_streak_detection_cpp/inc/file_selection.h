/* ==========================================================================
* PK SPACEKIT LIBRARY
* (C) 2016 PLANETEK Italia SRL
* ========================================================================== */

/* ==========================================================================
*   INCLUDE FILE NAME: file_selection.h
* INCLUDE DESCRIPTION: File for selection of file to be process
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

#ifndef FILE_SELECTION_H
#define FILE_SELECTION_H

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
* file_selection File for selection of file to be process
* @param imgIn Input image
* @return 
*/
bool file_selection(char* input, bool folderMod);

#endif /* FILE_SELECTION_H */