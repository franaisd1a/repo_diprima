/* ==========================================================================
* PK SPACEKIT LIBRARY
* (C) 2016 PLANETEK Italia SRL
* ========================================================================== */

/* ==========================================================================
*   INCLUDE FILE NAME: macros.h
* INCLUDE DESCRIPTION: Macros.
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

#ifndef MACRO_H
#define MACRO_H
 
/* ==========================================================================
* MACROS
* ========================================================================== */

#define SPD_DEBUG                  0U

//SPD_FOLDER Abilita la ricerca dei file da file system
#define SPD_FOLDER                 1U

//SPD_SAVE_FIGURE Abilita il salvataggio delle immagini su disco "nomeFile.jpg"
#define SPD_SAVE_FIGURE            1U
//SPD_FIGURE Abilita l'apertura delle immagini significative a video
#define SPD_FIGURE                 1U
//SPD_FIGURE_1 Abilita l'apertura delle immagini di debug a video
#define SPD_FIGURE_1               0U
        
//SPD_STAMP Abilita le stampe
#define SPD_STAMP                  1U
//SPD_STAMP_FILE_RESULT Abilita la stampa dei risultati su disco "nomeFile.txt"
#define SPD_STAMP_FILE_RESULT      1U
//SPD_STAMP_FILE_INFO Abilita la stampa di debug e di info su disco "nomeFile_info.txt"
#define SPD_STAMP_FILE_INFO        1U
//SPD_STAMP_CONSOLE Abilita la stampa su console
#define SPD_STAMP_CONSOLE          1U
//SPD_STAMP_CONSOLE Abilita la stampa dell'header FIT su console
#define SPD_STAMP_FIT_HEADER       1U
        
        
#define SPD_CLEAR                  0U
#define SPD_BACKGROUND_SUBTRACTION 1U
#define SPD_DIFFERENT_THRESHOLD    1U
#define SPD_DILATE                 1U
 
#endif /* MACRO_H */
