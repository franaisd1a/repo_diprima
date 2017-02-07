/* ==========================================================================
* PK SPACEKIT LIBRARY
* (C) 2016 PLANETEK Italia SRL
* ========================================================================== */

/* ==========================================================================
*   INCLUDE FILE NAME: function_os_posix.h
* INCLUDE DESCRIPTION: Function posix for manage file system.
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

#ifndef FUNCTION_OS_POSIX_H
#define FUNCTION_OS_POSIX_H

/* ==========================================================================
* INCLUDE: Basic include file.
* ========================================================================== */
//#include <windows.h>

#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#include <dirent.h>
#include <iostream>


/*#include <stdlib.h>
#include <stdint.h>
#include <iostream>
#include <cstring>
#include <vector>

#include <io.h>
#include <fcntl.h>*/

/* ==========================================================================
* MACROS
* ========================================================================== */

/* ==========================================================================
* CLASS DECLARATION
* ========================================================================== */

/* ==========================================================================
* FUNCTION DECLARATION
* ========================================================================== */
namespace spd_os
{
  /**
   * Information about a directory item.
   */
  struct DirectoryItem
  {
    char strName[1024];       /**< Name of the item */
    size_t szSizeInBytes;     /**< Size in bytes */
    bool bIsRegularFile;      /**< Flag: is a regular file */
    bool bIsDirectory;        /**< Flag: is a directory */
  };

  constexpr static uint32_t
    MakeU32LE(uint8_t x0, uint8_t x1, uint8_t x2, uint8_t x3)
  {
    return ((x3 << 24) | (x2 << 16) | (x1 << 8) | x0);
  }

  bool createDirectory(const char* strName);
  bool removeEmptyDirectory(const char* strName);
  bool removeFile(const char* strName);
  bool fileExists(const char* pStrName);
  bool directoryExists(const char* strName);
  bool directoryOpen(void* dhOut, const char* pStrName);
  bool directoryNextItem(void* dhIn, DirectoryItem& out);
  bool directoryClose(void* dhIn);
  bool scan(void* hdir, char * nameFile);
  /**
  * fileExt Get file extension
  * @param nameFile Input file name
  * @return File extension
  */
  std::vector<char*> fileExt(const char* strN);
}

#endif /* FUNCTION_OS_POSIX_H */
