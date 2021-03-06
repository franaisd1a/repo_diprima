/* ==========================================================================
* ALGO STREAK DETECTION
* ========================================================================== */

/* ==========================================================================
* MODULE FILE NAME: function_os_win.cpp
*      MODULE TYPE: 
*
*         FUNCTION: Function win for manage file system.
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
#include "function_os_win.h"

/* ==========================================================================
* MODULE PRIVATE MACROS
* ========================================================================== */

/* ==========================================================================
* MODULE PRIVATE TYPE DECLARATIONS
* ========================================================================== */
struct PrivTDir
{
  uint32_t uMarker;
  HANDLE hFind;
  WIN32_FIND_DATA ffd;
  bool bIsFirst;
};

/* ==========================================================================
* STATIC VARIABLES FOR MODULE
* ========================================================================== */

/* ==========================================================================
* STATIC MEMBERS
* ========================================================================== */

/* ==========================================================================
* NAME SPACE
* ========================================================================== */

/* ==========================================================================
 *        FUNCTION NAME: createDirectory
 * FUNCTION DESCRIPTION: create a directory
 *        CREATION DATE: 20170201
 *              AUTHORS: Francesco Diprima
 *           INTERFACES: None
 *         SUBORDINATES: None
 * ========================================================================== */
bool spd_os::createDirectory(const char* pStrName)
{
  bool res = true;
  if (nullptr == pStrName) {
    printf("NULL required parameter\n");
    res = false;
  }
  if (0 == CreateDirectory(pStrName, NULL)){
    printf("Can't create directory\n");
    res = false;
  }
  return res;
}

/* ==========================================================================
 *        FUNCTION NAME: removeEmptyDirectory
 * FUNCTION DESCRIPTION: remove Empty directory
 *        CREATION DATE: 20170201
 *              AUTHORS: Francesco Diprima
 *           INTERFACES: None
 *         SUBORDINATES: None
 * ========================================================================== */
bool spd_os::removeEmptyDirectory(const char* pStrName)
{
  bool res = true;
  if (nullptr == pStrName) {
    printf("NULL required parameter\n");
    res = false;
  }    
  if (0 != RemoveDirectory(pStrName)) {
    printf("Could not remove directory\n");
    res = false;
  }  
  return res;
}

/* ==========================================================================
 *        FUNCTION NAME: removeFile
 * FUNCTION DESCRIPTION: remove a file
 *        CREATION DATE: 20170201
 *              AUTHORS: Francesco Diprima
 *           INTERFACES: None
 *         SUBORDINATES: None
 * ========================================================================== */
bool spd_os::removeFile(const char* pStrName)
{
  bool res = true;
  if (nullptr == pStrName) {
    printf("NULL required parameter\n");
    res = false;
  }    
  if(0 == ::DeleteFile(pStrName)){
    printf("Could not remove file\n");
    res = false;
  }  
  return res;
}

/* ==========================================================================
 *        FUNCTION NAME: fileExists
 * FUNCTION DESCRIPTION: check whether specified file exists
 *        CREATION DATE: 20170201
 *              AUTHORS: Francesco Diprima
 *           INTERFACES: None
 *         SUBORDINATES: None
 * ========================================================================== */
bool spd_os::fileExists(const char* pStrName)
{
  bool res = true;
  if (nullptr == pStrName) {
    printf("NULL required parameter\n");
    res = false;
  }
  DWORD ftyp = GetFileAttributes(pStrName);
  if (ftyp == INVALID_FILE_ATTRIBUTES) {
    printf("Invalid file\n");
    res = false;
  }  
  return res;
}

/* ==========================================================================
 *        FUNCTION NAME: directoryExists
 * FUNCTION DESCRIPTION: Exists a directory
 *        CREATION DATE: 20170201
 *              AUTHORS: Francesco Diprima
 *           INTERFACES: None
 *         SUBORDINATES: None
 * ========================================================================== */
bool spd_os::directoryExists(const char* pStrName)
{
  bool res = true;
  if (nullptr == pStrName) {
    printf("NULL required parameter\n");
    res = false;
  }
  DWORD ftyp = GetFileAttributes(pStrName);
  if (ftyp == INVALID_FILE_ATTRIBUTES) {
    printf("Invalid attributes\n");
    res = false;
  }
  if (0 == (ftyp & FILE_ATTRIBUTE_DIRECTORY)) {
    printf("Not a directory\n");
    res = false;
  }
  return res;
}

/* ==========================================================================
 *        FUNCTION NAME: directoryOpen
 * FUNCTION DESCRIPTION: open a directory
 *        CREATION DATE: 20170201
 *              AUTHORS: Francesco Diprima
 *           INTERFACES: None
 *         SUBORDINATES: None
 * ========================================================================== */
bool spd_os::directoryOpen(void* dhOut, const char* pStrName)
{
  bool res = true;

  void* p_dhOut = dhOut;
  /*dhOut � creato con una malloc*/
  
  if (nullptr == pStrName) {
    printf("NULL required parameter\n");
    res = false;
  }
  if (nullptr == dhOut) {
    printf("Not NIL handle\n");
    res = false;
  }

  void* bufferForHandle = ::malloc(sizeof(PrivTDir));

  dhOut = std::move(bufferForHandle);
  
  PrivTDir* pDir = reinterpret_cast<PrivTDir*>(p_dhOut);
  
  ::memset(pDir,0,sizeof(PrivTDir));

  pDir->uMarker = MakeU32LE('J','D','I','R');
    
  bool expExists = directoryExists(pStrName);
  if (!expExists) {
    printf("Error, directory not exist.\n");
    res = false;
  }

  std::string lastP = "/*";
  std::string tf = pStrName + lastP;
  
  pDir->hFind = FindFirstFile(static_cast<const char*>(tf.c_str()), &pDir->ffd);
  
  /* Is first and no execute Next */
  pDir->bIsFirst = true;

  return res;
}

/* ==========================================================================
 *        FUNCTION NAME: directoryNextItem
 * FUNCTION DESCRIPTION: navigate to the next item
 *        CREATION DATE: 20170201
 *              AUTHORS: Francesco Diprima
 *           INTERFACES: None
 *         SUBORDINATES: None
 * ========================================================================== */
bool spd_os::directoryNextItem( void* dhIn, DirectoryItem& out)
{
  bool res = true;

  if (nullptr == dhIn) {
    printf("NIL handle\n");
    res = false;
  }
  
  PrivTDir* pDir = reinterpret_cast<PrivTDir*>(dhIn);
  
  if (nullptr == pDir) {
    printf("NULL directory pointer\n");
    res = false;
  }
  if (pDir->uMarker != MakeU32LE('J', 'D', 'I', 'R')) {
    printf("Not a directory structure\n");
    res = false;
  }

  bool bFindOk = false;
  if (pDir->bIsFirst)
  {
    /* Force condition to execute always next */
    pDir->bIsFirst = false;
    if (INVALID_HANDLE_VALUE != pDir->hFind)
    {
      bFindOk = true;
    }
  }
  else
  {
    if (INVALID_HANDLE_VALUE == pDir->hFind) {
      printf("Invalid handle of directory Find\n");
      res = false;
    }

    /* go to next file */
    if (0 != FindNextFile(pDir->hFind, &pDir->ffd))
    {
      bFindOk = true;
    }
  }  
  if (bFindOk)
  {
    ::memset(out.strName,0,sizeof(out.strName));
    
    ::memcpy(out.strName, pDir->ffd.cFileName
      , ::strlen(pDir->ffd.cFileName)+1);

    out.bIsDirectory =
     (FILE_ATTRIBUTE_DIRECTORY == 
        (pDir->ffd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY));
    out.bIsRegularFile = !out.bIsDirectory;

#if 0
    printf("out.strName=%s\n", out.strName);
#endif
    return true;
  }

  return false;
}

/* ==========================================================================
 *        FUNCTION NAME: directoryClose
 * FUNCTION DESCRIPTION: close directory
 *        CREATION DATE: 20170201
 *              AUTHORS: Francesco Diprima
 *           INTERFACES: None
 *         SUBORDINATES: None
 * ========================================================================== */
bool spd_os::directoryClose( void* dhIn)
{
  bool res = true;

  if (nullptr == dhIn) {
    printf("NIL handle\n");
    res = false;
  }
  
  PrivTDir* pDir = reinterpret_cast<PrivTDir*>(dhIn);
  
  if (nullptr == pDir) {
    printf("NULL directory pointer\n");
    res = false;
  }
  if (pDir->uMarker != MakeU32LE('J', 'D', 'I', 'R')) {
    printf("Not a directory structure\n");
    res = false;
  }
    
  if (INVALID_HANDLE_VALUE == pDir->hFind) {
    printf("Directory not correctly opened\n");
    res = false;
  }
  if (0 == FindClose(pDir->hFind)) {
    printf("Could not close directory\n");
    res = false;
  }

  return res;
}

/* ==========================================================================
 *        FUNCTION NAME: scan
 * FUNCTION DESCRIPTION: Scan a folder to search files
 *        CREATION DATE: 20170201
 *              AUTHORS: Francesco Diprima
 *           INTERFACES: None
 *         SUBORDINATES: None
 * ========================================================================== */
bool spd_os::scan(void* hdir, char * nameFile)
{
  bool bExit = false;
  bool endExternalLoop = false;
  bool lastLoop = true;
  const char* extjpg = "jpg";
  const char* extJPG = "JPG";
  const char* extfit = "fit";
  const char* extFIT = "FIT";
  DirectoryItem ditm;

  while (!bExit)
  {    
    bool exp = directoryNextItem(hdir, ditm);

    if (exp)
    {
      std::string strFN{ ditm.strName };

      if ((strFN == ".") || (strFN == ".."))
      {
        continue;
      }

      std::vector<char*> vec = fileExt(ditm.strName);
      char* ext = vec.at(1);

      if (!((0 == strcmp(ext, extJPG)) || (0 == strcmp(ext, extjpg))
        || (0 == strcmp(ext, extFIT)) || (0 == strcmp(ext, extfit))))
      {
        continue;
      }
      bExit = true;
    }
    else
    {
      bExit = true;
      endExternalLoop = true;
      lastLoop = false;
    }
  }
  if (lastLoop) {
    strncpy(nameFile, ditm.strName, strlen(ditm.strName));
  }
  return endExternalLoop;
}

/* ==========================================================================
*        FUNCTION NAME: fileExt
* FUNCTION DESCRIPTION: Get file extension
*        CREATION DATE: 20160727
*              AUTHORS: Francesco Diprima
*           INTERFACES: None
*         SUBORDINATES: None
* ========================================================================== */
std::vector<char*> spd_os::fileExt(const char* strN)
{
  std::vector<char*> vec;

  char nameFile[1024];
  ::strcpy( nameFile, strN );
  char* pch;
  char *path[32][256];
  const char* slash = "\\";

  pch = ::strtok(nameFile,slash);
  size_t count = 0;
  while (pch != NULL)
  {
    //printf ("%s\n",pch);
    *path[count] = pch;
    pch = ::strtok (NULL, slash);
    count++;
  }
  char *name = *path[count-1];

  char s_pathFileName[256];
  ::strcpy(s_pathFileName, *path[0]);
  for (size_t i = 1; i < count - 1; ++i)
  {
    ::strcat(s_pathFileName, slash);
    ::strcat(s_pathFileName, *path[i]);
  }
  ::strcat(s_pathFileName, slash);
  
  char s_pathResFile[256];
  ::strcpy (s_pathResFile, s_pathFileName);
  ::strcat(s_pathResFile, "Result");
  ::strcat(s_pathResFile, slash);
  
  char *nameL[32][256];
  pch = ::strtok(name,".");
  count = 0;
  while (pch != NULL)
  {    
    //printf ("%s\n",pch);
    *nameL[count] = pch;
    pch = ::strtok (NULL, ".");
    count++;
  }
  char* ext = *nameL[count-1];
  //char* fileName = *path[count-2];
  char s_fileName[256];
  ::strcpy (s_fileName, *nameL[0]);
  for (size_t i = 1; i < count - 1; ++i)
  {
    ::strcat(s_fileName, ".");
    ::strcat(s_fileName, *nameL[i]);
  }

  vec.push_back(s_fileName);
  vec.push_back(ext);
  vec.push_back(s_pathFileName);
  vec.push_back(s_pathResFile);

  return vec;
}
