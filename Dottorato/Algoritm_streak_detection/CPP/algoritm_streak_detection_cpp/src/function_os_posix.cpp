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
#include "function_os_posix.h"

/* ==========================================================================
* MODULE PRIVATE MACROS
* ========================================================================== */

/* ==========================================================================
* MODULE PRIVATE TYPE DECLARATIONS
* ========================================================================== */
struct PrivTDir
{
  uint32_t uMarker;
  DIR* pOSDir;
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
  if (::mkdir(pStrName, S_IRWXU|S_IRGRP|S_IROTH) < 0){
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
  if (::rmdir(pStrName) < 0) {
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
  if(::unlink(pStrName) < 0){
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
  struct stat st;
  if (::stat(pStrName,&st) < 0) {
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
  struct stat st;
  
  if (::stat(pStrName,&st) < 0) {
    printf("Not a directory\n");
    res = false;
  }
  return (0 != (st.st_mode & S_IFDIR));
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
    
  pDir->pOSDir = ::opendir(pStrName);
  
  if(nullptr == pDir->pOSDir){
	  printf("Could not open directory\n");
    res = false;
  }
    
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

  struct dirent entry;
  struct dirent* pEntry = nullptr;

  if (0 == ::readdir_r(pDir->pOSDir,&entry,&pEntry))
  {
    if (nullptr != pEntry)
    {
	  ::memset(out.strName,0,sizeof(out.strName));
	  
      ::memcpy(out.strName, entry.d_name
      , ::strlen(entry.d_name)+1);
          
      out.bIsDirectory   = (entry.d_type == DT_DIR);
      out.bIsRegularFile = (entry.d_type == DT_REG);
      return true;
    }
  }
  else
  {
    if(true){
	printf("Error reading directory entry");
	}
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
    
  if (::closedir(pDir->pOSDir) < 0) {
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
  ::strcpy ( nameFile, strN );
  char* pch;
  char *path[32][256];
  //const char* slash = "\\";
  const char* slash = "//";

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
  ::strcpy (s_pathFileName, *path[0]);
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
  
  pch = strtok(name,".");
  while (pch != NULL)
  {    
    //printf ("%s\n",pch);
    *path[count] = pch;
    pch = ::strtok(NULL, ".");
    count++;
  }
  char* fileName = *path[count-2];
  char* ext = *path[count-1];
  
  vec.push_back(fileName);
  vec.push_back(ext);
  vec.push_back(s_pathFileName);
  vec.push_back(s_pathResFile);

  return vec;
}
