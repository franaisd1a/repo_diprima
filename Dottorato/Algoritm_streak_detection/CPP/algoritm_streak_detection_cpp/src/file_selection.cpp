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
#include "file_selection.h"
#include "function_os.h"
#include "algo_selection.h"
#include "macros.h"

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
bool file_selection(char* input, bool folderMod)
{
  bool computation = true;

  /*
   * folderMod = true folder path
   * folderMod = false  file path
   */
  
#if _WIN32
  const char slash = 92;
#else
  const char slash = 47;
#endif

  const char* extjpg = "jpg";
  const char* extJPG = "JPG";
  const char* extfit = "fit";
  const char* extFIT = "FIT";

  
  char nameFile[1024]; //File name with path and extension
  char onlyNameF[256]; //File name without path and extens
  char fileExt[8]; //File extension
  char namePath[1024]; //File path without name
  char nameResFolder[1024]; //Result folder
    
  
  ::memset(nameFile,0,sizeof(nameFile));
  ::memset(onlyNameF, 0, sizeof(onlyNameF));
  ::memset(fileExt,0,sizeof(fileExt));
  ::memset(namePath,0,sizeof(namePath));
  ::memset(nameResFolder,0,sizeof(nameResFolder));

  void* hdir = ::malloc(512);
  
  bool exitLoop = false;
  bool createResDir = true;

  //Open input directory and create result folder
  bool openDir = false;
  bool createResF = false;

  if (folderMod) {
    openDir = spd_os::directoryOpen(hdir,input);
    if (openDir)
    {
      ::strcpy(namePath, input);
      
      //Control if last char in the path is slash
      char lastC = namePath[strlen(namePath) - 1];
      if (lastC!=slash) {
        namePath[strlen(namePath)] = slash;
      }

      //Create result folder for directory mode
      const char* resFolder = "Res_SPD";
      ::strcpy(nameResFolder, namePath);
      ::strcat(nameResFolder, resFolder);
      nameResFolder[strlen(nameResFolder)] = slash;
      bool dirEx = spd_os::directoryExists(nameResFolder);
      if (!dirEx) {
        createResF = spd_os::createDirectory(nameResFolder);
      }
    }
    else
    {
      printf("Error in directory opening.");
      exitLoop = false;
    }
  }

  while (!exitLoop)
  {
    if (folderMod)
    {
/* ----------------------------------------------------------------------- *
 * Read files name from folder                                             *
 * ----------------------------------------------------------------------- */
      
      char file[1024];
      ::memset(file, 0, sizeof(file));
      ::memset(nameFile, 0, sizeof(nameFile));
      ::memset(onlyNameF, 0, sizeof(onlyNameF));
      ::memset(fileExt, 0, sizeof(fileExt));

      exitLoop = spd_os::scan(hdir, file);

      ::strcat(nameFile, namePath);
      ::strcat(nameFile, file);

      if (strlen(file)>1) {
        std::vector<char*> vec = spd_os::fileExt(file);
        ::strcpy(onlyNameF, vec.at(0));
        ::strcpy(fileExt, vec.at(1));
      }
    }
    else
    {
/* ----------------------------------------------------------------------- *
 * Read file name                                                          *
 * ----------------------------------------------------------------------- */
      ::memset(nameFile,0,sizeof(nameFile));

      //Control correct file extension
      std::vector<char*> vec = spd_os::fileExt(input);
      char* ext = vec.at(1);

      ::strcpy(onlyNameF, vec.at(0));
      ::strcpy(fileExt, vec.at(1));
      ::strcpy( namePath, vec.at(2));

      if ( (0 == ::strcmp(ext, extJPG)) || (0 == ::strcmp(ext, extjpg))
        || (0 == ::strcmp(ext, extFIT)) || (0 == ::strcmp(ext, extfit)))
      {
        ::strncpy(nameFile, input, strlen(input));
        //Create result folder for single input file mode
        ::strcpy(nameResFolder, vec.at(2));
        ::strcat(nameResFolder, vec.at(0));
        ::strcat(nameResFolder, "_");
        ::strcat(nameResFolder, "Res");
        nameResFolder[strlen(nameResFolder)] = slash;
        bool dirEx = spd_os::directoryExists(nameResFolder);
        if (!dirEx) {
          createResF = spd_os::createDirectory(nameResFolder);
        }        
      }
      else
      {
        printf("Error. Not supported file extension. Only .fit and .jpg file.\n");
        exitLoop = true;
        continue;
      }
      exitLoop = true;
    }    
    //add method to monitor changes in a specific folder


    if ((strlen(nameFile) > strlen(input)+1) || !folderMod)
    {
/* ----------------------------------------------------------------------- *
 * Algo execution                                                          *
 * ----------------------------------------------------------------------- */

      std::vector<char *> inputFileV(5);
      inputFileV[0] = nameFile; //File name with path and extension
      inputFileV[1] = onlyNameF; //File name without path and extension
      inputFileV[2] = fileExt; //File extension
      inputFileV[3] = namePath; //File path without name
      inputFileV[4] = nameResFolder; //Result folder

#if SPD_DEBUG
      std::cout << "nameFile " << nameFile << std::endl;
      std::cout << "onlyNameF " << onlyNameF << std::endl;
      std::cout << "fileExt " << fileExt << std::endl;
      std::cout << "namePath " << namePath << std::endl;
      std::cout << "nameResFolder " << nameResFolder << std::endl;
#else
      computation = algo_selection(inputFileV);
#endif
    }
    std::cout << std::endl << std::endl;
    if (!exitLoop) { continue; }
  }

  if (folderMod && openDir)
  {
    bool expClose = spd_os::directoryClose(hdir);
  }

  return computation;
}
