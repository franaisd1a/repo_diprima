echo OFF
echo Astrometric reduction on OS Windows

set /A inParam=0
for %%x in (%*) do Set /A inParam+=1


set /A expParam=1
set /A expParamArcsecpix=3

IF /I "%inParam%" NEQ "%expParam%" (
	IF /I "%inParam%" NEQ "%expParamArcsecpix%" ( 
		echo "Error in input parameter"
		exit
	)
)

echo "******************************"
set imgPath=%1
echo %imgPath%


set imgPath=%imgPath:\=/%
set imgPath=%imgPath:"=%

set disk=%imgPath:~0,1%

set name=%imgPath:~2%

IF /I "%disk%" EQU "C" ( 
	set strLeft=/cygdrive/c
)
IF /I "%disk%" EQU "D" ( 
	set strLeft=/cygdrive/d
)

set imgName=%strLeft%%name% 
echo %imgName%

echo "******************************"

set PATH=%PATH%:"C:\Cygwin\bin"

IF /I "%inParam%" EQU "%expParamArcsecpix%" ( 
	set arcsecpixScaleL=%2
	set arcsecpixScaleH=%3
	c:\cygwin\bin\bash.exe /cygdrive/d/repo_diprima/Dottorato/Algoritm_streak_detection/CPP/algoritm_streak_detection_cpp/src/astrometricReduction.sh %imgName% %arcsecpixScaleL% %arcsecpixScaleH%
) ELSE (
	c:\cygwin\bin\bash.exe /cygdrive/d/repo_diprima/Dottorato/Algoritm_streak_detection/CPP/algoritm_streak_detection_cpp/src/astrometricReduction.sh %imgName%
)


echo Fine
