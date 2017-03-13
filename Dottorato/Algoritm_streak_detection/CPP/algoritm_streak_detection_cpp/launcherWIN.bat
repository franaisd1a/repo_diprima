echo OFF
set "myAPP=.\astrometricReduction.sh"
set "param1=%1"
set "param2=%2"

set "myCMD=%myAPP% %param1% %param2%"
bash %myCMD%