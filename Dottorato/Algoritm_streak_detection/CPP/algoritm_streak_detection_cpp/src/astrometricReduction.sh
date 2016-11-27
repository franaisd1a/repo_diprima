#!/bin/bash
echo "========================================"
echo "Astrometric reduction"
echo "Start processing"

inParam=$#
expParam=1
expParamArcsecpix=3

if [ "$inParam" -ne "$expParam" ]
then
  if [ "$inParam" -ne "$expParamArcsecpix" ]
  then
	echo "Error in input parameter"
	echo "========================================"
	exit
  fi
fi

imgName=$1
echo $imgName


imgNameLength=${#imgName}
extLength=4

if [ "$imgNameLength" -le "$extLength" ]
then
  echo "Error in name file"
  echo "========================================"
  exit
fi


if [ "$inParam" -eq "$expParamArcsecpix" ]
then
	arcsecpixScaleL=$2
	arcsecpixScaleH=$3
	solve-field --overwrite --cpulimit 30 --guess-scale --no-plots --scale-units arcsecperpix --scale-low $arcsecpixScaleL --scale-high $arcsecpixScaleH $imgName
else
	solve-field --overwrite --cpulimit 30 --guess-scale $imgName
fi

#solve-field --overwrite --cpulimit 30 --guess-scale --no-plots --scale-units arcsecperpix --scale-low $arcsecpixScaleL --scale-high $arcsecpixScaleH $imgName



 
fileNameSz=imgNameLength-4
fileName=${imgName:0:$fileNameSz}

wcsFileName="$fileName.wcs"
wcsResult="$fileName.txt"

wcsinfo $wcsFileName>$wcsResult


echo "End processing"
echo "========================================"
