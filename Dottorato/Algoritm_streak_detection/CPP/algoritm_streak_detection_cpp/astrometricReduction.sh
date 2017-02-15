#!/bin/bash
echo "========================================"
echo "Astrometric reduction"
echo "Start processing"

inParam=$#
expParam=3
expParamArcsecpix=5

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

resWcsFile=$2
echo $resWcsFile

onlyFileName=$3
echo $onlyFileName

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
	arcsecpixScaleL=$3
	arcsecpixScaleH=$4
	solve-field --dir $resWcsFile --overwrite --cpulimit 30 --guess-scale --no-plots --scale-units arcsecperpix --scale-low $arcsecpixScaleL --scale-high $arcsecpixScaleH $imgName
else
	solve-field --dir $resWcsFile --overwrite --cpulimit 30 --guess-scale $imgName
fi

#solve-field --overwrite --cpulimit 30 --guess-scale --no-plots --scale-units arcsecperpix --scale-low $arcsecpixScaleL --scale-high $arcsecpixScaleH $imgName




fileName="$resWcsFile/$onlyFileName"
echo $fileName

wcsFileName="$fileName.wcs"
wcsResult="$fileName.txt"

wcsinfo $wcsFileName>$wcsResult


echo "End processing"
echo "========================================"
