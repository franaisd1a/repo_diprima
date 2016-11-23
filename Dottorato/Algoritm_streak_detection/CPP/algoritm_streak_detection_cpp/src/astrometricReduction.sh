#!/bin/bash
echo "Astrometric reduction"
echo "Start processing"

inParam=$#
expParam=3

if [ "$inParam" -ne "$expParam" ]
then
  echo "Error in input parameter"
exit
fi

imgName=$1
echo $imgName

arcsecpixScaleL=$2
arcsecpixScaleH=$3

#solve-field --overwrite --cpulimit 30 --guess-scale --no-plots --scale-units arcsecperpix --scale-low $arcsecpixScaleL --scale-high $arcsecpixScaleH $imgName

imgNameLength=${#imgName}

fileNameSz=imgNameLength-4
fileName=${imgName:0:$fileNameSz}

wcsFileName="$fileName.wcs"
wcsResult="$fileName.txt"

wcsinfo $wcsFileName>$wcsResult






