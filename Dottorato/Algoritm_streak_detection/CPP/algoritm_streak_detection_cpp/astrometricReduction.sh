#!/bin/bash
echo "========================================"
echo "Astrometric reduction"
echo "Start processing"

#SOLVE_BIN=${SOLVE_BIN}

inParam=$#
expParam=2
expParamArcsecpix=4

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
	solve-field --dir $resWcsFile --overwrite --cpulimit 30 --no-plots --guess-scale --scale-units arcsecperpix --scale-low $arcsecpixScaleL --scale-high $arcsecpixScaleH $imgName
else
	solve-field --dir $resWcsFile --overwrite --cpulimit 30 --no-plots --guess-scale $imgName
	#/cygdrive/c/cygwin/lib/astrometry/bin/solve-field.exe --dir $resWcsFile --overwrite --cpulimit 30 --no-plots --guess-scale $imgName  
fi

#solve-field --overwrite --cpulimit 30 --guess-scale --no-plots --scale-units arcsecperpix --scale-low $arcsecpixScaleL --scale-high $arcsecpixScaleH $imgName


echo "End processing"
echo "========================================"