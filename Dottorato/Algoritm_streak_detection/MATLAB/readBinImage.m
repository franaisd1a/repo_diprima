fi  =fopen('D:\Space_debris\repo_diprima.git\Dottorato\Algoritm_streak_detection\CPP\algoritm_streak_detection_cpp\builds\img.bin','rb');
AA = fread(fi,[4096,4096],'uint16');
fclose(fi);
clear fi;

rawImg=uint16(AA);

 colorRange=255;
 percentile=[0.432506, (1-0.97725)];
 %percentile=[0.370699 (1-0.999968)];
 
 histStretch = histogramStretching(rawImg, colorRange, percentile);
 
 Img_input = histStretch.stretchImg;
 
 imshow(Img_input)