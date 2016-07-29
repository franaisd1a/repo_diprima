clear all;
close all;
clc;

%% Global variables

global FIGURE FIGURE_1

%% Macro

FIGURE=1;
FIGURE_1=0;
FILE=0;
CLEAR=0;

%% Input Folder

wrkDir=pwd;
fileDir=fullfile(wrkDir,'src');
addpath(fileDir);
% wrkDir='D:\Dottorato\Space debris image\HAMR-14_15-05-2013\Foto\Foto 14-05-2013';
inputDataDir=pwd;

name_file='hamr_.fit';%hamr_186 %picture
name_fileJPG='hamr_.jpg';%hamr_186 %picture

fprintf('File name: %s\n', name_file);

Img_inputJPG=imread(name_fileJPG);

Img_input = (fitsread(name_file));%.*(2^8)./(2^16));
% [data_img,map] = imread(image_fit);
info = fitsinfo(name_file);
fits_info = imfinfo(name_file);

if 8==fits_info.BitDepth
    Img_input=uint8(Img_input);
elseif 16==fits_info.BitDepth
    Img_input=uint16(Img_input);
else
    disp('Error! Unsupported pixel type.')
end

outputFit = histogramStretching(Img_input, 255,[0.370699 (1-0.999968)]);


figure('name','Input file fit');
imshow(outputFit.stretchImg);

figure('name','Input file jpg');
imshow(Img_inputJPG);




