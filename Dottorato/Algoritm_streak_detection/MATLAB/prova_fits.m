clear all; 
close all; 
clc; 

%% Global variables

%% Macro

FIGURE=1;
FIGURE_1=0;
FILE=0;
CLEAR=0;
backgroundSubtraction=1;
differentThreshold=0;

%% Input Folder
wrkDir='D:\Dottorato\Space debris image\HAMR-14_15-05-2013\Foto\Foto 14-05-2013';

fit_format='*.fit';

directory=fullfile(wrkDir,fit_format);
files=dir(directory);
t_start=tic;
    
%% Strart processing

for file_number=1:1%length(files)
    
    name_file=files(file_number,1).name;
    [pathstr,name,ext] = fileparts(name_file);
    image=name;
        
    image_fit=strcat(image,ext);
    
    data = fitsread(image_fit);
    % [data_img,map] = imread(image_fit);
%     info = fitsinfo(image_fit);
%     fits_info = imfinfo(image_fit);
    
    maxValue=max(max(data));
    minValue=min(min(data));
    rangeValue=maxValue-minValue;
    
    img8=uint8(data.*(2^8)./(2^16));
    imgScalataMax=uint8(data.*(2^8)./maxValue);
    
    imgScalata=uint8(((data-minValue)./(rangeValue)).*(2^8));
    
    
%     imgScalata=uint8(data.*rangeValue./maxValue);
    imgHisteq = histeq(data,[0, 65535]);
    
    figure('name','Histogram');
    imhist(data);
    figure('name','Input file 8byte');
    imshow(img8);
    figure('name','Input file scaled respect max value');
    imshow(imgScalataMax);
    figure('name','Input file scaled');
    imshow(imgScalata);
    figure('name','Equalized image');
    imshow(imgHisteq);
    
end


