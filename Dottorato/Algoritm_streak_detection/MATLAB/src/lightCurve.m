clear all;
close all;
clc;

%% Input file

% file = 'D:\repo_diprima\Dottorato\Algoritm_streak_detection\CPP\algoritm_streak_detection_cpp\builds\lightCurve_Along_0.txt';
file=cell(2);
file{1} = 'D:\repo_diprima\Dottorato\Algoritm_streak_detection\CPP\algoritm_streak_detection_cpp\builds\lightCurveRot_Along_1.txt';
file{2} = 'D:\repo_diprima\Dottorato\Algoritm_streak_detection\CPP\algoritm_streak_detection_cpp\builds\lightCurve_Along_0.txt';

name_fileJPG = 'D:\repo_diprima\Dottorato\Algoritm_streak_detection\CPP\algoritm_streak_detection_cpp\builds\ROI.jpg';

name_fileJPG2 = 'D:\repo_diprima\Dottorato\Algoritm_streak_detection\CPP\algoritm_streak_detection_cpp\builds\ROIrot.jpg';

% name_fileFIT = 'C:\Users\Francesco Diprima\Desktop\prova\41384.00007944.TRK\41384.00007944.TRK.FIT';

%% Plot light curve

for i=1:length(file)
    fid = fopen(file{i},'r');
    wcs = textscan(fid,'[%d] %d %d');
    fclose(fid);
    
    value = wcs{1,1};
    pos   = [wcs{1,2},wcs{1,3}]+1;
    
    figure(i);
    plot(value);
    axis on;
    grid on;
    hold on;
    order = 4;
    framelen = 21;
    
    y = sgolayfilt(double(value),order,framelen);
    plot(y,'r')
end

%% Input image

Img_input=imread(name_fileJPG);
SzImg_input = size(Img_input);

figure(20);
imshow(Img_input);
hold on;
plot(pos(:,1),pos(:,2));

figure(30);
[X,Y] = meshgrid(1:SzImg_input(1,2),1:SzImg_input(1,1));
surf(X,Y,double(Img_input));

% rawImg = (fitsread(name_fileFIT));
% 
% figure(3);
% imshow(rawImg);
% hold on;
% plot(pos(:,1),pos(:,2));

%% Image rot


Img_input2=imread(name_fileJPG2);
SzImg_input2 = size(Img_input2);

figure(31);
[X2,Y2] = meshgrid(1:SzImg_input2(1,2),1:SzImg_input2(1,1));
surf(X2,Y2,double(Img_input2));


