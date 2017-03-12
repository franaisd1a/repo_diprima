clear all;
close all;
clc;

%% Plot light curve

file = 'D:\repo_diprima\Dottorato\Algoritm_streak_detection\CPP\algoritm_streak_detection_cpp\builds\lightCurve_Along_0.txt';
name_fileJPG = 'C:\Users\Francesco Diprima\Desktop\prova\41384.00007944.TRK\41384.00007944.TRK.jpg';
name_fileFIT = 'C:\Users\Francesco Diprima\Desktop\prova\41384.00007944.TRK\41384.00007944.TRK.FIT';

fid = fopen(file,'r');
wcs = textscan(fid,'[%d] %d %d');
fclose(fid);

value = wcs{1,1};
pos   = [wcs{1,2},wcs{1,3}]+1;

figure(1);
plot(value);
axis on;
grid on;


Img_input=imread(name_fileJPG);

figure(2);
imshow(Img_input);
hold on;
plot(pos(:,1),pos(:,2));




rawImg = (fitsread(name_fileFIT));

figure(3);
imshow(rawImg);
hold on;
plot(pos(:,1),pos(:,2));


