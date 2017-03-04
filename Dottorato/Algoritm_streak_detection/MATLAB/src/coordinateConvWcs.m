clear all;
close all;
clc;

%% Read .wcs file

file = 'C:\Users\Francesco Diprima\Desktop\prova\41384.00007944.TRK\out.txt';

fid = fopen(file,'r');
wcs = textscan(fid,'%s %s');
fclose(fid);

for i=1:length(wcs{1,1})    
    if (strcmp(wcs{1,1}{i,1},'pixscale'))
        pixscale = str2double(wcs{1,2}{i,1});
    elseif (strcmp(wcs{1,1}{i,1},'orientation'))
        orientation = str2double(wcs{1,2}{i,1});
    elseif (strcmp(wcs{1,1}{i,1},'ra_center'))
        ra_center = str2double(wcs{1,2}{i,1});
    elseif (strcmp(wcs{1,1}{i,1},'dec_center'))
        dec_center = str2double(wcs{1,2}{i,1});
    elseif (strcmp(wcs{1,1}{i,1},'imagew'))
        imagew = str2double(wcs{1,2}{i,1});
    elseif (strcmp(wcs{1,1}{i,1},'imageh'))
        imageh = str2double(wcs{1,2}{i,1});
    end    
end

%% Coordinate conversion

%                   Colonna   , Riga
%Centroid streaks: (621.167664,588.136780) deb_260
%Centroid streaks: (1987.132202,2092.038818) 7800.trk
%Centroid streaks: (1669.546875,2122.958008) 7944-trk

stkPimg = [1669.546875,2122.958008];

imgC = [imagew/2 , imageh/2];


stkCs = pixscale*(imgC - stkPimg);



% ra = ra_center + (stkCs(2)*cosd(orientation) + stkCs(1)*sind(orientation))/(3600);
% dec = dec_center + (-stkCs(2)*sind(orientation) + stkCs(1)*cosd(orientation))/(3600);

ra = ra_center + (stkCs(1)*cosd(orientation) + stkCs(2)*sind(orientation))/(3600);
dec = dec_center + (-stkCs(1)*sind(orientation) + stkCs(2)*cosd(orientation))/(3600);

decHMS = degrees2dms(dec)

raArcSec = ra*3600;
raSec = raArcSec/15;
t=raSec;
hours = floor(t / 3600)
t = t - hours * 3600;
mins = floor(t / 60)
secs = t - mins * 60



