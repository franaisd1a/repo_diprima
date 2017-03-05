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
    elseif (strcmp(wcs{1,1}{i,1},'cd11'))
        cd11 = str2double(wcs{1,2}{i,1});
    elseif (strcmp(wcs{1,1}{i,1},'cd12'))
        cd12 = str2double(wcs{1,2}{i,1});
    elseif (strcmp(wcs{1,1}{i,1},'cd21'))
        cd21 = str2double(wcs{1,2}{i,1});
    elseif (strcmp(wcs{1,1}{i,1},'cd22'))
        cd22 = str2double(wcs{1,2}{i,1});
    elseif (strcmp(wcs{1,1}{i,1},'crpix0'))
        crpix0 = str2double(wcs{1,2}{i,1});
    elseif (strcmp(wcs{1,1}{i,1},'crpix1'))
        crpix1 = str2double(wcs{1,2}{i,1});
    elseif (strcmp(wcs{1,1}{i,1},'crval0'))
        crval0 = str2double(wcs{1,2}{i,1});
    elseif (strcmp(wcs{1,1}{i,1},'crval1'))
        crval1 = str2double(wcs{1,2}{i,1});
    end    
end

%% Coordinate conversion

%                   Colonna   , Riga
%Centroid streaks: (621.167664,588.136780) deb_260
%Centroid streaks: (1987.132202,2092.038818) 7800.trk
%Centroid streaks: (1669.546875,2122.958008) 7944-trk

stkPimg = [1669.546875,2122.958008];

% imgC = [imagew/2 , imageh/2];
% stkCs = pixscale*(imgC - stkPimg);
% 
% ra = ra_center + (stkCs(2)*cosd(orientation) + stkCs(1)*sind(orientation))/(3600);
% dec = dec_center + (-stkCs(2)*sind(orientation) + stkCs(1)*cosd(orientation))/(3600);
% 
% % ra = ra_center + (stkCs(1)*cosd(orientation) + stkCs(2)*sind(orientation))/(3600);
% % dec = dec_center + (-stkCs(1)*sind(orientation) + stkCs(2)*cosd(orientation))/(3600);
% 
% decHMS = degrees2dms(dec)
% 
% raArcSec = ra*3600;
% raSec = raArcSec/15;
% t=raSec;
% hours = floor(t / 3600)
% t = t - hours * 3600;
% mins = floor(t / 60)
% secs = t - mins * 60


%% Gnomonic (Tangent Plane) Projection

u=-crpix0+stkPimg(1);
v=-crpix1+stkPimg(2);

A_ORDER =                    2;
A_0_2   =   -6.41784648205E-08;
A_1_1   =    2.64025876332E-08;
A_2_0   =    1.80569466722E-07;
B_ORDER =                    2;
B_0_2   =   -7.05225042001E-08;
B_1_1   =    5.77274145963E-07;
B_2_0   =   -1.40391055853E-07;

f = A_0_2 * v^2 + A_1_1 * u*v + A_2_0 * u^2;
g = B_0_2 * v^2 + B_1_1 * u*v + B_2_0 * u^2;

GM = [cd11, cd12; cd21, cd22];

cc = GM * [u+f;v+g];

x=crval0+cc(1)
y=crval1+cc(2)

decHMS = degrees2dms(y)

raArcSec = x*3600;
raSec = raArcSec/15;
t=raSec;
hours = floor(t / 3600)
t = t - hours * 3600;
mins = floor(t / 60)
secs = t - mins * 60
