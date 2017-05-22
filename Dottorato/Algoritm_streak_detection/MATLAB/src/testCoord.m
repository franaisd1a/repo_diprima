clear all;


pixel = [1670,2123];

pixel=pixel+1;

crpix0=2342.36734009;%2048;
crpix1=3116.41888428;%2048;
crval0=22.7446156954;%22.6753524881;
crval1=-0.858843676832;%-1.46606115242;
cd11=0.000544507564836;%5.4428578231399997e-004;
cd12=-8.56621045122E-05;%-8.56810168331e-005;
cd21=8.57150429781E-05;%8.5605731121699997e-005;
cd22=0.000544708340449;%5.4476445286300002e-004;

A_ORDER =                    2;
A_0_2   =   -1.32725964601E-07;
A_1_1   =   -1.32773881533E-07;
A_2_0   =     3.6464561227E-07;
B_ORDER =                    2;
B_0_2   =   -2.05542270573E-07;
B_1_1   =    2.79096231608E-07;
B_2_0   =   -2.84630422325E-08;


%% SIP distortion

u=pixel(1)-crpix0;
v=pixel(2)-crpix1;

f = A_0_2 * v^2 + A_1_1 * u*v + A_2_0 * u^2;
g = B_0_2 * v^2 + B_1_1 * u*v + B_2_0 * u^2;

u=u+f;
v=v+g;

% U=u+f;
% V=v+g;
% U=U+crpix0;
% V=V+crpix1;

%% tan_pixelxy2radec input U V output ra dec

%tan_pixelxy2xyzarr

xyz=zeros(1,3);

%%%%%%%%%%%%%%%%%%% tan_pixelxy2iwc
%tan_pixelxy2iwc
% u=U-crpix0;
% v=V-crpix1;


GM = [cd11, cd12; cd21, cd22];

x=cd11*u+cd12*v;
y=cd21*u+cd22*v;

%tan_iwc2xyzarr

x=-deg2rad(x);
y= deg2rad(y);

cosdec = cosd(crval1);
rx = cosdec * cosd(crval0);
ry = cosdec * sind(crval0);
rz = sind(crval1);

ix=ry;
iy=-rx;
norm=hypot(ix, iy);
ix= ix/ norm;
iy=	iy/ norm;

jx = iy * rz;
jy =         - ix * rz;
jz = ix * ry - iy * rx;

%no_rm=norm([jx,jy,jz]);
no_rm=sqrt(jx^2+jy^2+jz^2);

xyz(1) = ix*x + jx*y + rx;
xyz(2) = iy*x + jy*y + ry;
xyz(3) =        jz*y + rz; % iz = 0

no_rm2=sqrt(xyz(1)^2+xyz(2)^2+xyz(3)^2);

raF = atan2(xyz(2), xyz(1));
if (raF < 0)
    raF = raF+ 2.0 * pi;
end

decF = asin(xyz(3));

ra=degrees2dms(rad2deg(raF/15));
dec=degrees2dms(rad2deg(decF));


dataMaxImDMS=[1,29,51.28, -1,27,24.9];%8009
% dataMaxImSec(1,1)=dataMaxImDMS(1)*3600+dataMaxImDMS(2)*60+dataMaxImDMS(3);
% dataMaxImSec(1,2)=dataMaxImDMS(4)*3600+dataMaxImDMS(5)*60+dataMaxImDMS(6);

dataMaxImDeg(1,1)=(dataMaxImDMS(1)+dataMaxImDMS(2)/60+dataMaxImDMS(3)/3600)*15;
data(1,1)=(ra(1)+ra(2)/60+ra(3)/3600)*15;
if(dataMaxImDMS(4)>0)
    dataMaxImDeg(1,2)=dataMaxImDMS(4)+dataMaxImDMS(5)/60+dataMaxImDMS(6)/3600;
    data(1,2)=dec(1)+dec(2)/60+dec(3)/3600;
else
    dataMaxImDeg(1,2)=dataMaxImDMS(4)-dataMaxImDMS(5)/60-dataMaxImDMS(6)/3600;
    data(1,2)=dec(1)-dec(2)/60-dec(3)/3600;
end

res=(data-dataMaxImDeg)*3600


% cc = GM * [u+f;v+g];
% 
% x=crval0+cc(1)
% y=crval1+cc(2)
% 
% decHMS = degrees2dms(y)
% 
% raArcSec = raF*3600;
% raSec = raArcSec/15;
% t=raSec;
% hours = floor(t / 3600)
% t = t - hours * 3600;
% mins = floor(t / 60)
% secs = t - mins * 60