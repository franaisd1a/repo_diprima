function [ output ] = fftZeroPaddingInterpolation( varargin )

% Interpolate a N1 X N2 image using FFT, with interpolation factor F,
% creating an image with pixel size reduced by F respect to the original one

global FIGURE FIGURE_1
disp('Start Interpolation function.')
tStart=tic;

%% Input

%  1) N1 X N2 image (Aij); N1,2 must be a power of 2
%  2) interpolation factor F which must be a power of 2

%% Output

% Output struct with different fields:
% 1) output.error: boolean value, 1 is error
% 2) output.paddImg: interpolate image 

output={};
output.error=1;

%% Input validation

if nargin~=2
    disp('Error! Wrong number of parameters.')
    return
else
    disp('Correct number of parameters.')
    img=varargin{1};
    F=varargin{2};
end

%Image dimension
szimg=size(img);
if rem(log2(szimg(1)),1)~=0 || rem(log2(szimg(2)),1)~=0
    disp('Error! The image size is not a power of 2.')
    output.error=1;
    return
end

% *********************************************************************** %
%% Processing
% *********************************************************************** %
try
  
% ----------------------------------------------------------------------- %
%% FFT of the input image
% ----------------------------------------------------------------------- %

% FFT of the input image img[i,j]: IMG[i,j]=FFT(img[i,j]), i=0,N1-1 j=0,N2-1
% it is supposed that the FFT implementation does leave the spectrum wrapped,
% i.e. (N1,2/2) and (N1,2-1) correspond to the lowest and to the central
% frequencies respectively.
    
    %Intensity calculation before fft
    imgINTENSITYbeforeFFT = sum(sum(abs(img.*img)));
    
    FFTimage = fft2(img);
    
    %Intensity calculation before fft
    imgINTENSITYbeforeFFT2 = sum(sum(abs(FFTimage.*FFTimage)));    
    
    if FIGURE_1
        figure('name','FFT image')
        imagesc(log(abs(FFTimage))); colormap(gray); colorbar;
        figure('name','FFT line')
        plot(abs(FFTimage(end/2,1:end))); colormap(gray); colorbar;
    end
    
    %%%%PROVA
    iFFTimage = ifft2(FFTimage);
    INTENSITY = sum(sum(abs(iFFTimage.*iFFTimage)));
    %%%%PROVA
    
% ----------------------------------------------------------------------- %
%% Zero padding of the transformed image
% ----------------------------------------------------------------------- %
    
    szInterpolateImg=[F*szimg(1),F*szimg(2)];
    interpolateImg=zeros(szInterpolateImg(1),szInterpolateImg(2));
        
    interpolateImg(1:szimg(1)/2,1:szimg(2)/2)=...
        FFTimage(1:szimg(1)/2,1:szimg(2)/2);
    
    interpolateImg(1:szimg(1)/2,szInterpolateImg(2)-szimg(2)/2:end)=...
        FFTimage(1:szimg(1)/2,szimg(2)/2:end);
    
    interpolateImg(szInterpolateImg(1)-szimg(1)/2:end,1:szimg(2)/2)=...
        FFTimage(szimg(1)/2:end,1:szimg(2)/2);
    
    interpolateImg(szInterpolateImg(1)-szimg(1)/2:end,szInterpolateImg(2)-szimg(2)/2:end)=...
        FFTimage(szimg(1)/2:end,szimg(2)/2:end);
    
% ----------------------------------------------------------------------- %
%% Inverse FFT of the zero padded image
% ----------------------------------------------------------------------- %
    
    %Intensity calculation after fft
    imgINTENSITYafterFFT2 = sum(sum(abs(interpolateImg.*interpolateImg)));
    
    output.paddImg=abs(ifft2(interpolateImg));%*F;
        
    %Intensity calculation after fft
    imgINTENSITYafterFFT = sum(sum(abs(output.paddImg.*output.paddImg)));
    %rap=imgINTENSITYbeforeFFT/imgINTENSITYafterFFT;
    
    if imgINTENSITYbeforeFFT/imgINTENSITYafterFFT > F%*F
        output.paddImg=output.paddImg*F;
    end
    
    output.paddImgSz=size(output.paddImg);
    
    if FIGURE
        figure('name','FFT zero padding interpolation')
        imagesc(uint8(abs(output.paddImg))); 
        colormap(gray); colorbar; axis equal;
    end
    
    tElapsed = toc(tStart);    
    disp(sprintf('End fftZeroPaddingInterpolation funtion %d sec.', tElapsed));
    output.error=0;
    
% ??????????????????????????????????????????????????????????????????????? %
%% Error handling
% ??????????????????????????????????????????????????????????????????????? %  

catch ME
    output.error=1;
    disp('Error using fftZeroPaddingInterpolation function.');
    disp(ME.message);
    disp(sprintf('\n'));
    
    for i=1:length(ME.stack)
        disp(sprintf('Error in %s (line %d)', ME.stack(i,1).name, ME.stack(i,1).line));
        
        disp(sprintf('\n'));
    end
    
    %rethrow(ME);
end

end