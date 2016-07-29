clc; % Clear the command window.
close all; % Close all figures (except those of imtool.)
imtool close all; % Close all imtool figures.
clear; % Erase all existing variables.
workspace; % Make sure the workspace panel is showing.
fontSize = 20;

% Change the current folder to the folder of this m-file.
if(~isdeployed)
cd(fileparts(which(mfilename)));
end

% Read in standard MATLAB grayscale demo image.
grayImage = imread('cameraman.tif');
subplot(2, 2, 1);
imshow(grayImage, []);
title('Original Grayscale Image');
set(gcf, 'Position', get(0,'Screensize')); % Enlarge figure to full screen.

% Let's get its histogram.
[pixelCount grayLevels] = imhist(grayImage);
subplot(2, 2, 2);
bar(pixelCount);
title('Histogram of original image');
xlim([0 grayLevels(end)]); % Scale x axis manually.

% Convert to a 16 bit image
grayImage16 = 256 * uint16(grayImage);

% Let's get its 16 bit histogram.
numberOfBins16 = double(intmax('uint16'));
[pixelCount16 grayLevels16] = imhist(grayImage16, numberOfBins16);
subplot(2, 2, 3);
bar(grayLevels16, pixelCount16);
title('Histogram of 16 bit image');
xlim([0 grayLevels16(end)]); % Scale x axis manually.

% Convert 16 bit image to a 12 bit image.
grayImage12 = uint16(grayImage16 / 16);
numberOfBins12 = round(numberOfBins16 / 16);

% Let's get its 12 bit histogram.
[pixelCount12 grayLevels12] = imhist(grayImage12, numberOfBins12);
subplot(2, 2, 4);
bar(grayLevels12, pixelCount12);
title('Histogram of 12 bit image');
xlim([0 grayLevels12(end)/16]); % Scale x axis manually.