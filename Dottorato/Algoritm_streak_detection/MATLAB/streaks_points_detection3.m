clear all; 
close all; 
clc;
fprintf('\nCode name: %s\n\n',mfilename)

%% Global variables

global FIGURE FIGURE_1

%% Macro

FIGURE=1;
FIGURE_1=0;
FILE=0;
CLEAR=0;
backgroundSubtraction=1;
differentThreshold=0;
FIT=1;
dilate=0;

%% Input Folder

wrkDir=pwd;
fileDir=fullfile(wrkDir,'src');
addpath(fileDir);
if FIT
    inputDataDir='D:\Dottorato\Space debris image\HAMR-14_15-05-2013\Foto\Foto 14-05-2013';
    extension='.fit';
else
    inputDataDir=wrkDir;
    extension='.jpg';
end
resultDir=fullfile(inputDataDir,'Result');
mkdir(resultDir);

if FILE     %Lettura da cartella
    extensionSearch=strcat('*',extension);
    directory=fullfile(inputDataDir,extensionSearch);
    files=dir(directory);
else        %Lettura singolo file
    files=1;
    name_picture=strcat('hamr_204',extension);%hamr_186 150 209 204 170
end

for file_number=1:length(files)
    
    t_start=tic;
    
%% Strart processing

    if FILE
        name_picture=files(file_number,1).name;
    end
    name_file=fullfile(inputDataDir,name_picture);
    
    fprintf('File name: %s\n', name_picture);
    
    [pathstr,name,ext] = fileparts(name_file);
    
    if FIT
        rawImg = (fitsread(name_file));
        info = fitsinfo(name_file);
        fits_info = imfinfo(name_file);
        
        if 8==fits_info.BitDepth
            rawImg=uint8(rawImg);
        elseif 16==fits_info.BitDepth
            rawImg=uint16(rawImg);
        else
            disp('Error! Unsupported pixel type.')
        end

%% Histogram Stretching

        colorRange=255;
        percentile=[0.432506, (1-0.97725)];
        %percentile=[0.370699 (1-0.999968)];

        histStretch = histogramStretching(rawImg, colorRange, percentile);
        
        Img_input = histStretch.stretchImg;
        
    else
        Img_input=imread(name_file);
    end
    
    I_input_size = size(Img_input);
    borders = [0.015, 0.985];   %[0.005, 0.995]
    I_borders = [ceil(borders(1).*I_input_size), ...
                 floor(borders(2).*I_input_size)];
    
    if(FIGURE)
        a=300;
        if FILE
            a=a+file_number;
            h=figure(a);
            imshow(Img_input);
        else
            h=figure(a);%'name','Input file');
            imshow(Img_input);
        end
    end

% ======================================================================= %
%% Points detection
% ======================================================================= %

%% Gaussian filter    
    
    hsize=[100 100];
    sigma=30;%10 25
    gaussFilter = gaussianFilter( Img_input, hsize, sigma);

%% Median filters

    if ~gaussFilter.error
        littleKerlen=[3,3];
        bigKerlen=[15,15];%[21,21];
        madian = medianFilters( Img_input, littleKerlen, bigKerlen);
    end
    
%% Hough transform

    if ~madian.error
        angle = houghTransform( madian.medianImg );
    end
    
%% Morphology opening

    if ~angle.error
        dimLine=60;%40 20
        morphOpen = morphologyOpen( madian.medianImg, dimLine, ...
                                    angle.tetaStreak);
    end
    
%% Morphology TopHat
    
    if ~morphOpen.error
        figureName='Morphology TopHat for points detection';
        morphTopHat = imgSubtraction( madian.medianImg, ...
                                      morphOpen.openImg, ...
                                      figureName);
    end
    
%% Background subtraction

    if ~morphTopHat.error
        if backgroundSubtraction
            figureName='Background subtraction image for points detection';
            backgroundSub = imgSubtraction( morphTopHat.subtractionImg, ...
                                            gaussFilter.blurImg, ...
                                            figureName);
        else
            backgroundSub.subtractionImg = morphTopHat.subtractionImg;
        end
    end

%% Binarization
    
    if ~backgroundSub.error
        figureName='Binary image for points detection';
        pointBinary = binarization( backgroundSub.subtractionImg, ...
                                    differentThreshold, ...
                                    figureName);
    end
    
%% Convolution kernel
    
    if ~pointBinary.error
        k=ones(3);
        figureName='Convolution kernel for points detection';
        convPoint = convolution( pointBinary.binaryImg, ...
                                  k, ...
                                  sum(sum(k)), ...
                                  figureName);
    end
    
%% Remove Salt and Pepper Noise from Image

    if ~convPoint.error
        P=5;%3
        remSalPep = removeSaltPepper( convPoint.convImg, P);
    end

%% Connected components: points
    
    if ~remSalPep.error
        point = connectedComponentsPoints( remSalPep.remSaltPepperImg, ...
                                           I_borders);
    end
    
% ======================================================================= %
%% Streaks detection
% ======================================================================= %

%% Binarization of the morphological opening image for streaks detection

    if ~point.error
        figureName='Binary image for streaks detection';
        streakBinary = binarization( morphOpen.openImg, ...
                                     ~differentThreshold, ...
                                     figureName);
    end
    
%% Morphology dilation

    if ~streakBinary.error
        if dilate
            dilatation = morphologyDilatation( remSalPep.remSaltPepperImg, ...
                                               point.min_points_diameter/2)
        else
            dilatation.error=0;
            dilatation.dilateImg=remSalPep.remSaltPepperImg;
        end
    end
    
%% Points subtraction

    if ~dilatation.error
        figureName='Points subtraction for streaks detection';
        streakLessPoint = imgSubtraction( streakBinary.binaryImg, ...
                                          dilatation.dilateImg, ...
                                          figureName);
    end
    
%% Convolution kernel for streaks detection
    
    if ~streakLessPoint.error
        k=ones(1,21);
        convThreshold=9;
        figureName='Covolution kernel for streak detection';
        convStreak = convolution( streakLessPoint.subtractionImg, ...
                                  k, ...
                                  convThreshold, ...
                                  figureName);
    end
    
%% Connected components: streaks
    
    if ~convStreak.error
        streaks = connectedComponentsStreaks( convStreak.convImg, ...
                                            I_borders, ...
                                            point);
    end
        
%% Write result    
    
    nameTXT=strcat(name,'.txt');
    resultFileName=fullfile(resultDir,nameTXT);
    
    if ~streaks.error
        writeFile = writeResult( resultFileName, point, streaks);
    end
    
%% Plot of streaks and points    
    
    if(FIGURE)
        figure(a);
        hold on;
        % Plot streaks' centroids
        if isfield(streaks, 'STREAKS')
            plot(streaks.STREAKS(:,1),streaks.STREAKS(:,2),'*r')
        end
        if exist('STREAKSbig','var')
            plot(STREAKSbig(:,1),STREAKSbig(:,2),'*m')
        end
        %Plot points' centroids
        if isfield(point, 'POINTS')
            plot(point.POINTS(:,1),point.POINTS(:,2),'+g')
        end
        if exist('POINTSbig','var')
            plot(POINTSbig(:,1),POINTSbig(:,2),'+b')
        end
    end
    nameFig=strcat(name,'.fig');
    pathFig=fullfile(resultDir,nameFig);
    saveas(h,pathFig);
    
%% End of process
%Time calculation
    
    t_tot=toc(t_start);
    fprintf('Total file time: %d sec\n\n', t_tot);
    if FILE
        timeTOT(file_number)=t_tot;
        
        close all;
        clearvars -except   timeTOT FIGURE FIGURE_1 FILE CLEAR ...
                            backgroundSubtraction differentThreshold ...
                            FIT dilate files jpg_format inputDataDir ...
                            resultDir
    end    
end

if FILE
    nameMatrix='computationTime.mat';
    pathMatrix=fullfile(resultDir,nameMatrix);
    save(pathMatrix,'timeTOT');
end

fprintf('\nEnd process.\n');
