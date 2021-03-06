clear all; 
% close all; 
% clc;
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
subtractPointsImg=0;
FIT=0;
dilate=0;
ELLIPSE=0;

%% Input Folder

wrkDir=pwd;
fileDir=fullfile(wrkDir,'src');
addpath(fileDir);
if FIT
    %inputDataDir='D:\Dottorato\Space debris image\HAMR-14_15-05-2013\Foto\Foto 14-05-2013';
    inputDataDir='G:\Dottorato\Space debris image\SPADE\20161003';
    %inputDataDir='D:\Dottorato\II anno\img detriti';
    extension='.fit';
else
    %inputDataDir=wrkDir;
    %inputDataDir='D:\Dottorato\II anno\img detriti';
    %inputDataDir='D:\Dottorato\Algoritm_streak_detection';
    inputDataDir='D:\Dottorato\Space debris image\SPADE\20161005\jpegFormat';%SPADE\20161005\jpegFormat
    %inputDataDir='G:\Dottorato\Space debris image\EUTELSAT\1 Luglio 2013';
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
    name_picture=strcat('41384.00007909.TRK',extension);%hamr_186 150 209 204 170 deb_260 41384.00007800.TRK
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
    thickness = 0.005;%0.015 %0.005
    borders = [thickness, 1-thickness];
    I_borders = [ceil(borders(1).*I_input_size), ...
                 floor(borders(2).*I_input_size)];
    
    if(FIGURE)
        a=100;
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
    
    hsize=[31 31];%[100 100];
    sigma=30;%10 25
    gaussFilter = gaussianFilter( Img_input, hsize, sigma);
    meanFIlter = mean2(Img_input);

%% Median filters

    if ~gaussFilter.error
        littleKerlen=[3,3];%[7 7];%[3,3];
        bigKerlen=[21,21];%[15,15];%[21,21];
        madian = medianFilters( Img_input, littleKerlen);%, bigKerlen);
    end
   
%% Background subtraction

    if ~madian.error
        if backgroundSubtraction
            figureName='Background subtraction image for points detection';
            backgroundSub = imgSubtraction( madian.medianImg, ...
                                            gaussFilter.blurImg, ...
                                            figureName);
        else
            backgroundSub.subtractionImg = madian.medianImg;
        end
    end
    
%% Hough transform

    if ~backgroundSub.error
        %angle = houghTransform( backgroundSub.subtractionImg);
        angle.error = 0;
        angle.tetaStreak = 0;
    end
    
%% Morphology opening

    if ~angle.error
        dimLine=40;%20
        morphOpen = morphologyOpen( backgroundSub.subtractionImg, ...
                                    dimLine, ...
                                    angle.tetaStreak);
    end

%% Morphology TopHat
    
    if ~morphOpen.error
        figureName='Morphology TopHat for points detection';
        morphTopHat = imgSubtraction( backgroundSub.subtractionImg, ...
                                      morphOpen.openImg, ...
                                      figureName);
    end
    
%% Binarization
    
    if ~morphTopHat.error
        figureName='Binary image for points detection';
        pointBinary = binarization( morphTopHat.subtractionImg, ...
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

%% Connected components: streaks
    
    if ~remSalPep.error
        point = connectedComponentsPoints( remSalPep.remSaltPepperImg, ...
                                           I_borders);
    end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % % Median filters
% % 
% %     if ~gaussFilter.error
% %         littleKerlen=[21 21];%[3,3];
% %         bigKerlen=[15,15];%[21,21];
% %         madianBig = medianFilters( Img_input, littleKerlen);%, bigKerlen);
% %     end
% %    
% % % Background subtraction
% % 
% %     if ~madian.error
% %         if backgroundSubtraction
% %             figureName='Background subtraction image for points detection';
% %             backgroundSubBig = imgSubtraction( madianBig.medianImg, ...
% %                                             gaussFilter.blurImg, ...
% %                                             figureName);
% %         else
% %             backgroundSubBig.subtractionImg = madianBig.medianImg;
% %         end
% %     end
% % 
% % % Morphology opening
% % 
% %     if ~angle.error
% %         dimLine=40;%20
% %         morphOpenBig = morphologyOpen( backgroundSubBig.subtractionImg, ...
% %                                     dimLine, ...
% %                                     angle.tetaStreak);
% %     end
% % 
% % % Morphology TopHat
% %     
% %     if ~morphOpenBig.error
% %         figureName='Morphology TopHat for points detection';
% %         morphTopHatBig = imgSubtraction( backgroundSubBig.subtractionImg, ...
% %                                       morphOpenBig.openImg, ...
% %                                       figureName);
% %     end
% %     
% % % Binarization
% %     
% %     if ~morphTopHatBig.error
% %         figureName='Binary image for points detection';
% %         pointBinaryBig = binarization( morphTopHatBig.subtractionImg, ...
% %                                     differentThreshold, ...
% %                                     figureName);
% %     end
% %     
% % % Convolution kernel
% %     
% %     if ~pointBinaryBig.error
% %         k=ones(3);
% %         figureName='Convolution kernel for points detection';
% %         convPointBig = convolution( pointBinaryBig.binaryImg, ...
% %                                   k, ...
% %                                   sum(sum(k)), ...
% %                                   figureName);
% %     end
% %     
% % % Remove Salt and Pepper Noise from Image
% % 
% %     if ~convPointBig.error
% %         P=5;%3
% %         remSalPepBig = removeSaltPepper( convPointBig.convImg, P);
% %     end
% % 
% % % Connected components: streaks
% %     
% %     if ~remSalPepBig.error
% %         POINTSbig = connectedComponentsPoints( remSalPepBig.remSaltPepperImg, ...
% %                                            I_borders);
% %     end
% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
        if subtractPointsImg
            figureName='Points subtraction for streaks detection';
            streakLessPoint = imgSubtraction( streakBinary.binaryImg, ...
                                              dilatation.dilateImg, ...
                                              figureName);
        else
            streakLessPoint.error = 0;
            streakLessPoint.subtractionImg = streakBinary.binaryImg;
        end
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
            if(ELLIPSE)
                t = linspace(0,2*pi);
                for i=1:length(streaks.STREAKS(:,1))
                    X1 = streaks.majoraxis(i)*cos(t)/2;
                    Y1 = streaks.minoraxis(i)*sin(t)/2;
                    w= -streaks.orientation(i);
                    x = streaks.STREAKS(i,1) + X1*cosd(w) - Y1*sind(w);
                    y = streaks.STREAKS(i,2) + X1*sind(w) + Y1*cosd(w);
                    plot(x,y,'r')
                end
            end
        end
        if exist('STREAKSbig','var')
            plot(STREAKSbig(:,1),STREAKSbig(:,2),'*m')
        end
        %Plot points' centroids
        if isfield(point, 'POINTS')
            plot(point.POINTS(:,1),point.POINTS(:,2),'+g')
            if(ELLIPSE)
                t = linspace(0,2*pi);
                for i=1:length(point.POINTS(:,1))
                    X1 = point.majoraxis(i)*cos(t)/2;
                    Y1 = point.minoraxis(i)*sin(t)/2;
                    w= -point.orientation(i);
                    x = point.POINTS(i,1) + X1*cosd(w) - Y1*sind(w);
                    y = point.POINTS(i,2) + X1*sind(w) + Y1*cosd(w);
                    plot(x,y,'g')
                end
            end
        end
        if exist('POINTSbig','var')
            if isfield(POINTSbig, 'POINTS')
                plot(POINTSbig.POINTS(:,1),POINTSbig.POINTS(:,2),'+g')
                if(ELLIPSE)
                    t = linspace(0,2*pi);
                    for i=1:length(POINTSbig.POINTS(:,1))
                        X1 = POINTSbig.majoraxis(i)*cos(t)/2;
                        Y1 = POINTSbig.minoraxis(i)*sin(t)/2;
                        w= -POINTSbig.orientation(i);
                        x = POINTSbig.POINTS(i,1) + X1*cosd(w) - Y1*sind(w);
                        y = POINTSbig.POINTS(i,2) + X1*sind(w) + Y1*cosd(w);
                        plot(x,y,'g')
                    end
                end
            end
        end


    end
    nameFig=strcat(name,'.fig');
    pathFig=fullfile(resultDir,nameFig);
    if (FIGURE)
        saveas(h,pathFig);
    end
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
                            resultDir subtractPointsImg ELLIPSE
    end    
end

if FILE
    nameMatrix='computationTime.mat';
    pathMatrix=fullfile(resultDir,nameMatrix);
    save(pathMatrix,'timeTOT');
end

fprintf('\nEnd process.\n');
