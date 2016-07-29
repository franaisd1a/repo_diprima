clear all; 
% close all; 
clc; 

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

%% Input Folder

% wrkDir='D:\Dottorato\Space debris image\HAMR-14_15-05-2013\Foto\Foto 14-05-2013';
wrkDir=pwd;
fit_extension='*.fit';
jpg_extension='*.jpg';
fit_format='.fit';
jpg_format='.jpg';

if FILE     %Lettura da cartella
    if FIT
        directory=fullfile(wrkDir,fit_extension);
    else
        directory=fullfile(wrkDir,jpg_extension);
    end
    files=dir(directory);
    
else        %Lettura singolo file
    files=1;
    name_file='hamr_101.jpg';%hamr_186 %picture
end

for file_number=1:length(files)
    
    t_start=tic;
    
%% Strart processing

    if FILE
        name_file=files(file_number,1).name;
    end
    
    fprintf('File name: %s\n', name_file);
    
    [pathstr,name,ext] = fileparts(name_file);
    
    if ext==fit_format
        FIT=1;
    else
        FIT=0;
    end
    
    if FIT
        Img_input = (fitsread(name_file));%.*(2^8)./(2^16));
        % [data_img,map] = imread(image_fit);
        info = fitsinfo(name_file);
        fits_info = imfinfo(name_file);
    else
        Img_input=imread(name_file);
    end
    I_input_size=size(Img_input);
    I_borders=[ceil(0.005.*I_input_size), floor(0.995.*I_input_size)];
    
    if(FIGURE)
        a=100;
        if FILE
            a=a+file_number;
            figure(a);
            imshow(Img_input);
        else
            figure(a);%'name','Input file');
            imshow(Img_input);
        end
    end

% ======================================================================= %
%% Points detection
% ======================================================================= %

%% Gaussian filter

    tGaussianFilter=tic;
    
    hsize=[100 100];%[100 100];
    sigma=30;%10 25
    h = fspecial('gaussian', hsize, sigma);
    Iblur1 = imfilter(Img_input,h);
    
    tElapsedGaussianFilter=toc(tGaussianFilter);
    fprintf('Gaussian filter time: %d\n', tElapsedGaussianFilter);
    
    if(FIGURE_1)
        figure('name','Gaussain filter');
        imshow(Iblur1);
    end
    
    if CLEAR
        clear h;
    end

%% Median filters
    
    tMedianFilter=tic;
    
    littleKerlen=[3,3];
    bigKerlen=[21,21];
    
    littleMedianImg = medfilt2(Img_input, littleKerlen);
    bigMedianImg = medfilt2(Img_input, bigKerlen);
    
    medianImg = littleMedianImg - bigMedianImg;
    
    tElapsedMedianFilter=toc(tMedianFilter);
    fprintf('Median filter time: %d\n', tElapsedMedianFilter);
    
    if(FIGURE_1)
        figure('name','Median filter');
        imshow(medianImg);
        figure('name','Little median filter');
        imshow(littleMedianImg);
        figure('name','Big median filter');
        imshow(bigMedianImg);
    end
    
    if CLEAR
        clear h;
    end
    
%% Hough transform
%Calcolo dell'inclinazione delle strisciate per applicazione di filtro
%morfologico
    
    tHough=tic;
    [H1,T1,R1] = hough(medianImg,'RhoResolution',0.5,'ThetaResolution',0.5);%Img_input
    
    %P  = houghpeaks(H,5,'threshold',ceil(0.3*max(H(:))));
    P1  = houghpeaks(H1,1,'threshold',ceil(0.9*max(H1(:))),'NHoodSize',[31 31]);
    x1 = T1(P1(:,2));
    teta_streak=x1+90;
    
    tElapsedHough=toc(tHough);
    fprintf('Hough transform time: %d\n', tElapsedHough);
    
    if CLEAR
        clear H1 T1 R1;
    end
    
%% Morphology opening
%Apertura morfologica con kernel rettangolare inclinato dell'angolo di
%inclinazione delle strisciate
%Preserva le strisciate
    
    tMorphologyOpening=tic;
    %Calcolare dalla trasformata di Hough la lunghezza della strisciata
    dimLine=20;
    seR = strel('line', dimLine, -teta_streak);
    iopen=imopen(medianImg,seR);%Img_input
    
    tElapsedMorphologyOpening=toc(tMorphologyOpening);
    fprintf('Morphology opening time: %d\n', tElapsedMorphologyOpening);
    
    % iclose=imclose(I_input,se);
    if(FIGURE_1)
        figure('name','Morphology opening for points detection');
        imshow(iopen);
    end
    % imshow(iclose);
    
    if CLEAR
        clear seR;
    end
    
%% Morphology TopHat
%Elimina le strisciate
    tMorphologyTopHat=tic;
    TopHat=medianImg-iopen;%Img_input
    
    tElapsedMorphologyTopHat=toc(tMorphologyTopHat);
    fprintf('Morphology TopHat time: %d\n', tElapsedMorphologyTopHat);
    
    if(FIGURE_1)
        figure('name','Morphology TopHat for points detection');
        imshow(TopHat);
    end    
    
    if CLEAR
        %clear iopen;
    end
    
%% Background subtraction
    
    tBackgroundSubtraction=tic;
    
    if backgroundSubtraction
        isottr=TopHat-Iblur1;
    else
        isottr=TopHat;
    end
    
    tElapsedBackgroundSubtraction=toc(tBackgroundSubtraction);
    fprintf('Background subtraction time: %d\n', tElapsedBackgroundSubtraction);
    
    if(FIGURE_1)
        figure('name','Background subtraction image for points detection');
        imshow(isottr);
    end
    
    if CLEAR
        clear TopHat;
    end
    
%% Binarization
%Apply threshold to binarize the image
    
    tBinarization=tic;
    
    level=zeros(5,1);
    EM=zeros(5,1);
    [level(1) EM(1)] = graythresh(isottr(1:end/2,1:end/2));
    [level(2) EM(2)] = graythresh(isottr(1:end/2,end/2:end));
    [level(3) EM(3)] = graythresh(isottr(end/2:end,1:end/2));
    [level(4) EM(4)] = graythresh(isottr(end/2:end,end/2:end));
    [level(5) EM(5)] = graythresh(isottr);
    
    BW=zeros(I_input_size(1),I_input_size(2));
    
    if(differentThreshold)
        BW(1:end/2,1:end/2) = im2bw(isottr(1:end/2,1:end/2), level(1));
        BW(1:end/2,end/2:end) = im2bw(isottr(1:end/2,end/2:end), level(2));
        BW(end/2:end,1:end/2) = im2bw(isottr(end/2:end,1:end/2), level(3));
        BW(end/2:end,end/2:end) = im2bw(isottr(end/2:end,end/2:end), level(4));
    else
        maxLevel=max(level);
        BW = im2bw(isottr, maxLevel);
    end
    
    tElapsedBinarization=toc(tBinarization);
    fprintf('Binarization time: %d\n', tElapsedBinarization);
    
    if(FIGURE_1)
        figure('name','Binary image for points detection');
        imshow(BW);
    end
    
    if CLEAR
        clear isottr;
    end
    
%% Convolution kernel
    
    tCovolutionKernel=tic;
    
    k=ones(3);
    BWconv = conv2(im2uint8(BW)./255,k);
    sizeBWconv=size(BWconv);
    filterI=zeros(I_input_size(1),I_input_size(2));
    diffSize=floor(size(k)/2);
    startElement=1+diffSize;
    endElement=sizeBWconv-diffSize;
    for i=startElement(1):endElement(1)
        for j=startElement(2):endElement(2)
            if(BWconv(i,j)==sum(sum(k)))%>=sum(sum(k))
                filterI(i-1,j-1)=1;
            else
                filterI(i-1,j-1)=0;
            end
        end
    end
    
    tElapsedCovolutionKernel=toc(tCovolutionKernel);
    fprintf('Covolution kernel time: %d\n', tElapsedCovolutionKernel);
    
    if(FIGURE_1)
        figure('name','Convolution kernel for points detection');
        imshow(filterI);
    end
    
    if CLEAR
        clear BW BWconv;
    end
    
%% Remove Salt and Pepper Noise from Image
    
    tSaltPepper=tic;
    
    % B = medfilt2(BW);
    P=5;%3
    conn=8;
    BWlessSP = bwareaopen(filterI,P,conn);
    
    tElapsedSaltPepper=toc(tSaltPepper);
    fprintf('Remove Salt and Pepper time: %d\n', tElapsedSaltPepper);
    
    if(FIGURE_1)
        figure('name','Bynari image less salt and pepper noise for points detection');
        imshow(BWlessSP);
    end
    
    if CLEAR
        clear filterI;
    end
    
%% Connected components: points
    
    tConnComp=tic;
    
    CCpoints = bwconncomp(BWlessSP);
    statsP = regionprops(CCpoints,'Centroid','Area','Eccentricity','MajorAxisLength','MinorAxisLength','Orientation');
    max_points_diameter=0;
    min_points_diameter=max(I_input_size);
    
    if (length(statsP)~=0)
        points       = zeros(length(statsP),1);
        centroidP    = zeros(length(statsP),2);
        areaP        = zeros(length(statsP),1);
        eccentricityP= zeros(length(statsP),1);
        majoraxisP   = zeros(length(statsP),1);
        minoraxisP   = zeros(length(statsP),1);
        orientationP = zeros(length(statsP),1);
        
        for i=1:length(statsP)
            centroidP(i,:) = round(statsP(i).Centroid);
            if((centroidP(i,2)>I_borders(1) && centroidP(i,2)<I_borders(3)) ...
               && (centroidP(i,1)>I_borders(2) && centroidP(i,1)<I_borders(4)))
                areaP(i,:)            = statsP(i).Area;
                eccentricityP(i,:)    = statsP(i).Eccentricity;
                majoraxisP(i,:)       = statsP(i).MajorAxisLength;
                minoraxisP(i,:)       = statsP(i).MinorAxisLength;
                orientationP(i,:)     = statsP(i).Orientation;
% Identify points
                if (majoraxisP(i)/minoraxisP(i)<1.6)%1.6 %mettere condizione di punto se circolare
                    points(i)=1;
                    if(majoraxisP(i,:)>max_points_diameter)
                        max_points_diameter=majoraxisP(i,:);
                    end
                    if(minoraxisP(i,:)<min_points_diameter)
                        min_points_diameter=minoraxisP(i,:);
                    end
                end
            else
            end
        end
        
        n_points  = sum(points);
        if(n_points)
            noise=find(majoraxisP<ceil(max_points_diameter/2));%Per eliminare i punti piccoli
            %noise=find(minoraxisP<ceil(max_streaks_minoraxis/2));
            points(noise,:)          = [];
            centroidP(noise,:)       = [];
            areaP(noise,:)           = [];
            eccentricityP(noise,:)   = [];
            majoraxisP(noise,:)      = [];
            minoraxisP(noise,:)      = [];
            orientationP(noise,:)    = [];
            
            max_dim_array=length(round(centroidP(find(points==1),1)));
            POINTS=zeros(max_dim_array,3);
            POINTS =[centroidP(find(points==1),1) , centroidP(find(points==1),2) , sub2ind(I_input_size, centroidP(find(points ==1),2), centroidP(find(points ==1),1))];
        end
    end
    
    tElapsedConnComp=toc(tConnComp);
    fprintf('Connected components points time: %d\n', tElapsedConnComp);
    
% ======================================================================= %
%% Streaks detection
% ======================================================================= %

%% Binarization of the morphological opening image for streaks detection
%Apply threshold to binarize the image
    
    tBinarization2=tic;
    
    BWstreak = zeros(I_input_size(1),I_input_size(2));
    
    if(~differentThreshold)
        BWstreak(1:end/2,1:end/2) = im2bw(iopen(1:end/2,1:end/2), level(1));
        BWstreak(1:end/2,end/2:end) = im2bw(iopen(1:end/2,end/2:end), level(2));
        BWstreak(end/2:end,1:end/2) = im2bw(iopen(end/2:end,1:end/2), level(3));
        BWstreak(end/2:end,end/2:end) = im2bw(iopen(end/2:end,end/2:end), level(4));
    else
        BWstreak = im2bw(iopen, maxLevel);
    end
    
    tElapsedBinarization2=toc(tBinarization2);
    fprintf('Binarization for streaks detection time: %d\n', tElapsedBinarization2);
    
    if(FIGURE_1)
        figure('name','Binary image for streaks detection');
        imshow(BWstreak);
    end
    
    if CLEAR
        clear iopen;
    end
    
%% Morphology dilation
%Dilatazione morfologica con kernel 
%Probabilmente è inutile
    dilate=0;
    if dilate
        tMorphologyDilation = tic;
        seMask = strel('disk',ceil(min_points_diameter/2));
        
        idilate=imdilate(BWlessSP,seMask);
        
        tElapsedMorphologyDilation=toc(tMorphologyDilation);
        fprintf('Morphology dilation time: %d\n', tElapsedMorphologyDilation);
        
        if(FIGURE_1)
            figure('name','Morphology dilation');
            imshow(idilate);
        end

        if CLEAR
            clear seMask;
        end
    else
        idilate=BWlessSP;
    end
    
%% Points subtraction
%Probabilmente è inutile
    tPointsSubtraction=tic;

    BWstreakLessPoint=BWstreak-idilate;

    tElapsedPointsSubtraction=toc(tPointsSubtraction);
    fprintf('Points subtraction time: %d\n', tElapsedPointsSubtraction);

    if(FIGURE_1)
        figure('name','Points subtraction for streaks detection');
        imshow(BWstreakLessPoint);
    end

%% Convolution kernel for streaks detection 
%Probabilmente è inutile
    tCovolutionKernel2=tic;
    
    k=ones(1,21);
    %Aggiungere inclinazione della strisciata
    BWstreakConv = conv2(im2uint8(BWstreakLessPoint)./255,k);
    sizeBWconv=size(BWstreakConv);
    filterIstreak=zeros(I_input_size(1),I_input_size(2));
    diffSize=floor(size(k)/2);
    startElement=1+diffSize;
    endElement=sizeBWconv-diffSize;
    for i=startElement(1):endElement(1)
        for j=startElement(2):endElement(2)
            if(BWstreakConv(i,j)>=9)
                filterIstreak(i-diffSize(1),j-diffSize(2))=1;
            else
                filterIstreak(i-diffSize(1),j-diffSize(2))=0;
            end
        end        
    end
    
    tElapsedCovolutionKernel2=toc(tCovolutionKernel2);
    fprintf('Covolution kernel for streak detection time: %d\n', tElapsedCovolutionKernel2);
    
    if(FIGURE_1)
        figure('name','Convolution kernel for points detection');
        imshow(filterIstreak);
    end
    
    if CLEAR
        clear BW BWconv;
    end
    
%% Connected components: streaks
    
    CCstreaks = bwconncomp(filterIstreak);
    PixelIdxListStreaks = CCstreaks.PixelIdxList;
    stats = regionprops(CCstreaks,'Centroid','Area','Eccentricity','MajorAxisLength','MinorAxisLength','Orientation');
    min_streaks_minoraxis=max(I_input_size);
    max_streaks_minoraxis=0;
    max_streaks_majoraxis=0;
    
    if (length(stats)~=0)
        streaks     = zeros(length(stats),1);
        centroid    = zeros(length(stats),2);
        area        = zeros(length(stats),1);
        eccentricity= zeros(length(stats),1);
        majoraxis   = zeros(length(stats),1);
        minoraxis   = zeros(length(stats),1);
        axisratio   = zeros(length(stats),2);
        orientation = zeros(length(stats),1);
        
        for i=1:length(stats)
            centroid(i,:) = round(stats(i).Centroid);
            if((centroid(i,2)>I_borders(1) && centroid(i,2)<I_borders(3)) ...
                    && (centroid(i,1)>I_borders(2) && centroid(i,1)<I_borders(4)))
                area(i,:)            = stats(i).Area;
                eccentricity(i,:)    = stats(i).Eccentricity;
                majoraxis(i,:)       = stats(i).MajorAxisLength;
                minoraxis(i,:)       = stats(i).MinorAxisLength;
                axisratio(i,:)       = [majoraxis(i)/minoraxis(i),i];
                orientation(i,:)     = stats(i).Orientation;
                
% Identify streaks
                
                if(majoraxis(i)/minoraxis(i)>6)%eccentricity(i,:)>0.9)
                    streaks(i)=1;
                    if(min_streaks_minoraxis>minoraxis(i,:))
                        min_streaks_minoraxis=minoraxis(i,:);
                    end
                    if(max_streaks_minoraxis<minoraxis(i,:))
                        max_streaks_minoraxis=minoraxis(i,:);
                    end
                    if(max_streaks_majoraxis<majoraxis(i,:))
                        max_streaks_majoraxis=majoraxis(i,:);
                    end
                else
                end
            else
            end
        end
        n_streaks = sum(streaks);
        if(n_streaks)%==0)
            %             min_streaks_minoraxis=min_points_diameter/2;%max_points_diameter
            %         end
            if(n_streaks>1)
                noiseThin=find(minoraxis<ceil(min_streaks_minoraxis));%Per eliminare le strisciate sottili
                %noiseThin=find(minoraxis<ceil(max_streaks_minoraxis/2));
            else
                min_streaks_minoraxis=2;
                noiseThin=find(minoraxis<ceil(min_streaks_minoraxis));%min_streaks_minoraxis);
            end
            streaks(noiseThin,:)        = [];
            centroid(noiseThin,:)       = [];
            area(noiseThin,:)           = [];
            eccentricity(noiseThin,:)   = [];
            majoraxis(noiseThin,:)      = [];
            minoraxis(noiseThin,:)      = [];
            axisratio(noiseThin,:)      = [];
            orientation(noiseThin,:)    = [];
            PixelIdxListStreaks(noiseThin) = [];
            stats(noiseThin,:)          = [];
            
            noiseShort=find(majoraxis<ceil(max_streaks_majoraxis/2));%Per eliminare le strisciate corte
            streaks(noiseShort,:)        = [];
            centroid(noiseShort,:)       = [];
            area(noiseShort,:)           = [];
            eccentricity(noiseShort,:)   = [];
            majoraxis(noiseShort,:)      = [];
            minoraxis(noiseShort,:)      = [];
            axisratio(noiseShort,:)      = [];
            orientation(noiseShort,:)    = [];
            PixelIdxListStreaks(noiseShort) = [];
            stats(noiseShort,:)          = [];
            
            max_dim_array=max(length(centroid(find(streaks==1),1)));
            STREAKS=zeros(max_dim_array,3);
            STREAKS=[centroid(find(streaks==1),1) , centroid(find(streaks==1),2) , sub2ind(I_input_size, centroid(find(streaks==1),2), centroid(find(streaks==1),1))];
        end
    end
    
%Delete points on streak
    if exist('POINTS')
        if exist('STREAKS')
            for j=1:length(STREAKS(:,1))
                for i=1:length(POINTS(:,1))
                    if(find(PixelIdxListStreaks{1,j}==POINTS(i,3)))
                        POINTS(i,3)=-1;
                    end
                end
            end
            noisePoint=find(POINTS(:,3)<0);
            POINTS(noisePoint,:)          = [];
            points(noisePoint,:)          = [];
            centroidP(noisePoint,:)       = [];
            areaP(noisePoint,:)           = [];
            eccentricityP(noisePoint,:)   = [];
            majoraxisP(noisePoint,:)      = [];
            minoraxisP(noisePoint,:)      = [];
            orientationP(noisePoint,:)    = [];
        end
    end

%% Search big objects
    
    bigObjects = littleMedianImg-Iblur1;
    bigObjectsBW = im2bw(bigObjects, maxLevel);
    pointStreak = BWlessSP + filterIstreak;
    onlyBig = bigObjectsBW - pointStreak;
    
%% Convolution kernel for big objects
    
    tCovolutionKernelBIG=tic;
    
    k=ones(3);
    onlyBigconv = conv2(im2uint8(onlyBig)./255,k);
    sizeBWconv=size(onlyBigconv);
    bigObj=zeros(I_input_size(1),I_input_size(2));
    diffSize=floor(size(k)/2);
    startElement=1+diffSize;
    endElement=sizeBWconv-diffSize;
    for i=startElement(1):endElement(1)
        for j=startElement(2):endElement(2)
            if(onlyBigconv(i,j)==sum(sum(k)))%>=sum(sum(k))
                bigObj(i-1,j-1)=1;
            else
                bigObj(i-1,j-1)=0;
            end
        end
    end
    
    tElapsedCovolutionKernelBIG=toc(tCovolutionKernelBIG);
    fprintf('Covolution kernel for big objects time: %d\n', tElapsedCovolutionKernelBIG);
    
    if(FIGURE_1)
        figure('name','Convolution kernel for big objects');
        imshow(bigObj);
    end
    
    if CLEAR
        clear BW BWconv;
    end
    
%% Connected components: big objects
    if(0)
    CCbigObj = bwconncomp(bigObj);
    PixelIdxListBigObj = CCbigObj.PixelIdxList;
    statsBigObj = regionprops(CCbigObj,'Centroid','Area','Eccentricity','MajorAxisLength','MinorAxisLength','Orientation');
    
    if (length(statsBigObj)~=0)
        pointsBig         = zeros(length(statsBigObj),1);
        streaksBig        = zeros(length(statsBigObj),1);
        centroidBigObj    = zeros(length(statsBigObj),2);
        areaBigObj        = zeros(length(statsBigObj),1);
        majoraxisBigObj   = zeros(length(statsBigObj),1);
        minoraxisBigObj   = zeros(length(statsBigObj),1);
        
        for i=1:length(statsBigObj)
            centroidBigObj(i,:) = round(statsBigObj(i).Centroid);
            if((centroidBigObj(i,2)>I_borders(1) && centroidBigObj(i,2)<I_borders(3)) ...
                    && (centroidBigObj(i,1)>I_borders(2) && centroidBigObj(i,1)<I_borders(4)))
                areaBigObj(i,:)            = statsBigObj(i).Area;
                majoraxisBigObj(i,:)       = statsBigObj(i).MajorAxisLength;
                minoraxisBigObj(i,:)       = statsBigObj(i).MinorAxisLength;
                
                if(majoraxisBigObj(i)/minoraxisBigObj(i)>6)%eccentricity(i,:)>0.9)
                    streaksBig(i)=1;
                elseif (majoraxisBigObj(i)/minoraxisBigObj(i)<1.6)
                    if(majoraxisBigObj(i)>max_points_diameter)
                        pointsBig(i)=1;
                    end
                else
                end
            else
            end
        end
        max_dim_arrayBigSTREAKS=max(length(centroidBigObj(find(streaksBig==1),1)));
        STREAKSbig=zeros(max_dim_arrayBigSTREAKS,3);
        STREAKSbig=[centroidBigObj(find(streaksBig==1),1) , centroidBigObj(find(streaksBig==1),2) , sub2ind(I_input_size, centroidBigObj(find(streaksBig==1),2), centroidBigObj(find(streaksBig==1),1))];
        max_dim_arrayBigPOINTS=length(round(centroidBigObj(find(pointsBig==1),1)));
        POINTSbig=zeros(max_dim_arrayBigPOINTS,3);
        POINTSbig=[centroidBigObj(find(pointsBig==1),1) , centroidBigObj(find(pointsBig==1),2) , sub2ind(I_input_size, centroidBigObj(find(pointsBig ==1),2), centroidBigObj(find(pointsBig ==1),1))];
    end
    end
    
%% ROI selection for centroid enhancement
    
%ROI selection around points
    
    if exist('POINTS')
        pointROI=cell(3,length(POINTS(:,1)));
        %pointROIdimension=round(max_points_diameter);
        expROIdimension=ceil(log2(round(max_points_diameter)));
        pointROIdimension=(2^expROIdimension);
        for i=1:length(POINTS(:,1))
            limInfX=POINTS(i,2)-pointROIdimension;
            limSupX=POINTS(i,2)+pointROIdimension-1;
            if(limInfX<0)
                limInfX=1;
                limSupX=limInfX+2*pointROIdimension-1;
            end
            if(limSupX>I_input_size(1))
                limInfX=I_input_size(1)-2*pointROIdimension+1;
                limSupX=I_input_size(1);
            end
            
            limInfY=POINTS(i,1)-pointROIdimension;
            limSupY=POINTS(i,1)+pointROIdimension-1;
            if(limInfY<0)
                limInfY=1;
                limSupY=limInfY+2*pointROIdimension-1;
            end
            if(limSupY>I_input_size(2))
                limInfY=I_input_size(2)-2*pointROIdimension+1;
                limSupY=I_input_size(2);
            end
            
            %prendere area pari a potenza di 2
            
            pointROI{1,i} = Img_input(limInfX:limSupX-1, limInfY:limSupY-1);
            pointROI{2,i} = [limInfX,limInfY , limSupX-1,limSupY-1];
            if(FIGURE_1)
                c=400;
                c=c+i;
                figure(c);
                imshow(pointROI{1,i});
            end
        end
    end
    
%ROI selection around streaks

    if exist('STREAKS')
        streakROI=cell(3,length(STREAKS(:,1)));
        
        expSTREAKdimensionX=ceil(log2(round(max_streaks_majoraxis/2)));
        streakROIdimensionX=(2^expSTREAKdimensionX);
        expSTREAKdimensionY=ceil(log2(round(max_streaks_minoraxis/2)));
        streakROIdimensionY=(2^expSTREAKdimensionY);
        
%         streakROIdimensionX=round(max_streaks_majoraxis*0.6);
%         streakROIdimensionY=round(max_streaks_minoraxis);
        for i=1:length(STREAKS(:,1))
            limInfXs=STREAKS(i,2)-streakROIdimensionY;
            limSupXs=STREAKS(i,2)+streakROIdimensionY-1;
            if(limInfXs<1)
                limInfXs=1;
                limSupXs=limInfXs+2*streakROIdimensionY-1;
            end
            if(limSupXs>I_input_size(1))
                limInfXs=I_input_size(1)-2*streakROIdimensionY+1;
                limSupXs=I_input_size(1);
            end
            
            limInfYs=STREAKS(i,1)-streakROIdimensionX;
            limSupYs=STREAKS(i,1)+streakROIdimensionX-1;
            if(limInfYs<1)
                limInfYs=1;
                limSupYs=limInfYs+2*streakROIdimensionX-1;
            end
            if(limSupYs>I_input_size(2))
                limInfYs=I_input_size(2)-2*streakROIdimensionX+1;
                limSupYs=I_input_size(2);
            end
            
            %prendere area pari a potenza di 2
            
            streakROI{1,i} = Img_input(limInfXs:limSupXs, limInfYs:limSupYs);
            streakROI{2,i} = [limInfXs,limInfYs , limSupXs,limSupYs];
            fprintf('Streak %d center %d %d\n', i, STREAKS(i,1), STREAKS(i,2));
            if(FIGURE_1)
                b=500;
                b=b+i;
                figure(b);
                imshow(streakROI{1,i});
            end
        end
    end

%% FFT Zero Padding Interpolator

    % Interpolation factor 
    F=2;
    
    if exist('POINTS')
        for i=1:length(POINTS(:,1))
            pointROI{3,i} = fftZeroPaddingInterpolation( pointROI{1,i}, F);
        end
    end
    
    if exist('STREAKS')
        for i=1:length(STREAKS(:,1))
            streakROI{3,i} = fftZeroPaddingInterpolation( streakROI{1,i}, F);
        end
    end
    
%% Precise centroid position

    if exist('POINTS')
        for i=1:length(POINTS(:,1))
%             trova il centroide
        end
    end
    
    if exist('STREAKS')
        for i=1:length(STREAKS(:,1))
%             trova il centroide
        end
    end


%% Plot of streaks and points    
    
    if(FIGURE)
        figure(a);
        hold on;
        % Plot streaks' centroids
        if exist('STREAKS')
            plot(STREAKS(:,1),STREAKS(:,2),'*r')
        end
        if exist('STREAKSbig')
            plot(STREAKSbig(:,1),STREAKSbig(:,2),'*m')
        end
        %Plot points' centroids
        if exist('POINTS')
            plot(POINTS(:,1),POINTS(:,2),'+g')
        end
        if exist('POINTSbig')
            plot(POINTSbig(:,1),POINTSbig(:,2),'+b')
        end
    end
    
%% End of process
%Time calculation
    
    t_tot=toc(t_start);
    fprintf('Total time: %d sec\n', t_tot);
    if FILE
        timeTOT(file_number)=t_tot;
        
        clearvars -except   timeTOT FIGURE FIGURE_1 FILE CLEAR ...
                            backgroundSubtraction differentThreshold ...
                            files jpg_format fit_format
    end
end

