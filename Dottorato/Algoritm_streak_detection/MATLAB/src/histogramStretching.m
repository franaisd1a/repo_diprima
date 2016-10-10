function [ output ] = histogramStretching( varargin )

% Filter a N1 X N2 image using median filter
% creating an N1 X N2 image

global FIGURE_1
disp('Start histogramStretching function.')

%% Input

%  1) N1 X N2 image
%  2) 

%% Output

% Output struct with different fields:
% 1) output.error: boolean value, 1 is error
% 2) output.XXX: 

output={};
output.error=1;

%% Input validation

if nargin~=3
    disp('Error! Wrong number of parameters.')
    disp(sprintf('\n'));
    return
else
%     disp('Correct number of parameters.')
    img=varargin{1};
    outputByteDepth=varargin{2};
    percentile=varargin{3};
end

% *********************************************************************** %
%% Processing
% *********************************************************************** %
try
  
% ----------------------------------------------------------------------- %
%% Histogram
% ----------------------------------------------------------------------- %

    tStart=tic;
    
    imgSz=size(img);
    classType = class(img);
    
    if     strcmp('uint8' ,classType) ||  strcmp('int8' ,classType)
        color=2^8-1;
    elseif strcmp('uint16',classType) || strcmp('int16',classType)
        color=2^16-1;
    elseif strcmp('uint32',classType) || strcmp('int32',classType)
        color=2^32-1;
    elseif strcmp('uint64',classType) || strcmp('int64',classType)
        color=2^64-1;
    elseif strcmp('single',classType)
        color=2^16-1;
    elseif strcmp('double',classType)
        color=2^16-1;
    else
        output.error=1;
        disp('Error! Unsupported pixel type.')
        disp(sprintf('\n'));
        return
    end

% Compute the histogram

    if strcmp('double',classType)
        histogram=zeros(color+1,1);
        histogramXaxis=0:1:color;
        for i=1:imgSz(1)
            for j=1:imgSz(2)
                value=double(img(i,j))+1;
                histogram(value)=histogram(value)+1;
            end
        end
    else
        [histogram,histogramXaxis] = imhist(img,color);
    end
    
    maxHistValue=max(max(img));
    
    zeroValue=find(histogramXaxis>maxHistValue+1);
    histogram(zeroValue)=[];
    histogramXaxis(zeroValue)=[];
   
% ----------------------------------------------------------------------- %
%% Compute the cumulative distribution histogram function
% ----------------------------------------------------------------------- %

    cdfHistogram=zeros(histogram,1);
    cdfHistogramVV=zeros(histogram,1);
    sumCDF=0;
    for i=1:length(histogram)
        sumCDF = sumCDF + histogram(i);
        cdfHistogram(i) = sumCDF;
%         cdfHistogramVV(i)=sum(histogram(1:i));
    end
    
% ----------------------------------------------------------------------- %
%% Contrast Stretching
% ----------------------------------------------------------------------- %
    
    LUT=zeros(histogram,1);
    peakMax=max(histogram);
    grayColor=find(histogram==peakMax);
    
% Compute value in function of the max peak

    lowThresh=peakMax*percentile(1);
    highThresh=peakMax*percentile(2);
    
    for i=0:grayColor
        k=grayColor-i;
        if histogram(k)<lowThresh
            minValue=k;
            break;
        end
    end
        
    for i=grayColor:length(histogram)-1
        if histogram(i)<highThresh
            maxValue=i*1.05;%1.5
            break;
        end
    end
       
    scaleFactor=outputByteDepth/(maxValue-minValue);
    
    for i=1:length(histogram)
        if i<minValue
            LUT(i)=0;
        elseif i>maxValue
            LUT(i)=outputByteDepth;
        else
            LUT(i)=(i-minValue)*scaleFactor;
        end
    end
    
    for i=1:imgSz(1)
        for j=1:imgSz(2)
            output.stretchImg(i,j)= uint64(round(LUT(img(i,j)+1)));
        end
    end

    output.stretchImg=uint8(output.stretchImg);
        
    if(FIGURE_1)
        figure('name','Histogram and cumulative distribution function');
        title('Histogram and cumulative distribution function');
        %semilogy(histogramXaxis, histogram);
        loglog(histogramXaxis, histogram);
        hold on;
%         semilogy(histogramXaxis, cdfHistogram,'r');
        loglog(histogramXaxis, cdfHistogram,'r');
        xlabel('Grayscale value'); grid on;
        
%         figure('name','Stretch image');
%         imshow(output.stretchImg);
    end
    
    tElapsed = toc(tStart);    
    disp(sprintf('End histogramStretching funtion %d sec.', tElapsed));
    disp(sprintf('\n'));
    output.error=0;
    
% ??????????????????????????????????????????????????????????????????????? %
%% Error handling
% ??????????????????????????????????????????????????????????????????????? %  

catch ME
    output.error=1;
    disp('Error using histogramStretching function.');
    disp(ME.message);
    disp(sprintf('\n'));
    
    for i=1:length(ME.stack)
        disp(sprintf('Error in %s (line %d)', ME.stack(i,1).name, ME.stack(i,1).line));
        
        disp(sprintf('\n'));
    end
    
    %rethrow(ME);
end

end
