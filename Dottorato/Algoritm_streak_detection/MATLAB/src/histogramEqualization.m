function [ output ] = histogramEqualization( varargin )

% Filter a N1 X N2 image using median filter
% creating an N1 X N2 image

global FIGURE FIGURE_1
disp('Start histogramEqualization function.')
tStart=tic;

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
%% Histogram Equalization
% ----------------------------------------------------------------------- %

    tStart=tic;
    
    imgSz=size(img);
    numberPixel=imgSz(1)*imgSz(2);
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

%     [histogram,histogramXaxis] = imhist(img);

% Compute the normalized histogram
    histogram=zeros(color+1,1);
    histogramXaxis=0:1:color;
    for i=1:imgSz(1)
        for j=1:imgSz(2)
            value=double(img(i,j))+1;
            histogram(value)=histogram(value)+1;        
        end
        a=1;
    end
%     histogramNorm=histogram./numberPixel;
    
% Compute the cumulative distribution histogram function

    cdfHistogram=zeros(color+1,1);
%     cdfHistogramNorm=zeros(color+1,1);
    for i=1:length(histogram)
        cdfHistogram(i)=sum(histogram(1:i));
%         cdfHistogramNorm(i)=sum(histogramNorm(1:i));
    end
    
    LUT=zeros(color+1,1);
    colorRange=max(max(img))-min(min(img));
    lowThresh=colorRange*percentile(1);%numberPixel
    highThresh=colorRange*percentile(2);%numberPixel
    
    
    minValue=lowThresh;%0;
    maxValue=highThresh;%color;

%     for i=1:length(histogram)
%         if histogram(i)>lowThresh    %cdfHistogram
%             minValue=i-1;%forse serve il meno 1
%             break;
%         end
%     end
%     
%     for i=0:length(histogram)-1
%         k=color+1-i;
%         if histogram(k)<highThresh    %cdfHistogram
%             maxValue=k+1;
%             break;
%         end
%     end
    
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
            output.histEqImg(i,j)= round(LUT(img(i,j)+1));
        end
    end

    output.histEqImg=uint8(output.histEqImg);
    
    if(FIGURE)
        figure('name','Histogram and cumulative distribution function');
        title('Histogram and cumulative distribution function');
        plot(histogramXaxis,histogram); grid on;
%         title('Histogram'); xlabel('Grayscale value');
        hold on;
        plot(histogramXaxis,cdfHistogram,'r');
        
        figure('name','Histogram Matlab');
        imhist(img);
        figure('name','Stretch image');
        imshow(output.histEqImg);
    end
    
    tElapsed = toc(tStart);    
    disp(sprintf('End histogramEqualization funtion %d sec.', tElapsed));
    disp(sprintf('\n'));
    output.error=0;
    
% ??????????????????????????????????????????????????????????????????????? %
%% Error handling
% ??????????????????????????????????????????????????????????????????????? %  

catch ME
    output.error=1;
    disp('Error using histogramEqualization function.');
    disp(ME.message);
    disp(sprintf('\n'));
    
    for i=1:length(ME.stack)
        disp(sprintf('Error in %s (line %d)', ME.stack(i,1).name, ME.stack(i,1).line));
        
        disp(sprintf('\n'));
    end
    
    %rethrow(ME);
end

end



% Compute the normalized histogram
%     histogram=zeros(color+1,1);
%     histogramXaxis=0:1:color;
%     for i=1:imgSz(1)
%         for j=1:imgSz(2)
%             value=double(img(i,j))+1;
%             histogram(value)=histogram(value)+1;        
%         end
%     end
%         figure('name','Cumulative distribution histogram function');
%         plot(histogramXaxis,cdfHistogram); grid on;
%         title('Cumulative distribution histogram function');
%         xlabel('Grayscale value');



% Histogram equalization
%     output.histEqImg=zeros(imgSz(1),imgSz(2));
%     for i=1:imgSz(1)
%         for j=1:imgSz(2)
%             value=double(img(i,j))+1;
%             output.histEqImg(i,j)= ...
%                 round( (cdfHistogram(value)-min(cdfHistogram))*color ...
%                 /(numberPixel-min(cdfHistogram)));        
%         end
%     end
%         figure('name','Histogram equalization');
%         imshow(output.histEqImg); 
%         title('Histogram equalization');



%     if     strcmp('uint8' ,classType) ||  strcmp('int8' ,classType)
%         histogram=uint64(histogram);
%     elseif strcmp('uint16',classType) || strcmp('int16',classType)
%         histogram=uint64(histogram);
%     elseif strcmp('uint32',classType) || strcmp('int32',classType)
%         histogram=uint64(histogram);
%     elseif strcmp('uint64',classType) || strcmp('int64',classType)
%         histogram=uint64(histogram);
%     else
%         output.error=1;
%         disp('Error! Unsupported pixel type.')
%         disp(sprintf('\n'));
%         return
%     end