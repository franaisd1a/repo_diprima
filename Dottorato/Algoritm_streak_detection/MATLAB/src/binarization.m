function [ output ] = binarization( varargin )

% Binarization on N1 X N2 images

global FIGURE_1
disp('Start binarization function.')

%% Input

%  1) N1 X N2 image

%% Output

% Output struct with different fields:
% 1) output.error: boolean value, 1 is error
% 2) output.binaryImg: binary image

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
    differentThreshold=varargin{2};
    figureName=varargin{3};
end

% *********************************************************************** %
%% Processing
% *********************************************************************** %
try
  
% ----------------------------------------------------------------------- %
%% Image Subtraction
% ----------------------------------------------------------------------- %

    tStart=tic;
    
    imgSz=size(img);
    output.binaryImg=zeros(imgSz(1),imgSz(2));
    
    level=zeros(5,1);
    
    level(1) = graythresh( img(1:end/2  ,   1:end/2) );
    level(2) = graythresh( img(1:end/2  , end/2:end) );
    level(3) = graythresh( img(end/2:end,   1:end/2) );
    level(4) = graythresh( img(end/2:end, end/2:end) );
    level(5) = graythresh( img                       );
    
%     level=level.*1.5;
    level
    
    meanLevel   = mean  (level);
    medianLevel = median(level);
    sortLevel   = sort  (level,'descend');
    
    minThreshold=sortLevel(2);
    if minThreshold<meanLevel
        minThreshold=meanLevel;
    end
    
    for i=1:length(level)
       if (level(i)<meanLevel || level(i)<medianLevel)
           level(i)=minThreshold;
       end
    end
    
    if(differentThreshold)
        output.binaryImg(1:end/2,1:end/2) = im2bw(img(1:end/2,1:end/2), level(1));
        output.binaryImg(1:end/2,end/2:end) = im2bw(img(1:end/2,end/2:end), level(2));
        output.binaryImg(end/2:end,1:end/2) = im2bw(img(end/2:end,1:end/2), level(3));
        output.binaryImg(end/2:end,end/2:end) = im2bw(img(end/2:end,end/2:end), level(4));
    else
        maxLevel=max(level);
        output.binaryImg = im2bw(img, maxLevel);
    end
    
    if(FIGURE_1)
        figure('name',figureName);
        imshow(output.binaryImg);
    end
    
    tElapsed = toc(tStart);    
    disp(sprintf('End binarization funtion %d sec.', tElapsed));
    disp(sprintf('\n'));
    output.error=0;
    
% ??????????????????????????????????????????????????????????????????????? %
%% Error handling
% ??????????????????????????????????????????????????????????????????????? %  

catch ME
    output.error=1;
    disp('Error using binarization function.');
    disp(ME.message);
    disp(sprintf('\n'));
    
    for i=1:length(ME.stack)
        disp(sprintf('Error in %s (line %d)', ME.stack(i,1).name, ME.stack(i,1).line));
        
        disp(sprintf('\n'));
    end
    
    %rethrow(ME);
end

end

