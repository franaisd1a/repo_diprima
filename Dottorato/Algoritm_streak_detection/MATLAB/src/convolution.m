function [ output ] = convolution( varargin )

% Convolution kernel on N1 X N2 images

global FIGURE_1
disp('Start convolution function.')

%% Input

%  1) N1 X N2 image
%  2) Kernel
%  3) Threshold
%  1) Figure name

%% Output

% Output struct with different fields:
% 1) output.error: boolean value, 1 is error
% 2) output.convImg: convolution image

output={};
output.error=1;

%% Input validation

if nargin~=4
    disp('Error! Wrong number of parameters.')
    disp(sprintf('\n'));
    return
else
%     disp('Correct number of parameters.')
    img=varargin{1};
    k=varargin{2};
    threshold=varargin{3};
    figureName=varargin{4};
end

% *********************************************************************** %
%% Processing
% *********************************************************************** %
try
  
% ----------------------------------------------------------------------- %
%% Convolution kernel
% ----------------------------------------------------------------------- %

    tStart=tic;
    
    imgSz=size(img);
    output.convImg=zeros(imgSz(1),imgSz(2));
    
    imgConv = conv2(im2uint8(img)./255,k);
    
    sizeImgConv=size(imgConv);
    
    diffSize=floor(size(k)/2);
    startElement=1+diffSize;
    endElement=sizeImgConv-diffSize;
    
    for i=startElement(1):endElement(1)
        for j=startElement(2):endElement(2)
            if(imgConv(i,j)>=threshold)
                output.convImg(i-diffSize(1),j-diffSize(2))=1;
            else
                output.convImg(i-diffSize(1),j-diffSize(2))=0;
            end
        end
    end
    
    if(FIGURE_1)
        figure('name',figureName);
        imshow(output.convImg);
    end
    
    tElapsed = toc(tStart);    
    disp(sprintf('End convolution funtion %d sec.', tElapsed));
    disp(sprintf('\n'));
    output.error=0;
    
% ??????????????????????????????????????????????????????????????????????? %
%% Error handling
% ??????????????????????????????????????????????????????????????????????? %  

catch ME
    output.error=1;
    disp('Error using convolution function.');
    disp(ME.message);
    disp(sprintf('\n'));
    
    for i=1:length(ME.stack)
        disp(sprintf('Error in %s (line %d)', ME.stack(i,1).name, ME.stack(i,1).line));
        
        disp(sprintf('\n'));
    end
    
    %rethrow(ME);
end

end

