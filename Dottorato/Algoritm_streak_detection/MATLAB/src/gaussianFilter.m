function [ output ] = gaussianFilter( varargin )

% Filter a N1 X N2 image using Gaussian lowpass filter
% creating an N1 X N2 image

global FIGURE_1
disp('Start gaussianFilter function.')

%% Input

%  1) N1 X N2 image
%  2) filter size
%  3) filter standard deviation

%% Output

% Output struct with different fields:
% 1) output.error: boolean value, 1 is error
% 2) output.blurImg: filtered image

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
    hsize=varargin{2};
    sigma=varargin{3};
end

% *********************************************************************** %
%% Processing
% *********************************************************************** %
try
  
% ----------------------------------------------------------------------- %
%% Gaussian filter
% ----------------------------------------------------------------------- %

    tStart=tic;
    
    h = fspecial('gaussian', hsize, sigma);
    output.blurImg = imfilter(img,h);
    
    if(FIGURE_1)
        figure('name','Gaussain filter');
        imshow(output.blurImg);
    end
    
    tElapsed = toc(tStart);    
    disp(sprintf('End gaussianFilter funtion %d sec.', tElapsed));
    disp(sprintf('\n'));
    output.error=0;
    
% ??????????????????????????????????????????????????????????????????????? %
%% Error handling
% ??????????????????????????????????????????????????????????????????????? %  

catch ME
    output.error=1;
    disp('Error using gaussianFilter function.');
    disp(ME.message);
    disp(sprintf('\n'));
    
    for i=1:length(ME.stack)
        disp(sprintf('Error in %s (line %d)', ME.stack(i,1).name, ME.stack(i,1).line));
        
        disp(sprintf('\n'));
    end
    
    %rethrow(ME);
end

end

