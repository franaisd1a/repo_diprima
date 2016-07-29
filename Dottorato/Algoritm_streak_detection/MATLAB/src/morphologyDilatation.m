function [ output ] = morphologyDilatation( varargin )

% Morphology dilatation on a N1 X N2 image with a disk kernel. 

global FIGURE FIGURE_1
disp('Start morphologyDilatation function.')
tStart=tic;

%% Input

%  1) N1 X N2 image

%% Output

% Output struct with different fields:
% 1) output.error: boolean value, 1 is error
% 2) output.dilateImg: morphology dilatation image

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
    dim=varargin{2};
end

% *********************************************************************** %
%% Processing
% *********************************************************************** %
try
  
% ----------------------------------------------------------------------- %
%% Hough transform
% ----------------------------------------------------------------------- %

    tStart=tic;
    
    seMask = strel('disk',ceil(dim));
        
    output.dilateImg=imdilate(img,seMask);

    if(FIGURE_1)
        figure('name','Morphology dilation');
        imshow(output.dilateImg);
    end
    
    tElapsed = toc(tStart);    
    disp(sprintf('End morphologyDilatation funtion %d sec.', tElapsed));
    disp(sprintf('\n'));
    output.error=0;
    
% ??????????????????????????????????????????????????????????????????????? %
%% Error handling
% ??????????????????????????????????????????????????????????????????????? %  

catch ME
    output.error=1;
    disp('Error using morphologyDilatation function.');
    disp(ME.message);
    disp(sprintf('\n'));
    
    for i=1:length(ME.stack)
        disp(sprintf('Error in %s (line %d)', ME.stack(i,1).name, ME.stack(i,1).line));
        
        disp(sprintf('\n'));
    end
    
    %rethrow(ME);
end

end

