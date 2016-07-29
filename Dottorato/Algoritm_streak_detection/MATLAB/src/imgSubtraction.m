function [ output ] = imgSubtraction( varargin )

% Image subtraction on two N1 X N2 images

global FIGURE_1
disp('Start imgSubtraction function.')

%% Input

%  1) N1 X N2 image

%% Output

% Output struct with different fields:
% 1) output.error: boolean value, 1 is error
% 2) output.subtractionImg: subtraction image

output={};
output.error=1;

%% Input validation

if nargin~=3
    disp('Error! Wrong number of parameters.')
    disp(sprintf('\n'));
    return
else
%     disp('Correct number of parameters.')
    imgA=varargin{1};
    imgB=varargin{2};
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
    
    output.subtractionImg=imgA-imgB;
    
    if(FIGURE_1)
        figure('name',figureName);
        imshow(output.subtractionImg);
    end  
    
    tElapsed = toc(tStart);    
    disp(sprintf('End imgSubtraction funtion %d sec.', tElapsed));
    disp(sprintf('\n'));
    output.error=0;
    
% ??????????????????????????????????????????????????????????????????????? %
%% Error handling
% ??????????????????????????????????????????????????????????????????????? %  

catch ME
    output.error=1;
    disp('Error using imgSubtraction function.');
    disp(ME.message);
    disp(sprintf('\n'));
    
    for i=1:length(ME.stack)
        disp(sprintf('Error in %s (line %d)', ME.stack(i,1).name, ME.stack(i,1).line));
        
        disp(sprintf('\n'));
    end
    
    %rethrow(ME);
end

end

