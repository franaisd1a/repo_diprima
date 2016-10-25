function [ output ] = morphologyOpen( varargin )

% Morphology opening on a N1 X N2 image with a rectangular kernel rotate of
% an angle. Delete noise and points object in the image and preserve the
% streaks

global FIGURE_1
disp('Start morphologyOpen function.')

%% Input

%  1) N1 X N2 image

%% Output

% Output struct with different fields:
% 1) output.error: boolean value, 1 is error
% 2) output.openImg: morphology opening image

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
    dimLine=varargin{2};
    teta_streak=varargin{3};
end

% *********************************************************************** %
%% Processing
% *********************************************************************** %
try
  
% ----------------------------------------------------------------------- %
%% Morphology Open
% ----------------------------------------------------------------------- %

    tStart=tic;
    
    seR = strel('line', dimLine, -teta_streak);
    output.openImg=imopen(img,seR);
    
    if(FIGURE_1)
        figure('name','Morphology opening with rectangular kernel');
        imshow(output.openImg);
    end
    
    tElapsed = toc(tStart);    
    disp(sprintf('End morphologyOpen funtion %d sec.', tElapsed));
    disp(sprintf('\n'));
    output.error=0;
    
% ??????????????????????????????????????????????????????????????????????? %
%% Error handling
% ??????????????????????????????????????????????????????????????????????? %  

catch ME
    output.error=1;
    disp('Error using morphologyOpen function.');
    disp(ME.message);
    disp(sprintf('\n'));
    
    for i=1:length(ME.stack)
        disp(sprintf('Error in %s (line %d)', ME.stack(i,1).name, ME.stack(i,1).line));
        
        disp(sprintf('\n'));
    end
    
    %rethrow(ME);
end

end

