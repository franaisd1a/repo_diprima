function [ output ] = removeSaltPepper( varargin )

% Remove salt and pepper noise on N1 X N2 images

global FIGURE_1
disp('Start removeSaltPepper function.')

%% Input

%  1) N1 X N2 image
%  2) Max noise dimension 

%% Output

% Output struct with different fields:
% 1) output.error: boolean value, 1 is error
% 2) output.remSaltPepperImg: image less salt and pepper noise

output={};
output.error=1;

%% Input validation

if nargin~=2
    disp('Error! Wrong number of parameters.')
    disp(sprintf('\n'));
    return
else
%     disp('Correct number of parameters.')
    img=varargin{1};
    P=varargin{2};
end

% *********************************************************************** %
%% Processing
% *********************************************************************** %
try
  
% ----------------------------------------------------------------------- %
%% Remove Salt and Pepper Noise from Image
% ----------------------------------------------------------------------- %

    tStart=tic;
    
    % B = medfilt2(BW);
    conn=8;
    output.remSaltPepperImg = bwareaopen(img,P,conn);
    
    if(FIGURE_1)
        figure('name','Binary image less salt and pepper noise for points detection');
        imshow(output.remSaltPepperImg);
    end
    
    tElapsed = toc(tStart);    
    disp(sprintf('End removeSaltPepper funtion %d sec.', tElapsed));
    disp(sprintf('\n'));
    output.error=0;
    
% ??????????????????????????????????????????????????????????????????????? %
%% Error handling
% ??????????????????????????????????????????????????????????????????????? %  

catch ME
    output.error=1;
    disp('Error using removeSaltPepper function.');
    disp(ME.message);
    disp(sprintf('\n'));
    
    for i=1:length(ME.stack)
        disp(sprintf('Error in %s (line %d)', ME.stack(i,1).name, ME.stack(i,1).line));
        
        disp(sprintf('\n'));
    end
    
    %rethrow(ME);
end

end

