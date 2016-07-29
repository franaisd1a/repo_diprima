function [ output ] = medianFilters( varargin )

% Filter a N1 X N2 image using median filter creating an N1 X N2 image.
% If the function is call with three inputs return the result of the
% subtraction between the two median filtered images

global FIGURE_1
disp('Start medianFilters function.')

%% Input

%  1) N1 X N2 image
%  2) Little Kerlen
%  3) Big Kerlen

%% Output

% Output struct with different fields:
% 1) output.error: boolean value, 1 is error
% 2) output.medianImg: filtered image

output={};
output.error=1;

%% Input validation

if nargin<2 || nargin>3
    disp('Error! Wrong number of parameters.')
    disp(sprintf('\n'));
    return
else
%     disp('Correct number of parameters.')
    img=varargin{1};
    littleKerlen=varargin{2};
    if 3==nargin
        bigKerlen=varargin{3};
    end
end

% *********************************************************************** %
%% Processing
% *********************************************************************** %
try
  
% ----------------------------------------------------------------------- %
%% Median filters
% ----------------------------------------------------------------------- %

    tStart=tic;
    
    littleMedianImg = medfilt2(img, littleKerlen);
    if 3==nargin
        bigMedianImg = medfilt2(img, bigKerlen);
        output.medianImg = littleMedianImg - bigMedianImg;
    else
        output.medianImg = littleMedianImg;
    end
    
    if(FIGURE_1)
        figure('name','Median filter');
        imshow(output.medianImg);
        if 3==nargin
            figure('name','Little median filter');
            imshow(littleMedianImg);
            figure('name','Big median filter');
            imshow(bigMedianImg);
        end
    end
    
    tElapsed = toc(tStart);    
    disp(sprintf('End medianFilters funtion %d sec.', tElapsed));
    disp(sprintf('\n'));
    output.error=0;
    
% ??????????????????????????????????????????????????????????????????????? %
%% Error handling
% ??????????????????????????????????????????????????????????????????????? %  

catch ME
    output.error=1;
    disp('Error using medianFilters function.');
    disp(ME.message);
    disp(sprintf('\n'));
    
    for i=1:length(ME.stack)
        disp(sprintf('Error in %s (line %d)', ME.stack(i,1).name, ME.stack(i,1).line));
        
        disp(sprintf('\n'));
    end
    
    %rethrow(ME);
end

end

