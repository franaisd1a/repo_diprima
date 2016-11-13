function [ output ] = roiExtraction( varargin )

% Extract ROI from image

global FIGURE_1
disp('Start roiExtraction function.')

%% Input

%  1) N1 X N2 image

%% Output

% Output struct with different fields:
% 1) output.error: boolean value, 1 is error
% 2) output.: 

output={};
output.error=1;

%% Input validation

if nargin~=5
    disp('Error! Wrong number of parameters.')
    disp(sprintf('\n'));
    return
else
%     disp('Correct number of parameters.')
    img=varargin{1};
    center=varargin{2};
    majoraxis=varargin{3};
    minoraxis=varargin{4};
    teta=varargin{5};
end

% *********************************************************************** %
%% Processing
% *********************************************************************** %
try
  
% ----------------------------------------------------------------------- %
%% ROI Extraction
% ----------------------------------------------------------------------- %

    tStart=tic;
    
    imgSz=size(img);
    
    num = length(center(:,1));
    output.ROI = cell(num,2);
    
    for i=1:num
        majAxis = majoraxis(i);
        minAxis = minoraxis(i);
        t = teta(i);
        
        %ROI dimension
        width  = ceil((abs(majAxis * cosd(t)) + abs(2*minAxis * sind(t))));%/2
        height = ceil((abs(majAxis * sind(t)) + abs(2*minAxis * cosd(t))));%/2
    
        % Power of 2 dimension
        expSTREAKdimensionX=ceil(log2(round(width)));
        width=(2^expSTREAKdimensionX);
        expSTREAKdimensionY=ceil(log2(round(height)));
        height=(2^expSTREAKdimensionY);
    

    
        limInfXs=center(i,2)-height;
        limSupXs=center(i,2)+height-1;
        if(limInfXs<1)
            limInfXs=1;
            limSupXs=limInfXs+2*height-1;
        end
        if(limSupXs>imgSz(1))
            limInfXs=imgSz(1)-2*height+1;
            limSupXs=imgSz(1);
        end
        
        limInfYs=center(i,1)-width;
        limSupYs=center(i,1)+width-1;
        if(limInfYs<1)
            limInfYs=1;
            limSupYs=limInfYs+2*width-1;
        end
        if(limSupYs>imgSz(2))
            limInfYs=imgSz(2)-2*width+1;
            limSupYs=imgSz(2);
        end
        
        %prendere area pari a potenza di 2
        
        output.ROI{i,1} = img(limInfXs:limSupXs, limInfYs:limSupYs);
        output.ROI{i,2} = [limInfXs,limInfYs , limSupXs,limSupYs];
        
        if(FIGURE_1)
            b=500;
            figure(b);
            imshow(output.ROI{i,1});
            b=b+i;
        end
    end
    
    
    tElapsed = toc(tStart);    
    disp(sprintf('End roiExtraction funtion %d sec.', tElapsed));
    disp(sprintf('\n'));
    output.error=0;
    
% ??????????????????????????????????????????????????????????????????????? %
%% Error handling
% ??????????????????????????????????????????????????????????????????????? %  

catch ME
    output.error=1;
    disp('Error using roiExtraction function.');
    disp(ME.message);
    disp(sprintf('\n'));
    
    for i=1:length(ME.stack)
        disp(sprintf('Error in %s (line %d)', ME.stack(i,1).name, ME.stack(i,1).line));
        
        disp(sprintf('\n'));
    end
    
    %rethrow(ME);
end

end

