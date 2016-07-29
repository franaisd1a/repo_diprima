function [ output ] = houghTransform( varargin )

% Hough Transform on a N1 X N2 image to calculate the inclination angle of
% the streaks

% global FIGURE FIGURE_1
disp('Start houghTransform function.')

%% Input

%  1) N1 X N2 image

%% Output

% Output struct with different fields:
% 1) output.error: boolean value, 1 is error
% 2) output.tetaStreak: inclination angle of the streaks

output={};
output.error=1;

%% Input validation

if nargin~=1
    disp('Error! Wrong number of parameters.')
    disp(sprintf('\n'));
    return
else
%     disp('Correct number of parameters.')
    img=varargin{1};
end

% *********************************************************************** %
%% Processing
% *********************************************************************** %
try
  
% ----------------------------------------------------------------------- %
%% Hough transform
% ----------------------------------------------------------------------- %

    tStart=tic;
    
    [H1,T1,R1] = hough(img,'RhoResolution',0.5,'ThetaResolution',0.5);
    
    %P  = houghpeaks(H,5,'threshold',ceil(0.3*max(H(:))));
    P1  = houghpeaks(H1,1,'threshold',ceil(0.9*max(H1(:))),'NHoodSize',[31 31]);
    x1 = T1(P1(:,2));
    output.tetaStreak=x1+90;
    
    tElapsed = toc(tStart);    
    disp(sprintf('End houghTransform funtion %d sec.', tElapsed));
    disp(sprintf('\n'));
    output.error=0;
    
% ??????????????????????????????????????????????????????????????????????? %
%% Error handling
% ??????????????????????????????????????????????????????????????????????? %  

catch ME
    output.error=1;
    disp('Error using houghTransform function.');
    disp(ME.message);
    disp(sprintf('\n'));
    
    for i=1:length(ME.stack)
        disp(sprintf('Error in %s (line %d)', ME.stack(i,1).name, ME.stack(i,1).line));
        
        disp(sprintf('\n'));
    end
    
    %rethrow(ME);
end

end

