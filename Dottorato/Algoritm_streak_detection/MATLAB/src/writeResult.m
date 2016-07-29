function [ output ] = writeResult( varargin )

% Write result on txt file

disp('Start writeResult function.')

%% Input

%  1) resultFileName
%  2) point
%  3) streaks

%% Output

% Output struct with different fields:
% 1) output.error: boolean value, 1 is error

output={};
output.error=1;

%% Input validation

if nargin~=3
    disp('Error! Wrong number of parameters.')
    disp(sprintf('\n'));
    return
else
%     disp('Correct number of parameters.')
    resultFileName=varargin{1};
    point=varargin{2};
    streaks=varargin{3};
end

% *********************************************************************** %
%% Processing
% *********************************************************************** %
try
  
% ----------------------------------------------------------------------- %
%% Write file
% ----------------------------------------------------------------------- %

    tStart=tic;
    
    fileID = fopen(resultFileName,'w');
    if isfield(streaks, 'STREAKS')
        fprintf(fileID,'Detected streaks\n');
        for i=1:length(streaks.STREAKS(:,1))
            fprintf(fileID, 'Pixel (%d,%d)\n', ...
                    streaks.STREAKS(i,1), streaks.STREAKS(i,2));
        end
    end
    if isfield(point, 'POINTS')
        fprintf(fileID,'\nDetected points\n');
        for i=1:length(point.POINTS(:,1))
            fprintf(fileID, 'Pixel (%d,%d)\n', ...
                    point.POINTS(i,1),point.POINTS(i,2));
        end
    end
    fprintf(fileID,'End\n');
    fclose(fileID);
    
    tElapsed = toc(tStart);    
    disp(sprintf('End writeResult funtion %d sec.', tElapsed));
    disp(sprintf('\n'));
    output.error=0;
    
% ??????????????????????????????????????????????????????????????????????? %
%% Error handling
% ??????????????????????????????????????????????????????????????????????? %  

catch ME
    output.error=1;
    disp('Error using writeResult function.');
    disp(ME.message);
    disp(sprintf('\n'));
    
    for i=1:length(ME.stack)
        disp(sprintf('Error in %s (line %d)', ME.stack(i,1).name, ME.stack(i,1).line));
        
        disp(sprintf('\n'));
    end
    
    %rethrow(ME);
end

end

