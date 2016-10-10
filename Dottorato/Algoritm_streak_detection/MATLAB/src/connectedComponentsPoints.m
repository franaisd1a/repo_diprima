function [ output ] = connectedComponentsPoints( varargin )

% Found connected component on N1 X N2 images

% global FIGURE FIGURE_1
disp('Start connectedComponentsPoints function.')

%% Input

%  1) N1 X N2 image
%  2) Border limits

%% Output

% Output struct with different fields:
% 1) output.error: boolean value, 1 is error
% 2) output.POINTS: connected component centroid
% 3) output.max_points_diameter: connected component centroid
% 4) output.min_points_diameter: connected component centroid

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
    borders=varargin{2};
end

% *********************************************************************** %
%% Processing
% *********************************************************************** %
try
  
% ----------------------------------------------------------------------- %
%% Remove Salt and Pepper Noise from Image
% ----------------------------------------------------------------------- %

    tStart=tic;
    
    imgSz=size(img);
    
    CCpoints = bwconncomp(img);
    statsP = regionprops(CCpoints,'Centroid','Area','Eccentricity','MajorAxisLength','MinorAxisLength','Orientation');
    max_points_diameter=0;
    min_points_diameter=max(imgSz);
    
    if (length(statsP)~=0)%(~isempty(statsP))
        points       = zeros(length(statsP),1);
        centroidP    = zeros(length(statsP),2);
        areaP        = zeros(length(statsP),1);
        eccentricityP= zeros(length(statsP),1);
        majoraxisP   = zeros(length(statsP),1);
        minoraxisP   = zeros(length(statsP),1);
        orientationP = zeros(length(statsP),1);
        
        for i=1:length(statsP)
            centroidP(i,:) = round(statsP(i).Centroid);
            if((centroidP(i,2)>borders(1) && centroidP(i,2)<borders(3)) ...
               && (centroidP(i,1)>borders(2) && centroidP(i,1)<borders(4)))
                areaP(i,:)            = statsP(i).Area;
                eccentricityP(i,:)    = statsP(i).Eccentricity;
                majoraxisP(i,:)       = statsP(i).MajorAxisLength;
                minoraxisP(i,:)       = statsP(i).MinorAxisLength;
                orientationP(i,:)     = statsP(i).Orientation;
% Identify points
                if (majoraxisP(i)/minoraxisP(i)<1.6)%1.6 %mettere condizione di punto se circolare
                    points(i)=1;
                    if(majoraxisP(i,:)>max_points_diameter)
                        max_points_diameter=majoraxisP(i,:);
                    end
                    if(minoraxisP(i,:)<min_points_diameter)
                        min_points_diameter=minoraxisP(i,:);
                    end
                end
            else
            end
        end
        
        n_points  = sum(points);
        if(n_points)
            noise=find(majoraxisP<ceil(max_points_diameter/2));%Per eliminare i punti piccoli
            %noise=find(minoraxisP<ceil(max_streaks_minoraxis/2));
            points(noise,:)          = [];
            centroidP(noise,:)       = [];
            areaP(noise,:)           = [];
            eccentricityP(noise,:)   = [];
            majoraxisP(noise,:)      = [];
            minoraxisP(noise,:)      = [];
            orientationP(noise,:)    = [];
            
            max_dim_array=length(round(centroidP(find(points==1),1)));
            output.POINTS = zeros(max_dim_array,3);
            output.POINTS =[ centroidP(find(points==1),1) , ...
                             centroidP(find(points==1),2) , ...
                             sub2ind( imgSz, ...
                                      centroidP(find(points ==1),2) , ...
                                      centroidP(find(points ==1),1)) ];
            output.majoraxis=majoraxisP(find(points==1));
            output.minoraxis=minoraxisP(find(points==1));
            output.orientation=orientationP(find(points==1));
        end
    end
    
    output.max_points_diameter=max_points_diameter;
    output.min_points_diameter=min_points_diameter;
    
    
    tElapsed = toc(tStart);    
    disp(sprintf('End connectedComponentsPoints funtion %d sec.', tElapsed));
    disp(sprintf('\n'));
    output.error=0;
    
% ??????????????????????????????????????????????????????????????????????? %
%% Error handling
% ??????????????????????????????????????????????????????????????????????? %  

catch ME
    output.error=1;
    disp('Error using connectedComponentsPoints function.');
    disp(ME.message);
    disp(sprintf('\n'));
    
    for i=1:length(ME.stack)
        disp(sprintf('Error in %s (line %d)', ME.stack(i,1).name, ME.stack(i,1).line));
        
        disp(sprintf('\n'));
    end
    
    %rethrow(ME);
end

end

