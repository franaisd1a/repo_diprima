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
    PixelIdxListPoints = CCpoints.PixelIdxList;
    statsP = regionprops(CCpoints,'Centroid','Area','Eccentricity', ...
        'MajorAxisLength','MinorAxisLength','Orientation','PixelList');
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
        %pixelListP   =  cell(length(statsP),1);
        
        for i=1:length(statsP)
            centroidP(i,:) = round(statsP(i).Centroid);
            if((centroidP(i,2)>borders(1) && centroidP(i,2)<borders(3)) ...
               && (centroidP(i,1)>borders(2) && centroidP(i,1)<borders(4)))
                areaP(i,:)         = statsP(i).Area;
                eccentricityP(i,:) = statsP(i).Eccentricity;
                majoraxisP(i,:)    = statsP(i).MajorAxisLength;
                minoraxisP(i,:)    = statsP(i).MinorAxisLength;
                orientationP(i,:)  = statsP(i).Orientation;
                %pixelListP{i,:}    = statsP(i).PixelList;
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
            % Per eliminare i punti piccoli
            threshValue=((max_points_diameter/4)+(mean(majoraxisP)/2))/2;
            noise=find(majoraxisP<threshValue);
            %noise=find(minoraxisP<ceil(max_streaks_minoraxis/4));%2
            
            % Remove noisy points
            points(noise,:)             = [];
            centroidP(noise,:)          = [];
            areaP(noise,:)              = [];
            eccentricityP(noise,:)      = [];
            majoraxisP(noise,:)         = [];
            minoraxisP(noise,:)         = [];
            orientationP(noise,:)       = [];
            PixelIdxListPoints(noise) = [];
%             for n=1:length(noise)
%                 pixelListP{noise(n)} = [];
%             end
%             pixelListP=pixelListP(~cellfun('isempty',pixelListP));
            
            % Remove not circular points
            circPoint=find(points==1);
            max_dim_array=length(circPoint);
            output.POINTS = zeros(max_dim_array,3);
            output.POINTS =[ centroidP(circPoint,1) , ...
                             centroidP(circPoint,2) , ...
                             sub2ind( imgSz, ...
                                      centroidP(circPoint,2) , ...
                                      centroidP(circPoint,1)) ];
            output.majoraxis=majoraxisP(circPoint);
            output.minoraxis=minoraxisP(circPoint);
            output.orientation=orientationP(circPoint);
            output.pixelIdxListPoints=PixelIdxListPoints(circPoint);
%             output.pixelList=cell(max_dim_array,1);
%             for c=1:max_dim_array
%                 output.pixelList{c}=pixelListP{circPoint(c)};
%             end
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

