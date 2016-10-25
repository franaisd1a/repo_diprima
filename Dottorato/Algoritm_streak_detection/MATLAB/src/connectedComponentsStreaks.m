function [ output ] = connectedComponentsStreaks( varargin )

% Found connected component on N1 X N2 images

disp('Start connectedComponentsStreaks function.')

%% Input

%  1) N1 X N2 image

%% Output

% Output struct with different fields:
% 1) output.error: boolean value, 1 is error
% 2) output.STREAKS: connected component centroid
% 3) output.min_streaks_minoraxis: min minor axis dimension
% 4) output.max_streaks_minoraxis: max minor axis dimension
% 5) output.max_streaks_majoraxis: min major axis dimension

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
    borders=varargin{2};
    points=varargin{3};
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
        
    %% Connected components: streaks
    
    CCstreaks = bwconncomp(img);
    PixelIdxListStreaks = CCstreaks.PixelIdxList;
    stats = regionprops(CCstreaks,'Centroid','Area','Eccentricity','MajorAxisLength','MinorAxisLength','Orientation');
    min_streaks_minoraxis=max(imgSz);
    max_streaks_minoraxis=0;
    max_streaks_majoraxis=0;
    
    if (length(stats)~=0)%(~isempty(stats))
        streaks     = zeros(length(stats),1);
        centroid    = zeros(length(stats),2);
        area        = zeros(length(stats),1);
        eccentricity= zeros(length(stats),1);
        majoraxis   = zeros(length(stats),1);
        minoraxis   = zeros(length(stats),1);
        axisratio   = zeros(length(stats),2);
        orientation = zeros(length(stats),1);
        
        for i=1:length(stats)
            centroid(i,:) = round(stats(i).Centroid);
            if((centroid(i,2)>borders(1) && centroid(i,2)<borders(3)) ...
                    && (centroid(i,1)>borders(2) && centroid(i,1)<borders(4)))
                area(i,:)            = stats(i).Area;
                eccentricity(i,:)    = stats(i).Eccentricity;
                majoraxis(i,:)       = stats(i).MajorAxisLength;
                minoraxis(i,:)       = stats(i).MinorAxisLength;
                axisratio(i,:)       = [majoraxis(i)/minoraxis(i),i];
                orientation(i,:)     = stats(i).Orientation;
                
% Identify streaks
                
                if(majoraxis(i)/minoraxis(i)>4)%6
                    streaks(i)=1;
                    if(min_streaks_minoraxis>minoraxis(i,:))
                        min_streaks_minoraxis=minoraxis(i,:);
                    end
                    if(max_streaks_minoraxis<minoraxis(i,:))
                        max_streaks_minoraxis=minoraxis(i,:);
                    end
                    if(max_streaks_majoraxis<majoraxis(i,:))
                        max_streaks_majoraxis=majoraxis(i,:);
                    end
                else
                end
            else
            end
        end
        n_streaks = sum(streaks);
        if(n_streaks)%==0)
            %             min_streaks_minoraxis=min_points_diameter/2;%max_points_diameter
            %         end
            if(n_streaks>1)
                noiseThin=find(minoraxis<ceil(min_streaks_minoraxis));%Per eliminare le strisciate sottili
                %noiseThin=find(minoraxis<ceil(max_streaks_minoraxis/2));
            else
                min_streaks_minoraxis=2;
                noiseThin=find(minoraxis<ceil(min_streaks_minoraxis));%min_streaks_minoraxis);
            end
            streaks(noiseThin,:)        = [];
            centroid(noiseThin,:)       = [];
            area(noiseThin,:)           = [];
            eccentricity(noiseThin,:)   = [];
            majoraxis(noiseThin,:)      = [];
            minoraxis(noiseThin,:)      = [];
            axisratio(noiseThin,:)      = [];
            orientation(noiseThin,:)    = [];
            PixelIdxListStreaks(noiseThin) = [];
            stats(noiseThin,:)          = [];
            
            noiseShort=find(majoraxis<ceil(max_streaks_majoraxis/2));%Per eliminare le strisciate corte
            streaks(noiseShort,:)        = [];
            centroid(noiseShort,:)       = [];
            area(noiseShort,:)           = [];
            eccentricity(noiseShort,:)   = [];
            majoraxis(noiseShort,:)      = [];
            minoraxis(noiseShort,:)      = [];
            axisratio(noiseShort,:)      = [];
            orientation(noiseShort,:)    = [];
            PixelIdxListStreaks(noiseShort) = [];
            stats(noiseShort,:)          = [];
            
            indexStreak = find(streaks==0);
            streaks(indexStreak,:)        = [];
            centroid(indexStreak,:)       = [];
            area(indexStreak,:)           = [];
            eccentricity(indexStreak,:)   = [];
            majoraxis(indexStreak,:)      = [];
            minoraxis(indexStreak,:)      = [];
            axisratio(indexStreak,:)      = [];
            orientation(indexStreak,:)    = [];
            PixelIdxListStreaks(indexStreak) = [];
            stats(indexStreak,:)          = [];
            
            indexStreakValid = find(streaks==1);
            max_dim_array=length(indexStreakValid);
            output.STREAKS=zeros(max_dim_array,3);
            
            output.STREAKS = [ centroid(indexStreakValid,1) ...
                             , centroid(indexStreakValid,2) ...
                             , sub2ind( imgSz ...
                                      , centroid(indexStreakValid,2) ...
                                      , centroid(indexStreakValid,1))];
            output.majoraxis = majoraxis(indexStreakValid);
            output.minoraxis = minoraxis(indexStreakValid);
            output.orientation = orientation(indexStreakValid);
        end
    end
    
%Delete points on streak
    if isfield(points,'POINTS')
        if isfield(output,'STREAKS')
            for j=1:length(output.STREAKS(:,1))
                for i=1:length(points.POINTS(:,1))
%                     if(find(PixelIdxListStreaks{1,j}==points.POINTS(i,3)))
%                         points.POINTS(i,3)=-1;
%                     end
                    if(find(output.STREAKS(j,3)==points.pixelIdxListPoints{1,i}))
                        output.STREAKS(j,3)=-1;
                    end
                end
            end
            
            noiseStreak=find(output.STREAKS(:,3)<0);
            output.STREAKS(noiseStreak,:)     = [];
            output.majoraxis(noiseStreak,:)   = [];
            output.minoraxis(noiseStreak,:)   = [];
            output.orientation(noiseStreak,:) = [];
%             noisePoint=find(points.POINTS(:,3)<0);
%             points.POINTS(noisePoint,:)      = [];
%             points.majoraxis(noisePoint,:)   = [];
%             points.minoraxis(noisePoint,:)   = [];
%             points.orientation(noisePoint,:) = [];
                        

        end
    end
    
    output.min_streaks_minoraxis=min_streaks_minoraxis;
    output.max_streaks_minoraxis=max_streaks_minoraxis;
    output.max_streaks_majoraxis=max_streaks_majoraxis;
    
    tElapsed = toc(tStart);    
    disp(sprintf('End connectedComponentsStreaks funtion %d sec.', tElapsed));
    disp(sprintf('\n'));
    output.error=0;
    
% ??????????????????????????????????????????????????????????????????????? %
%% Error handling
% ??????????????????????????????????????????????????????????????????????? %  

catch ME
    output.error=1;
    disp('Error using connectedComponentsStreaks function.');
    disp(ME.message);
    disp(sprintf('\n'));
    
    for i=1:length(ME.stack)
        disp(sprintf('Error in %s (line %d)', ME.stack(i,1).name, ME.stack(i,1).line));
        
        disp(sprintf('\n'));
    end
    
    %rethrow(ME);
end

end

