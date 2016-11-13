function [ output, streaks ] = falsePositive( varargin )

% Search for false positive detection

global FIGURE_1
debugFig = 1;
disp('Start falsePositive function.')

%% Input

%  1) N1 X N2 image

%% Output

% Output struct with different fields:
% 1) output.error: boolean value, 1 is error
% 2) output.: 

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
    index=varargin{2};
    streaks=varargin{3};
        
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
    b=1500;

% % Use the Gradient Magnitude as the Segmentation Function
%     hy = fspecial('sobel');
%     hx = hy';
%     Iy = imfilter(double(img), hy, 'replicate');
%     Ix = imfilter(double(img), hx, 'replicate');
%     gradmag = sqrt(Ix.^2 + Iy.^2);
%     if(debugFig)
%         figure(b);
%         imshow(gradmag,[]), title('Gradient magnitude (gradmag)')
%         b=b+1;
%     end
        
% Mark the Foreground Objects

    se = strel('disk', 3);

    % Opening-by-reconstruction
    Ie = imerode(img, se);
    Iobr = imreconstruct(Ie, img);
    if(debugFig)
        figure(b);
        imshow(Iobr), title('Opening-by-reconstruction (Iobr)')
        b=b+1;
    end
    
% Opening-Closing-by-reconstruction
    Iobrd = imdilate(Iobr, se);
    Iobrcbr = imreconstruct(imcomplement(Iobrd), imcomplement(Iobr));
    Iobrcbr = imcomplement(Iobrcbr);
    if(debugFig)
        figure(b);
        imshow(Iobrcbr); title('Opening-closing by reconstruction (Iobrcbr)')
        b=b+1;
    end

    threshold = 130/255;
    imgBW = im2bw(img, threshold);

% Noise removal
    seD = strel('disk',3);%6selezionare dimensione
    opening = imopen(imgBW,seD);
    closing = imclose(opening, seD);

% Sure background area
    bg = imdilate(closing,[seD seD seD]);

    fgm0 = imregionalmax(Iobrcbr);
    fgm = fgm0 & bg;
    if(debugFig)
        figure(b);
        imshow(fgm), title('Regional maxima of opening-closing by reconstruction (fgm)')
        b=b+1;
    end
    
    se2 = strel(ones(3));%5,5
    fgm2 = imclose(fgm, se2);
    fgm3 = imerode(fgm2, se2);

    %mettere minima area degli oggetti trovati al posto del 20
    fgm4 = bwareaopen(fgm3, 20);
    if(FIGURE_1)
        figure(b);
        imshow(fgm4), title('Foreground Objects')
        b=b+1;
    end
    
% Count objects

    CCstreaks = bwconncomp(fgm4);
    stats = regionprops(CCstreaks,'Centroid','Area','MajorAxisLength','MinorAxisLength','Orientation');
    
    if (length(stats)>1)
        streaks.STREAKS(index,:)     = [];
        streaks.majoraxis(index,:)   = [];
        streaks.minoraxis(index,:)   = [];
        streaks.orientation(index,:) = [];
    end
    
    
    tElapsed = toc(tStart);    
    disp(sprintf('End falsePositive funtion %d sec.', tElapsed));
    disp(sprintf('\n'));
    output.error=0;
    
% ??????????????????????????????????????????????????????????????????????? %
%% Error handling
% ??????????????????????????????????????????????????????????????????????? %  

catch ME
    output.error=1;
    disp('Error using falsePositive function.');
    disp(ME.message);
    disp(sprintf('\n'));
    
    for i=1:length(ME.stack)
        disp(sprintf('Error in %s (line %d)', ME.stack(i,1).name, ME.stack(i,1).line));
        
        disp(sprintf('\n'));
    end
    
    rethrow(ME);
end

end

