function [ output ] = falsePositiveDetection( varargin )

% Search for false positive detection

global FIGURE_1
disp('Start falsePositiveDetection function.')

%% Input

%  1) N1 X N2 image

%% Output

% Output struct with different fields:
% 1) output.error: boolean value, 1 is error
% 2) output.: 

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
%% ROI Extraction
% ----------------------------------------------------------------------- %

    tStart=tic;
    
    imgSz=size(img);
    b=500;
    if 0 %opencv method
        hy = fspecial('sobel');
        hx = hy';
        Iy = imfilter(double(img), hy, 'replicate');
        Ix = imfilter(double(img), hx, 'replicate');
        gradmag = sqrt(Ix.^2 + Iy.^2);
        figure(1000)
        imshow(gradmag,[]), title('Gradient magnitude (gradmag)')
        
        threshold = 130/255;
        imgBW = im2bw(img, threshold);
        
        % Noise removal
        seD = strel('disk',3);%6selezionare dimensione
        opening = imopen(imgBW,seD);
        
        closing = imclose(opening, seD);
                
        % Sure background area
        bg = imdilate(closing,[seD seD seD]);
        bgArea = ~bg;
        
        if(FIGURE_1)
            figure(b);imshow(bgArea);b=b+1;
        end
                
% Sure foreground area
        distBW = bwdist(closing,'euclidean');
        fgAreaN = im2bw(img, 0.7);
        se2 = strel(ones(3));%5,5
        fgAreaN2 = imclose(fgAreaN, se2);
        fgAreaN3 = imerode(fgAreaN2, se2);
        fgArea = bwareaopen(fgAreaN3, 20);
        if(FIGURE_1)
            figure(b);imshow(fgArea);b=b+1;
        end
        
        % Unknown area
        ukArea = bg - fgArea;
        if(FIGURE_1)
            figure(b);imshow(ukArea);b=b+1;
        end
        
        finalImg=zeros(imgSz);
        finalImg(bgArea)=1;
        finalImg(fgArea)=10;
        if(FIGURE_1)
            figure(b);imshow(finalImg);b=b+1;
        end
        D = bwdist(finalImg);
        DL = watershed(D);
        bgmF = DL == 0;
        figure
        imshow(bgmF), title('Watershed ridge lines (bgm)')
        
        
        
        zerooo=false(imgSz);
        gradmag2 = imimposemin(gradmag, zerooo | fgArea);
        L = watershed(gradmag2);
        
    else %matlab method
% Use the Gradient Magnitude as the Segmentation Function
        hy = fspecial('sobel');
        hx = hy';
        Iy = imfilter(double(img), hy, 'replicate');
        Ix = imfilter(double(img), hx, 'replicate');
        gradmag = sqrt(Ix.^2 + Iy.^2);
        figure(1000)
        imshow(gradmag,[]), title('Gradient magnitude (gradmag)')
        
% Mark the Foreground Objects
        if 1
            se = strel('disk', 3);

            % Opening-by-reconstruction
            Ie = imerode(img, se);
            Iobr = imreconstruct(Ie, img);
            %         figure
            %         imshow(Iobr), title('Opening-by-reconstruction (Iobr)')

            % Opening-Closing-by-reconstruction
            Iobrd = imdilate(Iobr, se);
            Iobrcbr = imreconstruct(imcomplement(Iobrd), imcomplement(Iobr));
            Iobrcbr = imcomplement(Iobrcbr);
            figure;imshow(Iobrcbr); title('Opening-closing by reconstruction (Iobrcbr)')

            
            threshold = 130/255;
            imgBW = im2bw(img, threshold);

            % Noise removal
            seD = strel('disk',3);%6selezionare dimensione
            opening = imopen(imgBW,seD);

            closing = imclose(opening, seD);

            % Sure background area
            bg = imdilate(closing,[seD seD seD]);
            
            
            fgm0 = imregionalmax(Iobrcbr);
            if 1
            fgm = fgm0 & bg;
            else 
                fgm = fgm0;
            end
            figure(b)
            imshow(fgm), title('Regional maxima of opening-closing by reconstruction (fgm)')
            b=b+1;
            
            se2 = strel(ones(3));%5,5
            fgm2 = imclose(fgm, se2);
            fgm3 = imerode(fgm2, se2);

            fgm4 = bwareaopen(fgm3, 20);
            figure
            imshow(fgm4), title('Foreground Objects')
        else
            % Sure foreground area
            distBW = bwdist(gradmag,'euclidean');
%             fgAreaN = im2bw(img, 0.7);
            se2 = strel(ones(3));%5,5
            fgAreaN2 = imclose(gradmag, se2);
            Iobrcbr = imerode(fgAreaN2, se2);
            if(FIGURE_1)
                figure(b);imshow(Iobrcbr);b=b+1;
            end
            fgm4 = bwareaopen(Iobrcbr, 20);
            if(FIGURE_1)
                figure(b);imshow(fgm4);b=b+1;
            end
        end
% Compute Background Markers
        level = graythresh( Iobrcbr );
        bw = im2bw(Iobrcbr, level);
        figure
        imshow(bw), title('Thresholded opening-closing by reconstruction (bw)')
        
        D = bwdist(bw);%~bw
        DL = watershed(D);
        bgm = DL == 0;
        figure
        imshow(bgm,[]), title('Watershed ridge lines (bgm)')
        
% Compute the Watershed Transform of the Segmentation Function
        gradmag2 = imimposemin(gradmag, ~(bgm | fgm4));
        if(FIGURE_1)
                figure(b);imshow(gradmag2,[]);b=b+1;
            end
        L = watershed(gradmag2);
        
% Visualize the Result
        
        Lrgb = label2rgb(L, 'jet', 'w', 'shuffle');
        figure
        imshow(Lrgb)
        title('Colored watershed label matrix (Lrgb)')
        
        
    end
    
    
    
    
    tElapsed = toc(tStart);    
    disp(sprintf('End falsePositiveDetection funtion %d sec.', tElapsed));
    disp(sprintf('\n'));
    output.error=0;
    
% ??????????????????????????????????????????????????????????????????????? %
%% Error handling
% ??????????????????????????????????????????????????????????????????????? %  

catch ME
    output.error=1;
    disp('Error using falsePositiveDetection function.');
    disp(ME.message);
    disp(sprintf('\n'));
    
    for i=1:length(ME.stack)
        disp(sprintf('Error in %s (line %d)', ME.stack(i,1).name, ME.stack(i,1).line));
        
        disp(sprintf('\n'));
    end
    
    rethrow(ME);
end

end

