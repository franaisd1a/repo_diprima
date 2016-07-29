clear all;  clc;
close all;

FIGURE=0;
FILE=0;

%% Input Folder

if FILE
    files=dir('*.jpg');
else
    files=1;
end
image='hamr_118';%hamr_ %picture
for file_number=1:length(files)
    t_start=tic;
    
    %% Strart processing
    if FILE
        name_file=files(file_number,1).name;
        [pathstr,name,ext] = fileparts(name_file);
        image=name;%hamr_ %picture
    end
    jpg_format='.jpg';
    fit_format='.fit';
    image_jpg=strcat(image,jpg_format);
    image_fit=strcat(image,fit_format);
    
    if(exist(image_fit))
        data = fitsread(image_fit);
        % [data_img,map] = imread(image_fit);
        info = fitsinfo(image_fit);
        fits_info = imfinfo(image_fit);
    end
    Img_input=imread(image_jpg);%picture
    %     I_input(:,1:2)=0;
    I_input_size=size(Img_input);
    
    figure(1);
    imshow(Img_input);
    
    %% SVD
    
    % Convert to double precision
    I = im2double(Img_input);
    norm_I=norm(I,'fro');
    [U,S,V] = svd(I);
    sigma=diag(S);
    I_svd = sigma(1)*U(:,1)*V(:,1)';
    norm_I_svd=norm(I_svd,'fro');
    %     figure(100)
    %     imshow(I_svd)
    % Reduce the singular values by a chosen rank k of the matrix
    rank=1;
    E=norm_I_svd/norm_I    %Percentuale di energeia
    while E<0.75
        rank=rank+1;
        if rank<min(size(Img_input))
            I_svd = I_svd + sigma(rank)*U(:,rank)*V(:,rank)';
            norm_I_svd=norm(I_svd,'fro');
            E=norm_I_svd/norm_I;
            % imshow(I_svd)
        else
            break
        end
    end
    figure(100)
    imshow(I_svd)
    
    % To plot singular values use the logarithmic based scale plot
    if(FIGURE)
        figure(110)
        semilogy(S/S(1),'.-');
        ylabel('singular values');
        grid;
    end
    % Convert image from double to uint8
    I_input=im2uint8(I_svd);
    
    %% Streak inclination
    
    [H1,T1,R1] = hough(I_input,'RhoResolution',0.5,'ThetaResolution',0.5);
    
    %P  = houghpeaks(H,5,'threshold',ceil(0.3*max(H(:))));
    P1  = houghpeaks(H1,1,'threshold',ceil(0.9*max(H1(:))),'NHoodSize',[31 31]);
    x1 = T1(P1(:,2));
    teta_streak=x1+90;
    
    %% Morphology opening
    
    %     se = strel('disk',20);%4 20
    
    se = strel('line', 20, -teta_streak);
    
    iopen=imopen(I_input,se);
    % iclose=imclose(I_input,se);
    if(FIGURE)
        figure(100);
        imshow(iopen);
    end
    % imshow(iclose);
    %% Subtraction 1
    
    isottr=I_input-iopen;
    if(FIGURE)
        figure(102);
        imshow(isottr);
    end
    %% Gaussian filter
    
    hsize=[100 100];%[100 100];
    sigma=30;%10 25
    h = fspecial('gaussian', hsize, sigma);
    Iblur1 = imfilter(I_input,h);
    if(FIGURE)
        figure(103);
        imshow(Iblur1);
    end
    %% Subtraction 2
    
    isottr2=isottr-Iblur1;
    if(FIGURE)
        figure(104);
        imshow(isottr2);
    end
    %% Binarization
    
    [level EM] = graythresh(isottr2);
    BW_point = im2bw(isottr2, level);
    
        figure(3);
        imshow(BW_point);
    
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Ricerca strisciate %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %% Morphology opening
    
        se = strel('disk',20);%4 20
    
%     se = strel('line', 20, -teta_streak);
    
    iopen=imopen(I_input,se);
    % iclose=imclose(I_input,se);
    if(FIGURE)
        figure(100);
        imshow(iopen);
    end
    % imshow(iclose);
    %% Subtraction 1
    
    isottr=I_input-iopen;
    if(FIGURE)
        figure(102);
        imshow(isottr);
    end
    %% Gaussian filter
    
    hsize=[100 100];%[100 100];
    sigma=30;%10 25
    h = fspecial('gaussian', hsize, sigma);
    Iblur1 = imfilter(I_input,h);
    if(FIGURE)
        figure(103);
        imshow(Iblur1);
    end
    %% Subtraction 2
    
    isottr2=isottr-Iblur1;
    if(FIGURE)
        figure(104);
        imshow(isottr2);
    end
    %% Binarization
    
    [level EM] = graythresh(isottr2);
    BW_streak = im2bw(isottr2, level)-BW_point;
    
        figure(30);
        imshow(BW_streak);
    
end

    
    
% % % % % % % % %     %% Remove Salt and Pepper Noise from Image
% % % % % % % % %     
% % % % % % % % %     % B = medfilt2(BW);
% % % % % % % % %     P=3;
% % % % % % % % %     conn=8;
% % % % % % % % %     B = bwareaopen(BW,P,conn);
% % % % % % % % %     
% % % % % % % % %     %% Connected components
% % % % % % % % %     
% % % % % % % % %     CC = bwconncomp(B);
% % % % % % % % %     stats = regionprops(CC,'Centroid','Area','Eccentricity','MajorAxisLength','MinorAxisLength','Orientation');
% % % % % % % % %     max_points_diameter=0;
% % % % % % % % %     min_streaks_minoraxis=max(I_input_size);
% % % % % % % % %     max_streaks_minoraxis=1;
% % % % % % % % %     
% % % % % % % % %     if (length(stats)~=0)
% % % % % % % % %         streaks     = zeros(length(stats),1);
% % % % % % % % %         points      = zeros(length(stats),1);
% % % % % % % % %         centroid    = zeros(length(stats),2);
% % % % % % % % %         area        = zeros(length(stats),1);
% % % % % % % % %         eccentricity= zeros(length(stats),1);
% % % % % % % % %         majoraxis   = zeros(length(stats),1);
% % % % % % % % %         minoraxis   = zeros(length(stats),1);
% % % % % % % % %         orientation = zeros(length(stats),1);
% % % % % % % % %         
% % % % % % % % %         for i=1:length(stats)
% % % % % % % % %             centroid(i,:)        = stats(i).Centroid;
% % % % % % % % %             area(i,:)            = stats(i).Area;
% % % % % % % % %             eccentricity(i,:)    = stats(i).Eccentricity;
% % % % % % % % %             majoraxis(i,:)       = stats(i).MajorAxisLength;
% % % % % % % % %             minoraxis(i,:)       = stats(i).MinorAxisLength;
% % % % % % % % %             orientation(i,:)     = stats(i).Orientation;
% % % % % % % % %             
% % % % % % % % %             %% Identify streaks and points
% % % % % % % % %             
% % % % % % % % %             if(majoraxis(i)/minoraxis(i)>6)%eccentricity(i,:)>0.9)
% % % % % % % % %                 streaks(i)=1;
% % % % % % % % %                 if(min_streaks_minoraxis>minoraxis(i,:))
% % % % % % % % %                     min_streaks_minoraxis=minoraxis(i,:);
% % % % % % % % %                 end
% % % % % % % % %                 if(max_streaks_minoraxis<minoraxis(i,:))
% % % % % % % % %                     max_streaks_minoraxis=minoraxis(i,:);
% % % % % % % % %                 end
% % % % % % % % %             elseif(majoraxis(i)/minoraxis(i)<1.5) %mettere condizione di punto se circolare
% % % % % % % % %                 points(i)=1;
% % % % % % % % %                 if(majoraxis(i,:)>max_points_diameter)
% % % % % % % % %                     max_points_diameter=majoraxis(i,:);
% % % % % % % % %                 end
% % % % % % % % %             else
% % % % % % % % %             end
% % % % % % % % %             
% % % % % % % % %         end
% % % % % % % % %         n_streaks = sum(streaks);
% % % % % % % % %         n_points  = sum(points);
% % % % % % % % %         if(n_streaks==0)
% % % % % % % % %             min_streaks_minoraxis=max_points_diameter/2;
% % % % % % % % %         end
% % % % % % % % %         noise=find(area<ceil(min_streaks_minoraxis));
% % % % % % % % %         stats(noise) = [];
% % % % % % % % %         
% % % % % % % % %         %% Remove noise
% % % % % % % % %         
% % % % % % % % %         B2 = bwareaopen(B,ceil(min_streaks_minoraxis)^2,conn);%bwareaopen(B,ceil(min_streaks_minoraxis^2),conn);
% % % % % % % % %         CC2 = bwconncomp(B2);
% % % % % % % % %         stats2 = regionprops(CC2,'Centroid','Area','Eccentricity','MajorAxisLength','MinorAxisLength','Orientation');
% % % % % % % % %         
% % % % % % % % %         streaks2     = zeros(length(stats2),1);
% % % % % % % % %         points2      = zeros(length(stats2),1);
% % % % % % % % %         centroid2    = zeros(length(stats2),2);
% % % % % % % % %         area2        = zeros(length(stats2),1);
% % % % % % % % %         eccentricity2= zeros(length(stats2),1);
% % % % % % % % %         majoraxis2   = zeros(length(stats2),1);
% % % % % % % % %         minoraxis2   = zeros(length(stats2),1);
% % % % % % % % %         orientation2 = zeros(length(stats2),1);
% % % % % % % % %         for i=1:length(stats2)
% % % % % % % % %             centroid2(i,:)        = stats2(i).Centroid;
% % % % % % % % %             area2(i,:)            = stats2(i).Area;
% % % % % % % % %             eccentricity2(i,:)    = stats2(i).Eccentricity;
% % % % % % % % %             majoraxis2(i,:)       = stats2(i).MajorAxisLength;
% % % % % % % % %             minoraxis2(i,:)       = stats2(i).MinorAxisLength;
% % % % % % % % %             orientation2(i,:)     = stats2(i).Orientation;
% % % % % % % % %             
% % % % % % % % %             %% Identify streaks and points
% % % % % % % % %             
% % % % % % % % %             if(majoraxis2(i)/minoraxis2(i)>6)%eccentricity(i,:)>0.9)
% % % % % % % % %                 streaks2(i)=1;
% % % % % % % % %             elseif(majoraxis(i)/minoraxis(i)<1.5)
% % % % % % % % %                 points2(i)=1;
% % % % % % % % %             else
% % % % % % % % %             end
% % % % % % % % %         end
% % % % % % % % %         n_streaks2 = sum(streaks2);
% % % % % % % % %         n_points2  = sum(points2);
% % % % % % % % %     end
% % % % % % % % %     
% % % % % % % % %     % figure(23);
% % % % % % % % %     % imshow(B);
% % % % % % % % %     % hold on;
% % % % % % % % %     % % Plot streaks' centroids
% % % % % % % % %     % plot(centroid(find(streaks==1),1),centroid(find(streaks==1),2),'*r')
% % % % % % % % %     % % Plot points' centroids
% % % % % % % % %     % plot(centroid(find(points==1),1),centroid(find(points==1),2),'+g')
% % % % % % % % %     
% % % % % % % % %     if(FIGURE)
% % % % % % % % %         figure(24);
% % % % % % % % %         imshow(B2);
% % % % % % % % %         hold on;
% % % % % % % % %     end
% % % % % % % % % %     max_dim_array=length(max(length(round(centroid2(find(streaks2==1),1))),round(centroid2(find(points2==1),1))));
% % % % % % % % %     max_dim_array=max(length(round(centroid2(find(streaks2==1),1))),length(round(centroid2(find(points2==1),1))));
% % % % % % % % %     STREAKS=zeros(max_dim_array,3);
% % % % % % % % %     POINTS=zeros(max_dim_array,3);
% % % % % % % % %     STREAKS=[round(centroid2(find(streaks2==1),1)) , round(centroid2(find(streaks2==1),2)) , sub2ind(I_input_size, round(centroid2(find(streaks2==1),2)), round(centroid2(find(streaks2==1),1)))];
% % % % % % % % %     POINTS =[round(centroid2(find(points2==1),1))  , round(centroid2(find(points2==1),2))  , sub2ind(I_input_size, round(centroid2(find(points2 ==1),2)), round(centroid2(find(points2 ==1),1)))];
% % % % % % % % %     if(FIGURE)
% % % % % % % % %         % Plot streaks' centroids
% % % % % % % % %         plot(STREAKS(:,1),STREAKS(:,2),'*r')
% % % % % % % % %         % Plot points' centroids
% % % % % % % % %         plot(POINTS(:,1),POINTS(:,2),'+g')
% % % % % % % % %     end
% % % % % % % % % %     %% Radon transform
% % % % % % % % % %     
% % % % % % % % % %     theta = 0:179;
% % % % % % % % % %     [R,xp] = radon(B,theta );
% % % % % % % % % %     
% % % % % % % % % %     % figure(4)
% % % % % % % % % %     % iptsetpref('ImshowAxesVisible','on')
% % % % % % % % % %     % imshow(R,[],'Xdata',theta,'Ydata',xp,'InitialMagnification','fit');
% % % % % % % % % %     % xlabel('\theta (degrees)')
% % % % % % % % % %     % ylabel('x''')
% % % % % % % % % %     % colormap(hot), colorbar
% % % % % % % % % %     % iptsetpref('ImshowAxesVisible','off')
% % % % % % % % % %     
% % % % % % % % % %     
% % % % % % % % % %     I_ra = iradon(R, theta);
% % % % % % % % % %     % figure(5);
% % % % % % % % % %     % imshow(I_ra);
% % % % % % % % % %     
% % % % % % % % % %     
% % % % % % % % % %     [p_max,theta_max]=find(R==max(max(R)));
% % % % % % % % % %     xp_max=xp(p_max);
% % % % % % % % % %     if(xp_max>=0)
% % % % % % % % % %         xp_max=xp_max-1;
% % % % % % % % % %     else
% % % % % % % % % %         xp_max=xp_max+1;
% % % % % % % % % %     end
% % % % % % % % % %     xp_X=(xp_max*cosd(theta_max))+I_input_size(2)/2;
% % % % % % % % % %     xp_Y=-(xp_max*sind(theta_max))+I_input_size(1)/2;
% % % % % % % % % %     
% % % % % % % % % %     
% % % % % % % % % %     x=1:I_input_size(2);
% % % % % % % % % %     retta=zeros(I_input_size(2),1);
% % % % % % % % % %     retta1=retta;
% % % % % % % % % %     retta2=retta;
% % % % % % % % % %     retta3=retta;
% % % % % % % % % %     dist=3;
% % % % % % % % % %     for ind=1:I_input_size(2);
% % % % % % % % % %         retta(ind)=-(tand(theta_max)*(x(ind)-I_input_size(2)/2))+(I_input_size(1)/2);
% % % % % % % % % %         %     retta1(ind)=(1/tand(theta_max))*(x(ind)-(I_input_size(2)/2)-xp_X)+(I_input_size(1)/2)+xp_Y;
% % % % % % % % % %         retta1(ind)=(1/tand(theta_max))*(x(ind)-xp_X)+xp_Y;
% % % % % % % % % %         retta2(ind)=(1/tand(theta_max))*(x(ind)-xp_X-dist)+xp_Y-dist;
% % % % % % % % % %         retta3(ind)=(1/tand(theta_max))*(x(ind)-xp_X+dist)+xp_Y+dist;
% % % % % % % % % %     end
% % % % % % % % % %     % figure(23)
% % % % % % % % % %     % plot(x,retta,'r',x,retta1,'m',x,retta2,'m',x,retta3,'m',xp_X,xp_Y,'*g');%,I_input_size(2)/2,I_input_size(1)/2,'*g'
% % % % % % % % % %     
% % % % % % % % %     
% % % % % % % % %     %% Hough transform
% % % % % % % % %     
% % % % % % % % %     % H --> Trasformata di Hough
% % % % % % % % %     % Per ogni valore della distanza rho viene variato l'angolo theta
% % % % % % % % %     %
% % % % % % % % %     % T --> Variazione dell'angolo theta=[-90:0.5:89.5]
% % % % % % % % %     % R --> Variazione della distanza rho=[-922:0.5:922]
% % % % % % % % %     % sqrt((size(image(1))^2)+(size(image(2))^2))=923.0211
% % % % % % % % %     
% % % % % % % % %     % RhoResolution=0.5;
% % % % % % % % %     % D = sqrt((size(I_input,1) - 1)^2 + (size(I_input,2) - 1)^2);
% % % % % % % % %     % nrho = 2*(ceil(D/RhoResolution)) + 1;
% % % % % % % % %     % diagonal = RhoResolution*ceil(D/RhoResolution);
% % % % % % % % %     
% % % % % % % % %     [H,T,R] = hough(B2,'RhoResolution',0.5,'ThetaResolution',0.5);
% % % % % % % % %     
% % % % % % % % %     %P  = houghpeaks(H,5,'threshold',ceil(0.3*max(H(:))));
% % % % % % % % %     P  = houghpeaks(H,10,'threshold',ceil(0.5*max(H(:))),'NHoodSize',[31 31]);
% % % % % % % % %     x = T(P(:,2)); y = R(P(:,1));
% % % % % % % % %     
% % % % % % % % %     % figure(10)
% % % % % % % % %     % imshow(H,[],'XData',T,'YData',R,'InitialMagnification','fit');
% % % % % % % % %     % xlabel('\theta');
% % % % % % % % %     % ylabel('\rho');
% % % % % % % % %     % colormap(hot), colorbar;
% % % % % % % % %     % axis on, axis normal, hold on;
% % % % % % % % %     % plot(x,y,'s','color','green');
% % % % % % % % %     
% % % % % % % % %     fillgap=5;
% % % % % % % % %     minlength=7;
% % % % % % % % %     if(max_points_diameter>minlength)
% % % % % % % % %         minlength=max_points_diameter;
% % % % % % % % %         fillgap=max_points_diameter;%-1;
% % % % % % % % %     end
% % % % % % % % %     % lines = houghlines(B2,T,R,P,'FillGap',fillgap,'MinLength',minlength);
% % % % % % % % %     lines = houghlines(B2,T,R,P,'MinLength',minlength);
% % % % % % % % %     if(FIGURE)
% % % % % % % % %         figure(11)
% % % % % % % % %         imshow(I_input);
% % % % % % % % %         hold on;
% % % % % % % % %     end
% % % % % % % % %     max_len = 0;
% % % % % % % % %     if(isfield(lines,'point1'))
% % % % % % % % %         for k = 1:length(lines)
% % % % % % % % %             
% % % % % % % % %             xy(k,:) = [lines(k).point1(1) lines(k).point2(1) lines(k).point1(2) lines(k).point2(2)];
% % % % % % % % %             if(FIGURE)
% % % % % % % % %                 plot(xy(k,1:2),xy(k,3:4),'LineWidth',2,'Color','green');
% % % % % % % % %                 
% % % % % % % % %                 % Plot beginnings and ends of lines
% % % % % % % % %                 plot(xy(k,1),xy(k,3),'x','LineWidth',2,'Color','yellow');
% % % % % % % % %                 plot(xy(k,2),xy(k,4),'x','LineWidth',2,'Color','red');
% % % % % % % % %             end
% % % % % % % % %             % Determine the endpoints of the longest line segment
% % % % % % % % %             len = norm(lines(k).point1 - lines(k).point2);
% % % % % % % % %             if ( len > max_len)
% % % % % % % % %                 max_len = len;
% % % % % % % % %                 xy_long = xy;
% % % % % % % % %             end
% % % % % % % % %         end
% % % % % % % % %         
% % % % % % % % %         %     plot(xy_long(1,1:2),xy_long(1,3:4),'LineWidth',2,'Color','cyan');
% % % % % % % % %         
% % % % % % % % %         %% Mask from Hough Trasform
% % % % % % % % %         
% % % % % % % % %         MASK=zeros(I_input_size);
% % % % % % % % %         MASK = im2bw(MASK, 0.5);
% % % % % % % % %         
% % % % % % % % %         for k = 1:length(lines)
% % % % % % % % %             %         len = norm(lines(k).point1 - lines(k).point2)+1;
% % % % % % % % %             len = lines(k).point2(1) - lines(k).point1(1)+1;
% % % % % % % % %             point_mask=zeros(len,2);
% % % % % % % % %             %         y=zeros(1,len);
% % % % % % % % %             angle=90+lines(k).theta;
% % % % % % % % %             for i=1:len
% % % % % % % % %                 if(abs(angle)~=90)
% % % % % % % % %                     point_mask(i,1) = i+lines(k).point1(1)-1;
% % % % % % % % %                     point_mask(i,2) = tand(angle)*(point_mask(i,1)-lines(k).point1(1))+lines(k).point1(2);
% % % % % % % % %                 else
% % % % % % % % %                     point_mask(i,1) = i+lines(k).point1(1)-1;
% % % % % % % % %                     point_mask(i,2) = lines(k).point1(2);
% % % % % % % % %                 end
% % % % % % % % %             end
% % % % % % % % %             MASK(round(point_mask(:,2)),point_mask(:,1))=1;
% % % % % % % % %         end
% % % % % % % % %         
% % % % % % % % %         se_mask = strel('disk',ceil(max_streaks_minoraxis));
% % % % % % % % %         MASK=imdilate(MASK,se_mask);
% % % % % % % % %         if(FIGURE)
% % % % % % % % %             figure(33)
% % % % % % % % %             imshow(MASK)
% % % % % % % % %         end
% % % % % % % % %         %     hold on
% % % % % % % % %         %     plot(xy_long(1,1:2),xy_long(1,3:4),'LineWidth',2,'Color','cyan');
% % % % % % % % %         CC_mask = bwconncomp(MASK);
% % % % % % % % %         stats_mask = regionprops(CC_mask,'Centroid','Area','Eccentricity','MajorAxisLength','MinorAxisLength','Orientation');
% % % % % % % % %         
% % % % % % % % %         [r_mask c_mask]=find(MASK==1);
% % % % % % % % %         index=1;
% % % % % % % % %         for c=1:I_input_size(2)
% % % % % % % % %             for r=1:I_input_size(1)
% % % % % % % % %                 if (index<=length(r_mask))
% % % % % % % % %                     if(c==c_mask(index))
% % % % % % % % %                         if(r==r_mask(index))
% % % % % % % % %                             IMAGE_F(r,c)=B2(r,c);
% % % % % % % % %                             index=index+1;
% % % % % % % % %                         else
% % % % % % % % %                             IMAGE_F(r,c)=0;
% % % % % % % % %                         end
% % % % % % % % %                     else
% % % % % % % % %                         IMAGE_F(r,c)=0;
% % % % % % % % %                     end
% % % % % % % % %                 else
% % % % % % % % %                     IMAGE_F(r,c)=0;
% % % % % % % % %                 end
% % % % % % % % %             end
% % % % % % % % %         end
% % % % % % % % %         
% % % % % % % % %         if (length(stats_mask)~=0)
% % % % % % % % %             found=zeros(CC_mask.NumObjects,2);
% % % % % % % % %             found_index=zeros(max_dim_array,2);
% % % % % % % % %             for i=1:CC_mask.NumObjects
% % % % % % % % %                 for n=1:max_dim_array
% % % % % % % % %                     %for n=1:length(max(length(STREAKS),length(POINTS)))
% % % % % % % % %                     if(n<=length(STREAKS(:,1)))
% % % % % % % % %                         find_streaks=find(CC_mask.PixelIdxList{1,i}==STREAKS(n,3));
% % % % % % % % %                         if (find_streaks~=0)
% % % % % % % % %                             found(i,1)=found(i,1)+1;
% % % % % % % % %                             found_index(n,1)=find_streaks;
% % % % % % % % %                         end
% % % % % % % % %                     end
% % % % % % % % %                     if(n<=length(POINTS(:,1)))
% % % % % % % % %                         find_points=find(CC_mask.PixelIdxList{1,i}==POINTS(n,3));
% % % % % % % % %                         if (find_points~=0)
% % % % % % % % %                             found(i,2)=found(i,2)+1;
% % % % % % % % %                             found_index(n,2)=find_points;
% % % % % % % % %                         end
% % % % % % % % %                     end
% % % % % % % % %                 end
% % % % % % % % %                 
% % % % % % % % %                 %Delete repeat points
% % % % % % % % %                 if(found(i,1)>1)
% % % % % % % % %                     for q=1:max_dim_array
% % % % % % % % %                         if(found_index(q,1)~=0)
% % % % % % % % %                             STREAKS(q,1)=round(stats_mask(i).Centroid(1,1));
% % % % % % % % %                             STREAKS(q,2)=round(stats_mask(i).Centroid(1,2));
% % % % % % % % % % % % % % % % %Viene assegnato alla centroide della
% % % % % % % % % % % % % % % % %strisciata il centroide della maschera
% % % % % % % % % % % % % % % % %CORREGGERE
% % % % % % % % %                             STREAKS(q,3)=-1;
% % % % % % % % %                         end
% % % % % % % % %                         if(found_index(q,2)~=0)
% % % % % % % % %                             POINTS(q,1)=-1;
% % % % % % % % %                             POINTS(q,2)=-1;
% % % % % % % % %                         end
% % % % % % % % %                     end
% % % % % % % % %                 end
% % % % % % % % %                 found_index=zeros(max_dim_array,2);
% % % % % % % % %             end
% % % % % % % % %             index_false_points=find(POINTS(:,1)==-1);
% % % % % % % % %             POINTS(index_false_points,:)=[];
% % % % % % % % %             index_false_streaks=find(STREAKS(:,3)==-1);
% % % % % % % % %             STREAKS(index_false_streaks(2:end),:)=[];
% % % % % % % % %         end
% % % % % % % % %         
% % % % % % % % %         
% % % % % % % % %         if(FIGURE)
% % % % % % % % %             figure(44)
% % % % % % % % %             imshow(B2);
% % % % % % % % %         end
% % % % % % % % %         
% % % % % % % % %         %% Identify type of observation
% % % % % % % % %         
% % % % % % % % %         sideral_tracking='sidereal_tracking';
% % % % % % % % %         NO_sideral_tracking='NO_sidereal_tracking';
% % % % % % % % %         
% % % % % % % % %         if length(STREAKS) > length(POINTS)
% % % % % % % % %             observation_type=NO_sideral_tracking
% % % % % % % % %         else
% % % % % % % % %             observation_type=sideral_tracking
% % % % % % % % %         end
% % % % % % % % %         
% % % % % % % % %         
% % % % % % % % %         %     imshow(IMAGE_F)
% % % % % % % % %         %     % IMAGE_F=B2(r_mask,c_mask);
% % % % % % % % %         %     se2 = strel('disk',3);
% % % % % % % % %         %     IMAGE_F=imdilate(IMAGE_F,se2);
% % % % % % % % %         %     figure(44)
% % % % % % % % %         %     imshow(IMAGE_F)
% % % % % % % % %         %     axis on
% % % % % % % % %         %
% % % % % % % % %         
% % % % % % % % %     end
% % % % % % % % %     
% % % % % % % % %     hold on
% % % % % % % % %     % Plot streaks' centroids
% % % % % % % % %     plot(STREAKS(:,1),STREAKS(:,2),'*r')
% % % % % % % % %     % Plot points' centroids
% % % % % % % % %     plot(POINTS(:,1),POINTS(:,2),'+g')
% % % % % % % % %     
% % % % % % % % %     %% Morphology operations
% % % % % % % % %     
% % % % % % % % %     % idil=imdilate(isottr2,se);
% % % % % % % % %     % figure(4);
% % % % % % % % %     % imshow(idil);
% % % % % % % % %     %
% % % % % % % % %     %
% % % % % % % % %     % Iedge=edge(I_input);
% % % % % % % % %     % figure(3);
% % % % % % % % %     % imshow(Iedge);
% % % % % % % % %     t_tot=toc(t_start);
% % % % % % % % %     fprintf('Total time: %d\n', t_tot);
% % % % % % % % % end
% % % % % % % % % 
% % % % % % % % % 
