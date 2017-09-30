format long;
clear all;
% close all;
clc;

PLOT = 0;
PLOT_TOT = 1;
PLOT_SFIG = 1;

%% Iput files 

root_path ='C:\Users\Francesco Diprima\Desktop\SPD_result\fileRDLS\'; 

nm_file={  '41384.00007958.TRK.rdls';
           '41384.00007959.TRK.rdls';
           '41384.00007960.TRK.rdls';
           '41384.00007961.TRK.rdls';
           '41384.00007962.TRK.rdls';
           '41384.00007963.TRK.rdls';
           };

name_file=cell(length(nm_file),1);
for i=1:length(nm_file)
    name_file{i,1}=fullfile(root_path,nm_file{i,1});
end
       
nF=4;%length(name_file);
info=cell(nF,1);
fits_info = cell(nF,1);
rawImg = cell(nF,1);
tableData = cell(nF,1);
stella = cell(nF,1);

%% Read files

for i=1:nF
    info{i,1} = fitsinfo(name_file{i,1});
    fits_info{i,1} = imfinfo(name_file{i,1});
    tableData{i,1} = fitsread(name_file{i,1},'binarytable','Info',info{i,1});
%     fitsdisp(name_file{i,1})
end

%% Coordinate stelle lette dal file risultato di SPD

indx=0;

% %deb_260
% stella{indx,1}=[
%             166.183088436775108,-2.492039656068517;
%             166.167240895569222,-2.488297292345185;
%             166.202455805194091,-2.453181226189948;
%             166.120294124377494,-2.440744748714962;
%             166.250115396026615,-2.426445512916378;
%             166.241043173050258,-2.416070425130488;
%             166.061090077730483,-2.402890537926634;
%             166.117524348064279,-2.400344748358740;
%             166.090111564848485,-2.380018771350147;
%             166.129994492221840,-2.372628083096713;
%             166.224386074280943,-2.347677783193941;
%             166.111472658610950,-2.339808814723067;
%             166.087888559033559,-2.321891940490584];    
    
% File 41384.00007958.TRK
indx=indx+1;
stella{indx,1}=[
            %Top left stars
            356.243534394062635,-1.835310697041982; %A
            356.661619319610850,-2.074765647720184; %B
            %Top Right
            357.638479250791477,-2.066235443032032; %C
            357.407656186252098,-1.315161725049868; %D
            %Botton left 
%             355.981118513323281,-0.587819206392723; %E
            356.102020682709963,-1.112002320688215;
            356.726755778870370,-0.983763426269217; %F
            %Botton Right
            357.347089637573333,-0.104830127631152; %G
            357.054225220686646,-0.775632907882328; %H
            %Inside polygon
            356.646720155966364,-1.522952664514025;
            ];
        
% File 41384.00007959.TRK
indx=indx+1;
stella{indx,1}=[
            %Top left stars
            356.756953165553455,-1.586778656890519; %A
            %Top Right
            358.660648392596499,-1.946418986750936; %B
            358.185934171144140,-1.015606518338745; %C
            %Botton Right
            358.344897278392580,-0.009935234907002; %D
            %Botton left
            357.620343232423409,-0.517632190068143; %E            
            357.309057452366517,-0.262715890675199; %F            
            %Inside polygon
            ];
        
% File 41384.00007960.TRK
indx=indx+1;
stella{indx,1}=[
            %Top left stars
            358.853046814477125,-2.110794830476918; %A
            358.843605665427788,-1.080006848934069; %B
            %Top Right
            357.640364043010493,-2.065922648176280; %C
            357.693595729476726,-1.742145423121909; %D
            %Botton Right
            358.989248174632394,-0.299957958319794; %E
            358.344872715444524,-0.009840935567643; %F
            %Botton left
            356.850749298600590,-0.271683947495023; %G
            357.409411276331809,-1.316055660869032; %H
            %Inside polygon            
            ];        

% File 41384.00007962.TRK
indx=indx+1;
stella{indx,1}=[
            %Top left stars
            358.662256479131941,-1.946247023890176; %A
            358.824924578306252,-1.572844516555741; %B
            %Top Right
            359.511822341403331,-1.979499795535894; %C
            359.629883344022119,-1.689131249067288; %D
            %Botton Right
            359.296729507755458,-0.650105965062850; %E
            359.296729507755458,-0.650105965062850; %F
            
            %Botton left
            358.647443100553346,-0.586364220973076; %G
            %Inside polygon
            359.489158504011698,-1.163561599662319            
            ];   
        
%% Compute difference

val = cell(nF,1);
cleanRes = cell(nF,1);
cleanRes2= cell(nF,1);

for j=1:nF
    for i=1:length(stella{j,1}(:,1))
        diff=[abs(tableData{j,1}{1,1}-stella{j,1}(i,1)), abs(tableData{j,1}{1,2}-stella{j,1}(i,2))];
        
        [rRA,cRA]=find(diff(:,1)==min(diff(:,1)));
        [rDEC,cDEC]=find(diff(:,2)==min(diff(:,2)));
        
        if ((rRA==rDEC) && (cRA==cDEC))
            val{j,1}(i,:)=[min(diff(:,1)),min(diff(:,2))];
            cleanRes{j,1}(i,:)=[tableData{j,1}{1,1}(rRA,cRA)-stella{j,1}(i,1),tableData{j,1}{1,2}(rRA,cRA)-stella{j,1}(i,2)];
            cleanRes2{j,1}(i,:)=[cleanRes{j,1}(i,1),cleanRes{j,1}(i,2)];
        else
            disp('NO');
            cleanRes2{j,1}(i,:)=[0,0];
        end
    end
    [r,c]=find(cleanRes{j,1}(:,1)==0);
    cleanRes{j,1}(r,:) = [];
end

%% Result
if nF>1
    res=vertcat(cleanRes{1,1},cleanRes{2,1});
else
    res=cleanRes{1,1};
end
for j=3:nF
    res=vertcat(res,cleanRes{j,1});
end

res=res.*3600;

str_0 = sprintf('Number of stars %d',length(res));
disp(str_0);

meanDiff = mean(res);%.*3600;
str_1 = sprintf('Mean error in arcsec RA=%f DEC=%f', meanDiff(1), meanDiff(2));
disp(str_1);

stdDiff = std(res);%.*3600;
str_2 = sprintf('Standard deviation error in arcsec RA=%f DEC=%f', stdDiff(1), stdDiff(2));
disp(str_2);

if PLOT
    if PLOT_TOT
    figure(1)
    subplot(1,2,1);
    % xbins1 = -0.2:0.1:0.15;
    set(gca,'FontSize',18);
    hist(res(:,1));
    xlabel('RA difference [arcsec]','FontSize',26,'FontWeight','bold');
    ylabel('Counts','FontSize',26,'FontWeight','bold');
    subplot(1,2,2);
    % xbins2 = -2.5:0.5:2.5;
    set(gca,'FontSize',18);
    hist(res(:,2));
    xlabel('DEC difference [arcsec]','FontSize',26,'FontWeight','bold');
    end
    
    if PLOT_SFIG
        for i=1:nF
            figure(i*10)
            subplot(1,2,1);
            % xbins1 = -0.2:0.1:0.15;
            set(gca,'FontSize',18);
            hist(cleanRes{i,1}(:,1).*3600);
            xlabel('RA difference [arcsec]','FontSize',26,'FontWeight','bold');
            ylabel('Counts','FontSize',26,'FontWeight','bold');
            subplot(1,2,2);
            % xbins2 = -2.5:0.5:2.5;
            set(gca,'FontSize',18);
            hist(cleanRes{i,1}(:,2).*3600);
            xlabel('DEC difference [arcsec]','FontSize',26,'FontWeight','bold');
        end
    end
end