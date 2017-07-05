format long;
clear all;
close all;
clc;

%% Iput files 

name_file={'C:\Users\Francesco Diprima\Desktop\SPD_result\fileRDLS\41384.00007944.TRK.rdls';
           'C:\Users\Francesco Diprima\Desktop\SPD_result\fileRDLS\deb_260.rdls'};

nF=length(name_file);       
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

% File 41384.00007944.TRK
stella{1,1}=[350.315259848932612,-0.188294909350993;
            350.859098296510751,-0.128354547759816;
            350.615304631909680,-0.247457979978040;
            350.588535571787020,-0.350757220952667;
            350.640109071041422,-0.726123782381036;
            350.669113746561607,-0.816396993485948;
            350.930289650033842,-0.885305773020347;
            350.844426459160275,-0.946942201021276;
            350.965690265228545,-1.051735123727294;
            350.941368279541109,-1.071166030017640;
            350.662441863931406,-1.152049920676465;
            350.106994417169233,-1.251991949999227;
            350.917232684072985,-1.188952392588972;
            351.011611394442980,-1.272714947424761;
            350.940706652983977,-1.318672487684441;
            350.683633521821378,-1.472268369910090;
            350.014376026265836,-1.604616223836642;
            350.382893314315311,-1.629012159927785;
            349.937800593620750,-1.709990465956337;
            350.827825096222057,-1.592780900752662;
            350.140603607888522,-1.725930864492507;
            350.748718086887493,-1.635245382794010;
            350.233215940395951,-1.835530020172313;
            350.517187669628640,-1.825618309871885;
            351.162282974640959,-1.741465788749631;
            351.104743570616449,-1.754950717939828;
            351.064043687553067,-1.781328617052294;
            350.262462373890344,-2.283028880781018];

stella{2,1}=[166.183088436775108,-2.492039656068517;
            166.167240895569222,-2.488297292345185;
            166.202455805194091,-2.453181226189948;
            166.120294124377494,-2.440744748714962;
            166.250115396026615,-2.426445512916378;
            166.241043173050258,-2.416070425130488;
            166.061090077730483,-2.402890537926634;
            166.117524348064279,-2.400344748358740;
            166.090111564848485,-2.380018771350147;
            166.129994492221840,-2.372628083096713;
            166.224386074280943,-2.347677783193941;
            166.111472658610950,-2.339808814723067;
            166.087888559033559,-2.321891940490584];    
    
%% Compute difference

val = cell(nF,1);
cleanRes = cell(nF,1);

for j=1:nF
    for i=1:length(stella{j,1}(:,1))
        diff=[abs(tableData{j,1}{1,1}-stella{j,1}(i,1)), abs(tableData{j,1}{1,2}-stella{j,1}(i,2))];
        
        [rRA,cRA]=find(diff(:,1)==min(diff(:,1)));
        [rDEC,cDEC]=find(diff(:,2)==min(diff(:,2)));
        
        if ((rRA==rDEC) && (cRA==cDEC))
            val{j,1}(i,:)=[min(diff(:,1)),min(diff(:,2))];
            cleanRes{j,1}(i,:)=[tableData{j,1}{1,1}(rRA,cRA)-stella{j,1}(i,1),tableData{j,1}{1,2}(rRA,cRA)-stella{j,1}(i,2)];
        else
            disp('NO');
        end
    end
    [r,c]=find(cleanRes{j,1}(:,1)==0);
    cleanRes{j,1}(r,:) = [];
end

%% Result

res=vertcat(cleanRes{1,1},cleanRes{2,1});

for j=3:nF
    res=vertcat(res,cleanRes{j,1});
end

str_0 = sprintf('Number of stars %d',length(res));
disp(str_0);

meanDiff = mean(res).*3600;
str_1 = sprintf('Mean error in arcsec RA=%f DEC=%f', meanDiff(1), meanDiff(2));
disp(str_1);

stdDiff = std(res).*3600;
str_2 = sprintf('Standard deviation error in arcsec RA=%f DEC=%f', stdDiff(1), stdDiff(2));
disp(str_2);




