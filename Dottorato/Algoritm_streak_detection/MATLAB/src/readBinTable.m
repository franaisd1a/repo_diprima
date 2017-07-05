format long;
clear all;
close all;
clc;

% name_file='C:\Users\Francesco Diprima\Desktop\index-4203-25.fits';

name_file={'C:\Users\Francesco Diprima\Desktop\SPD_result\astroSolve\41384.00007944.TRK-indx.xyls';
           'C:\Users\Francesco Diprima\Desktop\SPD_result\astroSolve\41384.00007944.TRK.rdls';
           'C:\Users\Francesco Diprima\Desktop\SPD_result\astroSolve\41384.00007944.TRK.axy';
           'C:\Users\Francesco Diprima\Desktop\SPD_result\astroSolve\41384.00007944.TRK.match'};

info=cell(4,1);
fits_info = cell(4,1);
rawImg = cell(4,1);
tableData = cell(4,1);

for i=1:4
    % rawImg = (fitsread(name_file));
    info{i,1} = fitsinfo(name_file{i,1});
    fits_info{i,1} = imfinfo(name_file{i,1});
    % rawImg = (fitsread(name_file));
    % [filename,extension,index,raw, info, pixelRegion, tableColumns, tableRows] = parseInputs(name_file);
    %
    % rawImg = fitsread(name_file,'primary');
    % rawImg = fitsread(name_file,'asciitable');
    % rawImg = fitsread(name_file,'binarytable');
    
    % rowend = info.BinaryTable.Rows;
    
    rawImg{i,1} = fitsread(name_file{i,1},'primary','Info',info{i,1});
    tableData{i,1} = fitsread(name_file{i,1},'binarytable','Info',info{i,1});
    fitsdisp(name_file{i,1})
end

%Coordinate stelle lette dal file risultato di SPD
stella=[350.315259848932612,-0.188294909350993;
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

%Result
val=zeros(size(stella));
cleanRes=zeros(size(stella));
cleanRes2=zeros(size(stella));

for i=1:length(stella(:,1))
    diff=[abs(tableData{2,1}{1,1}-stella(i,1)), abs(tableData{2,1}{1,2}-stella(i,2))];
    
    [rRA,cRA]=find(diff(:,1)==min(diff(:,1)));
    [rDEC,cDEC]=find(diff(:,2)==min(diff(:,2)));
    
    if ((rRA==rDEC) && (cRA==cDEC))
        val(i,:)=[min(diff(:,1)),min(diff(:,2))];
        cleanRes(i,:)=val(i,:);
        cleanRes2(i,:)=[tableData{2,1}{1,1}(rRA,cRA)-stella(i,1),tableData{2,1}{1,2}(rRA,cRA)-stella(i,2)];
    else
        disp('NO');
        val(i,:)=[NaN,NaN];
    end
end

[r,c]=find(cleanRes2(:,1)==0);

cleanRes2(r,:) = [];

meanDiff=mean(cleanRes2).*3600
stdDiff=std(cleanRes2).*3600
