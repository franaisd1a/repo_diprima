% name_file='C:\Users\Francesco Diprima\Desktop\index-4203-25.fits';

name_file={'C:\Users\Francesco Diprima\Desktop\astroSolve\41384.00007944.TRK-indx.xyls';
'C:\Users\Francesco Diprima\Desktop\astroSolve\41384.00007944.TRK.rdls';
'C:\Users\Francesco Diprima\Desktop\astroSolve\41384.00007944.TRK.axy';
'C:\Users\Francesco Diprima\Desktop\astroSolve\41384.00007944.TRK.match'};

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


stella=[350.315259848932612,-0.188294909350993];

diff=[abs(tableData{2,1}{1,1}-stella(1,1)), abs(tableData{2,1}{1,2}-stella(1,2))];

[rRA,cRA]=find(diff(:,1)==min(diff(:,1)));
[rDEC,cDEC]=find(diff(:,2)==min(diff(:,2)));

if ((rRA==rDEC) && (cRA==cDEC))
    disp('OK');
    val=[min(diff(:,1)),min(diff(:,2))]
end


