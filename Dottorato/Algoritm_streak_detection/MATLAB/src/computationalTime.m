clear all;
close all;
clc;

%% Read xls file

file = 'C:\Users\Francesco Diprima\Desktop\SPD_result\tabellaRes.xlsx';

operationSheet = 1;
val = xlsread(file, operationSheet);

operation= {'Astrometry ';
            'Open and read file ';
            'Histogram Stretching ';
            'Median filter ';
            'Background estimation ';
            'Background subtraction ';
            'Median filter ';
            'Binarization for points detection ';
            'Binarization for streaks detection ';
            'Distance transformation ';
            'Convolution for points detection ';
            'Morphology opening ';
            'Hough transform ';
            'Sum remove streaks binary ';
            'Subtract image ';
            'Convolution ';
            'Connected components ';
            'Somma valori';
            'Totale misurato'};

thread={  'CPU';
        'GPU 1';
        'GPU 2';
        'GPU 4';
        'GPU 8';
        'GPU 16';
        'GPU 32'};

dimensionSheet = 2;
dimVal = xlsread(file, dimensionSheet);


%% Plot operation result
if 0
for i=1:2%length(operation)
    figure(i+100);
    plot(val(i,:));
    title(operation(i),'FontWeight','bold','FontSize',14);
    ylabel('Time [s]','FontSize',10);
%     xlabel('X (km)','FontSize',12,'FontWeight','bold');
    set(gca,'XTickLabel',thread,'FontSize',10);
    grid on;

    figure(i+200);
    bar(val(i,:));
    title(operation(i),'FontWeight','bold','FontSize',14);
    ylabel('Time [s]','FontSize',10);
    set(gca,'XTickLabel',thread,'FontSize',10);
    grid on;
end
end
%% Plot dimension result

figure(1);
plot(dimVal(:,1),dimVal(:,2),'-s',dimVal(:,1),dimVal(:,3),'-o','LineWidth',2);
set(gca,'FontSize',26);
% title('Execution time','FontWeight','bold','FontSize',32);
ylabel('Time [s]','FontSize',28,'FontWeight','bold');
xlabel('Side of sqare image [pixel]','FontSize',28,'FontWeight','bold');
legend('CPU','GPU','FontSize',30)
grid on;


figure(2);
plot(dimVal(:,1),dimVal(:,4),'-s','LineWidth',2);
set(gca,'FontSize',26);
% title('Speed-up','FontWeight','bold','FontSize',32);
ylabel('Speed-up','FontSize',28,'FontWeight','bold');
xlabel('Side of sqare image [pixel]','FontSize',28,'FontWeight','bold');
grid on;



