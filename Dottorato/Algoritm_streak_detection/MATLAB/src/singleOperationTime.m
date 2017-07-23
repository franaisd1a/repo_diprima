close all;
clear all;
clc;

dim={  '128X128';
       '256X256';
       '512X512';
       '1024X1024';
       '2048X2048';
       '4096X4096'};
   
%% Time convolution

%       Up          Comp        Down        
tConv=[0.0002637	0.0000782	0.0002056;
        0.0004102	0.0000977	0.0002627;
        0.0011869	0.0000997	0.0004687;
        0.001922	0.0001455	0.0020895;
        0.0084622	0.0002009	0.0119381;
        0.01852     0.0002502	0.049374
];


timeConv=tConv*10^3;

timeConv(:,2)=timeConv(:,2)*2;

figure('Name','Conolution filter')
bar(timeConv,'stacked');
set(gca,'XTickLabel',dim);
set(gca,'FontSize',16);
% title('Conolution filter','FontWeight','bold','FontSize',32);
ylabel('Time [ms]','FontSize',28,'FontWeight','bold');
xlabel('Image dimension [pixel]','FontSize',20,'FontWeight','bold');
legend('Upload','Computation','Download','FontSize',30)
grid on;

%% Time median
            %Up         Comp        Down 
tMedian=[
            0.0003081	0.0001115	0.0002674;
            0.0004652	0.000133	0.0004391;
            0.0013685	0.0001661	0.0013303;
            0.0046586	0.0001908	0.0074325;
            0.0053527	0.0002374	0.0179277;
            0.0185173	0.0002682	0.049374
];

timeMedian=tMedian.*10^3;

timeMedian(:,2)=timeMedian(:,2)*2;

figure('Name','Median filter')
bar(timeMedian,'stacked');
set(gca,'XTickLabel',dim);
set(gca,'FontSize',16);
% title('Execution time','FontWeight','bold','FontSize',32);
ylabel('Time [ms]','FontSize',28,'FontWeight','bold');
xlabel('Image dimension [pixel]','FontSize',20,'FontWeight','bold');
legend('Upload','Computation','Download','FontSize',30)
grid on;

%% Time hist

%           Up          Comp        Down  
tHist = [
            0.0007198	0.0000948	0.000513;
            0.0009082	0.0001137	0.0006742;
            0.0013066	0.0001442	0.0005126;
            0.0044947	0.0001694	0.0006601;
            0.011458	0.0002035	0.0008507;
            0.039123	0.0002963	0.000853
];

timeHist=tHist(:,1:end)*10^3;

%timeConv(:,2)=timeConv(:,2)*2;

figure('Name','Histogram')
bar(timeHist,'stacked');
set(gca,'XTickLabel',dim);
set(gca,'FontSize',16);
% title('Execution time','FontWeight','bold','FontSize',32);
ylabel('Time [ms]','FontSize',28,'FontWeight','bold');
xlabel('Image dimension [pixel]','FontSize',20,'FontWeight','bold');
legend('Upload','Computation','Download','FontSize',30)
grid on;

