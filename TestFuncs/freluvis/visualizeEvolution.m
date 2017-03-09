function [] = visualizeEvolution(netPath,epochNum,ParNum)
if isempty(netPath)
    netPath = './results/CURRENT/'
end
if isempty(ParNum)
    ParNum = [];
end
for i = 1: epochNum
    load([netPath,'net-epoch-',int2str(i),'.mat']);
    visulaizeActivationParam(net,ParNum);
    drawnow;
    pause(0.05);
end
end