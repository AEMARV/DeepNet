expName = 'Rec0';
expdir = ['./results/exp',expName];
if ~exist(expdir) | false
unix(['mkdir ' ,expdir]);
unix(['cp cnn_cifar_init.m ',expdir]);
else
    error('Directory Exists')
end
prenet = [];
alltop1 = [];
for i = 1:100
    
[prenet,info] = cnn_cifar(prenet,'train',struct('gpus',1),'expDir','./results/CURRENT','continue',false);
unix(['cp ./results/CURRENT/net-epoch-30.mat ',expdir,'/iteration_',int2str(i),'.mat']);
%unix(['cp ./layers/em_nnloss.m ',expdir]);
unix(['cp ./results/CURRENT/net-train.pdf ',expdir]);
alltop1 = cat(2,alltop1,cat(1,info.val.top1err));
end
save([expdir,'/','alltop1'],'valtop1');