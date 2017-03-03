function overAll = runAll(prenet,cont,override)
if nargin ==0 
    cont = false;
end
if nargin<3
    override = false
end
decArr = [1,1,1,1,1,0,0];
encArr = [0,-inf,-inf,0,1,-inf,1];

for i = 1:100
expName = 'new_fBreluOnRec0';
expdir = ['./results/exp',expName];
if ~exist(expdir) || cont||override
unix(['mkdir ' ,expdir]);
unix(['cp cnn_cifar_init.m ',expdir]);
if isempty(prenet)
    warning('prenet not loaded')
else
    warning('prenet loaded------------------')
end
[prenet,info] = cnn_cifar(prenet,'train',struct('gpus',1),'expDir','./results/CURRENT','continue',cont);
unix(['cp ./results/CURRENT/net-epoch-20.mat ',expdir]);
%unix(['cp ./layers/em_nnloss.m ',expdir]);
unix(['cp ./results/CURRENT/net-train.pdf ',expdir]);
else
    error('Directory Exists !!!!')
end

end
end
