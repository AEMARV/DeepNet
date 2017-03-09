function overAll = runAll(prenet,cont,override)
if nargin ==0
    cont = false;
end
if nargin<3
    override = false
end
expName = 'new_freluM4fixedLayers';
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
    unix(['cp ./results/CURRENT/net-epoch-45.mat ',expdir]);
    %unix(['cp ./layers/em_nnloss.m ',expdir]);
    unix(['cp ./results/CURRENT/net-train.pdf ',expdir]);
else
    error('Directory Exists !!!!')
end

end

