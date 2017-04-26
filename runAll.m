function overAll = runAll(varargin)
%% prenet , continue, override,expname are must haves
opts.prenet = [];
opts.continue = false;
opts.override = false;
opts.folder = [];
opts.expname = 'TestRun';
opts.keepall = true;
opts = vl_argparse(opts,varargin);
if nargin ==0
    cont = false;
end
if nargin<3
    override = true
end
%expName = 'new_freluM4fixedLayers';
expdir = fullfile('.','results',opts.folder,opts.expname);
if ~exist(expdir) || opts.continue||opts.override
    unix(['mkdir ' ,expdir]);
    unix(['cp cnn_cifar_init.m ',expdir]);
    if isempty(opts.prenet)
        warning('prenet not loaded')
    else
        warning('prenet loaded------------------')
    end
    [prenet,info] = cnn_cifar(opts.prenet,'train',struct('gpus',1),'expDir','./results/CURRENT','continue',opts.continue);
    if opts.keepall
    unix(['cp ./results/CURRENT/net-epoch-*.mat ',expdir]);
    else
        netpath = ['./results/CURRENT/net-epoch-',int2str(45),'.mat'];
        unix(['cp ', netpath,' ' ,expdir]);
    end  
    %unix(['cp ./layers/em_nnloss.m ',expdir]);
    unix(['cp ./results/CURRENT/net-train.pdf ',expdir]);
else
    error('Directory Exists !!!!')
end

end

