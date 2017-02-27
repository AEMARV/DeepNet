function runAll(cont)
if nargin ==0 
    cont = false;
end
decArr = [1,1,1,1,1,0,0];
encArr = [0,-inf,-inf,0,1,-inf,1];

for i = 0
expdir = ['./results/exp',int2str(i)];
if ~exist(expdir) || i ==0
unix(['mkdir ' ,expdir]);
unix(['cp cnn_cifar_init.m ./results/exp',int2str(i),'/']);
cnn_cifar('train',struct('gpus',1),'expDir','./results','continue',cont);
unix(['cp ./results/net*.mat ',expdir]);
unix(['cp ./results/net-train.pdf ',expdir]);
else
    error('Directory Exists !!!!')
end

end
end