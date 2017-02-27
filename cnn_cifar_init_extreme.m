function net = cnn_cifar_init(decArr,encArr,varargin) 
opts.networkType = 'simplenn' ; 
opts = vl_argparse(opts, varargin) ; 
if isempty(decArr)
decArr = [2,2,2,2,2,2,0]; 
encArr = [-inf,0,-inf,0,-inf,0,-inf];
end
lr = [.1 2] ; 
k =1;
Initial = @eyeconv;
% Define network CIFAR10-quick 
net.layers = {} ; 
FiltNum = 32; 
blockNum = 1;
block = k*32;
decrec = 3;
encrec = 1;
% Block 1 
net.layers{end+1} = struct('type', 'conv', ... 
                           'weights', {{Initial(5,5,3,block, 'single'), zeros(1, block, 'single')}}, ... 
                           'learningRate', lr, ... 
                           'stride', 1, ... 
                           'blocks', 1, ...
                           'pad', 2) ; 
net.layers{end+1} = struct('type', 'birelu','scatter',false) ;
blockNum =2;
dec = decArr(1);
enc = encArr(1);
%% Auto 1
[net,blockNum,block] = addAutoEnc(net,dec,enc,block,lr,blockNum,decrec,encrec);
%%
net.layers{end+1} = struct('type', 'pool', ... 
                           'method', 'avg', ... 
                           'pool', [3 3], ... 
                           'stride', 2, ... 
                           'pad', [0 1 0 1]) ; 
dec = decArr(2);
enc =encArr(2);
%% Auto 2
[net,blockNum,block] = addAutoEnc(net,dec,enc,block,lr,blockNum,decrec,encrec);
%%
NewBlockSize = 64*k;

% Block 2 
net.layers{end+1} = struct('type', 'conv', ... 
                           'weights', {{Initial(5,5,block,NewBlockSize*blockNum, 'single'), zeros(1,NewBlockSize*blockNum,'single')}}, ... 
                           'learningRate', lr, ... 
                           'stride', 1, ... 
                           'blocks', blockNum, ...
                           'pad', 2) ; 
                       
block = NewBlockSize;
net.layers{end+1} = struct('type', 'birelu','block',block,'scatter',true) ; 
blockNum = blockNum *2;

dec = decArr(3);
enc = encArr(3);

%% Auto 3
[net,blockNum,block] = addAutoEnc(net,dec,enc,block,lr,blockNum,decrec,encrec);
%%



net.layers{end+1} = struct('type', 'pool', ... 
                           'method', 'avg', ... 
                           'pool', [3 3], ... 
                           'stride', 2, ... 
                           'pad', [0 1 0 1]) ; % Emulate caffe 
 dec = decArr(4);
enc = encArr(4);
%% Auto 4
[net,blockNum,block] = addAutoEnc(net,dec,enc,block,lr,blockNum,decrec,encrec);
%%
% Block 3 
NewBlockSize = 64*k;

<<<<<<< HEAD
% Loss layer
net.layers{end+1} = struct('type', 'revloss') ;
=======
net.layers{end+1} = struct('type', 'conv', ... 
                           'weights', {{Initial(5,5,block,blockNum*NewBlockSize, 'single'), zeros(1,blockNum*NewBlockSize,'single')}}, ... 
                           'learningRate', lr, ... 
                           'stride', 1, ... 
                           'blocks', blockNum, ...
                           'pad', 2) ; 
                       
block = NewBlockSize;
net.layers{end+1} = struct('type', 'birelu','block',block,'scatter',true) ;
blockNum = blockNum *2;
>>>>>>> BRelu

dec = decArr(5);
enc = encArr(5);

%% Auto 5
[net,blockNum,block] = addAutoEnc(net,dec,enc,block,lr,blockNum,decrec,encrec);
%%

net.layers{end+1} = struct('type', 'pool', ... 
                           'method', 'avg', ... 
                           'pool', [3 3], ... 
                           'stride', 2, ... 
                           'pad', [0 1 0 1]) ; % Emulate caffe 
dec = decArr(6);
enc = encArr(6);
%% Auto 6
[net,blockNum,block] = addAutoEnc(net,dec,enc,block,lr,blockNum,decrec,encrec);                       
 %%
% Block 4 
NewBlockSize =64*k;
net.layers{end+1} = struct('type', 'conv', ... 
                           'weights', {{Initial(4,4,block,NewBlockSize*blockNum, 'single'), zeros(1,NewBlockSize*blockNum,'single')}}, ... 
                           'learningRate', lr, ... 
                           'stride', 1, ... 
                           'blocks', blockNum, ...
                           'pad', 0) ; 
block = NewBlockSize;
net.layers{end+1} = struct('type', 'birelu','block',block,'scatter',true) ; 
blockNum = blockNum *2;
dec = decArr(7);
enc =encArr(7);

%% Auto 7
[net,blockNum,block] = addAutoEnc(net,dec,enc,block,lr,blockNum,1,1);
%%
% Block 5 
net.layers{end+1} = struct('type', 'conv', ... 
                           'weights', {{Initial(1,1,block*blockNum,10, 'single'), zeros(1,10,'single')}}, ... 
                           'learningRate', .1*lr, ... 
                           'stride', 1, ... 
                           'blocks', blockNum, ...
                           'pad', 0) ; 
%net.layers{end+1} = struct('type', 'avr') ; 
 
% Loss layer 
net.layers{end+1} = struct('type', 'softmaxloss') ; 
 
% Meta parameters 
net.meta.inputSize = [32 32 3] ; 
net.meta.trainOpts.learningRate = [0.05*ones(1,30) 0.005*ones(1,10) 0.0005*ones(1,05)] ; 
net.meta.trainOpts.weightDecay = 0.0001 ; 
net.meta.trainOpts.batchSize = 100 ; 
net.meta.trainOpts.numEpochs = numel(net.meta.trainOpts.learningRate) ; 
 
% Fill in default values 
net = vl_simplenn_tidy(net) ; 
 
% Switch to DagNN if requested 
switch lower(opts.networkType) 
  case 'simplenn' 
    % done 
  case 'dagnn' 
    net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ; 
    net.addLayer('error', dagnn.Loss('loss', 'classerror'), ... 
             {'prediction','label'}, 'error') ; 
  otherwise 
    assert(false) ; 
end 