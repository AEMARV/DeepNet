function net = cnn_cifar_init(varargin) 
opts.networkType = 'simplenn' ; 
opts = vl_argparse(opts, varargin) ; 
 
lr = [.1 2] ; 
k =1;
% Define network CIFAR10-quick 
net.layers = {} ; 
FiltNum = 32; 
blockNum = 1;
block = k*32;
% Block 1 
net.layers{end+1} = struct('type', 'conv', ... 
                           'weights', {{0.01*randn(5,5,3,block, 'single'), zeros(1, block, 'single')}}, ... 
                           'learningRate', lr, ... 
                           'stride', 1, ... 
                           'pad', 2) ; 
net.layers{end+1} = struct('type', 'birelu','scatter',false) ;
blockNum =2;
dec = 2;
enc = 0;
[net,blockNum] = addAutoEnc(net,dec,enc,block,lr,blockNum);

net.layers{end+1} = struct('type', 'pool', ... 
                           'method', 'avg', ... 
                           'pool', [3 3], ... 
                           'stride', 2, ... 
                           'pad', [0 1 0 1]) ; 

NewBlockSize = 120*k;

% Block 2 
net.layers{end+1} = struct('type', 'conv', ... 
                           'weights', {{0.05*randn(5,5,block,NewBlockSize*blockNum, 'single'), zeros(1,NewBlockSize*blockNum,'single')}}, ... 
                           'learningRate', lr, ... 
                           'stride', 1, ... 
                           'pad', 2) ; 
                       
block = NewBlockSize;
net.layers{end+1} = struct('type', 'birelu','block',block,'scatter',true) ; 
blockNum = blockNum *2;

dec = 0;
enc = -inf;


[net,blockNum] = addAutoEnc(net,dec,enc,block,lr,blockNum);
dec = 2;
enc = 0;

[net,blockNum] = addAutoEnc(net,dec,enc,block,lr,blockNum);


net.layers{end+1} = struct('type', 'pool', ... 
                           'method', 'avg', ... 
                           'pool', [3 3], ... 
                           'stride', 2, ... 
                           'pad', [0 1 0 1]) ; % Emulate caffe 
 
% Block 3 
NewBlockSize = 64*k;

net.layers{end+1} = struct('type', 'conv', ... 
                           'weights', {{0.05*randn(5,5,block,blockNum*NewBlockSize, 'single'), zeros(1,blockNum*NewBlockSize,'single')}}, ... 
                           'learningRate', lr, ... 
                           'stride', 1, ... 
                           'pad', 2) ; 
                       
block = NewBlockSize;
net.layers{end+1} = struct('type', 'birelu','block',block,'scatter',true) ;
blockNum = blockNum *2;

dec = 2;
enc = -inf;


[net,blockNum] = addAutoEnc(net,dec,enc,block,lr,blockNum);
dec = 2;
enc = 0;

[net,blockNum] = addAutoEnc(net,dec,enc,block,lr,blockNum);

net.layers{end+1} = struct('type', 'pool', ... 
                           'method', 'avg', ... 
                           'pool', [3 3], ... 
                           'stride', 2, ... 
                           'pad', [0 1 0 1]) ; % Emulate caffe 
 
% Block 4 
NewBlockSize =64*k;
net.layers{end+1} = struct('type', 'conv', ... 
                           'weights', {{0.05*randn(4,4,block,NewBlockSize*blockNum, 'single'), zeros(1,NewBlockSize*blockNum,'single')}}, ... 
                           'learningRate', lr, ... 
                           'stride', 1, ... 
                           'pad', 0) ; 
block = NewBlockSize;
net.layers{end+1} = struct('type', 'birelu','block',block,'scatter',true) ; 
blockNum = blockNum *2;
dec = 2;
enc = -inf;
[net,blockNum] = addAutoEnc(net,dec,enc,block,lr,blockNum);
% Block 5 
net.layers{end+1} = struct('type', 'conv', ... 
                           'weights', {{0.05*randn(1,1,block*blockNum,10, 'single'), zeros(1,10,'single')}}, ... 
                           'learningRate', .1*lr, ... 
                           'stride', 1, ... 
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