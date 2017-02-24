function net = cnn_cifar_init(varargin) 
opts.networkType = 'simplenn' ; 
opts = vl_argparse(opts, varargin) ; 
 
lr = [.1 2] ; 
k = 1;
% Define network CIFAR10-quick 
net.layers = {} ; 
 
% Block 1 
net.layers{end+1} = struct('type', 'conv', ... 
                           'weights', {{0.01*randn(5,5,3,k*32, 'single'), zeros(1, k*32, 'single')}}, ... 
                           'learningRate', lr, ... 
                           'stride', 1, ... 
                           'pad', 2) ; 
net.layers{end+1} = struct('type', 'birelu') ;
block = 32;
net = addAutoEnc(net,2,block,lr,k);
net.layers{end+1} = struct('type', 'conv', ... 
                           'weights', {{0.1*randn(3,3,k*32,k*32, 'single'), zeros(1, k*32, 'single')}}, ... 
                           'learningRate', lr, ... 
                           'stride', 2, ... 
                           'pad', [0 1 0 1]) ; 
net.layers{end+1} = struct('type', 'birelu') ;

% net.layers{end+1} = struct('type', 'pool', ... 
%                            'method', 'avg', ... 
%                            'pool', [3 3], ... 
%                            'stride', 2, ... 
%                            'pad', [0 1 0 1]) ; 

 
% Block 2 
net.layers{end+1} = struct('type', 'conv', ... 
                           'weights', {{0.05*randn(5,5,k*32,k*32, 'single'), zeros(1,k*32,'single')}}, ... 
                           'learningRate', lr, ... 
                           'stride', 1, ... 
                           'pad', 2) ; 
                      
net.layers{end+1} = struct('type', 'birelu') ; 
block = 32;
net = addAutoEnc(net,2,block,lr,k);
net.layers{end+1} = struct('type', 'conv', ... 
                           'weights', {{0.1*randn(3,3,k*32,k*64, 'single'), zeros(1, k*64, 'single')}}, ... 
                           'learningRate', lr, ... 
                           'stride', 2, ... 
                           'pad', [0 1 0 1]) ; 
net.layers{end+1} = struct('type', 'birelu') ;
% 
% net.layers{end+1} = struct('type', 'pool', ... 
%                            'method', 'max', ... 
%                            'pool', [3 3], ... 
%                            'stride', 2, ... 
%                            'pad', [0 1 0 1]) ; % Emulate caffe 
 
% Block 3 
net.layers{end+1} = struct('type', 'conv', ... 
                           'weights', {{0.05*randn(5,5,k*64,k*64, 'single'), zeros(1,k*64,'single')}}, ... 
                           'learningRate', lr, ... 
                           'stride', 1, ... 
                           'pad', 2) ; 
net.layers{end+1} = struct('type', 'birelu') ;
block = 64;
net = addAutoEnc(net,2,block,lr,k);
% net.layers{end+1} = struct('type', 'pool', ... 
%                            'method', 'max', ... 
%                            'pool', [3 3], ... 
%                            'stride', 2, ... 
%                            'pad', [0 1 0 1]) ; % Emulate caffe 
net.layers{end+1} = struct('type', 'conv', ... 
                           'weights', {{0.1*randn(3,3,k*64,k*64, 'single'), zeros(1, k*64, 'single')}}, ... 
                           'learningRate', lr, ... 
                           'stride', 2, ... 
                           'pad', [0 1 0 1]) ; 
net.layers{end+1} = struct('type', 'birelu') ; 
% Block 4 
net.layers{end+1} = struct('type', 'conv', ... 
                           'weights', {{0.05*randn(4,4,k*64,k*64, 'single'), zeros(1,k*64,'single')}}, ... 
                           'learningRate', lr, ... 
                           'stride', 1, ... 
                           'pad', 0) ; 
net.layers{end+1} = struct('type', 'birelu') ; 
block = 64;
 net = addAutoEnc(net,2,block,lr,k);
% Block 5 
net.layers{end+1} = struct('type', 'conv', ... 
                           'weights', {{0.05*randn(1,1,k*64,10, 'single'), zeros(1,10,'single')}}, ... 
                           'learningRate', .1*lr, ... 
                           'stride', 1, ... 
                           'pad', 0) ; 
 
% Loss layer 
net.layers{end+1} = struct('type', 'softmaxloss') ; 
 
% Meta parameters 
net.meta.inputSize = [32 32 3] ; 
net.meta.trainOpts.learningRate = [0.05*ones(1,50) 0.005*ones(1,30) 0.0005*ones(1,20)] ; 
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